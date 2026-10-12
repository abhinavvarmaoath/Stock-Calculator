import math
import re

import numpy as np
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, jsonify

import agent
import forecast
import limiter
import prices
import sim
from calc import calc

load_dotenv()       # the assistant's key and model live in .env

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024        # the assistant takes a chat, never anything big

# key, label, lowest, can it be the lowest, highest, required
calcfields = [
    ('allotment', 'Number of shares', 0, False, 1e9, True),
    ('iniprice', 'Initial share price', 0, False, 1e7, True),
    ('fnlprice', 'Final share price', 0, True, 1e7, True),
    ('buycmm', 'Buy commission', 0, True, 1e7, True),
    ('sellcmm', 'Sell commission', 0, True, 1e7, True),
    ('cptlgain', 'Capital gain tax rate', 0, True, 100, True),
]

# same trade fields minus the final price, plus the simulator ones
simfields = [f for f in calcfields if f[0] != 'fnlprice'] + [
    ('drift', 'Expected yearly return', -100, True, 200, True),
    ('curprice', 'Current price', 0, False, 1e7, False),
    ('vol', 'Yearly volatility', 0, True, 500, False),
]

horizons = {21: '1 month', 63: '3 months', 126: '6 months', 252: '1 year'}

forecasts = limiter.Limiter(6, 600)     # new forecasts one address can ask for in 10 minutes. each takes a while
maxprices = 4000                        # pasted closing prices, about 16 years of days


@app.template_filter()
def money(x):
    x = round(float(x), 2)
    return ('-' if x < 0 else '') + '${:,.2f}'.format(abs(x))


@app.template_filter()
def pct(x):
    return '{:.2f}%'.format(float(x))


@app.template_filter()
def signed(x):
    return '{:+.1f}%'.format(float(x))


@app.template_filter()
def luck(p):
    return '<0.001' if p < 0.001 else '{:.3f}'.format(p)


def getnum(form, key, label, low, lowok, high, need, errs):
    raw = form.get(key, '').strip().replace(',', '')
    if raw == '':
        if need:
            errs.append(label + ' is required')
        return None
    try:
        val = float(raw)
    except ValueError:
        val = math.nan
    if not math.isfinite(val):
        errs.append(label + ' must be a number')
        return None
    if val < low or (val == low and not lowok):
        errs.append('{} must be {} {:g}'.format(label, 'at least' if lowok else 'more than', low))
        return None
    if val > high:
        errs.append('{} must be {:,.10g} or less'.format(label, high))
        return None
    return val


def getfields(form, fields, errs):
    vals = {}
    for key, label, low, lowok, high, need in fields:
        vals[key] = getnum(form, key, label, low, lowok, high, need, errs)
    return vals


def getsym(form, errs):
    sym = form.get('stksymbol', '').strip().upper()
    if not re.fullmatch(r'[A-Z0-9.\-]{1,10}', sym):
        errs.append('Stock symbol must be 1 to 10 letters or numbers, like AAPL or BRK-B')
        return None
    return sym


def getprices(form, errs):
    # closing prices pasted in, oldest first, one a line. None when the box is empty.
    # commas only count as thousands separators, because "1,234" read as two prices would
    # change the numbers without anyone noticing
    text = form.get('closes', '').strip()
    if not text:
        return None
    out = []
    for tok in text.replace(';', ' ').split():
        raw = tok.lstrip('$')
        if not re.fullmatch(r'\d+(?:\.\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?', raw):
            errs.append("Couldn't read '{}' as a price. Paste just the closing prices, one on each line, oldest first.".format(tok[:20]))
            return None
        v = float(raw.replace(',', ''))
        if v <= 0:
            errs.append('Prices have to be above 0')
            return None
        out.append(v)
        if len(out) > maxprices:
            errs.append('Paste at most {:,} prices'.format(maxprices))
            return None
    return out


@app.route('/')
def index():
    return render_template('index.html', form={}, errs=[])


@app.route('/post', methods=['GET', 'POST'])
def report():
    if request.method == 'GET':
        return redirect(url_for('index'))

    errs = []
    stksymbol = getsym(request.form, errs)
    vals = getfields(request.form, calcfields, errs)
    if errs:
        return render_template('index.html', form=request.form, errs=errs), 400

    res = calc(**vals)

    # so the report can link to the simulator with the same trade filled in
    args = {k: '{:.10g}'.format(vals[k]) for k in ('allotment', 'iniprice', 'buycmm', 'sellcmm', 'cptlgain')}
    args['stksymbol'] = stksymbol
    return render_template('index2.html', stksymbol=stksymbol, args=args, **res)


def simerror(errs, openadv=False):
    return render_template('sim.html', form=request.form, errs=errs, horizons=horizons, openadv=openadv), 400


@app.route('/sim', methods=['GET', 'POST'])
def simulate():
    if request.method == 'GET':
        return render_template('sim.html', form=request.args, errs=[], horizons=horizons)

    errs = []
    stksymbol = getsym(request.form, errs)
    vals = getfields(request.form, simfields, errs)
    try:
        ndays = int(request.form.get('ndays', ''))
    except ValueError:
        ndays = 0
    if ndays not in horizons:
        errs.append('Pick how long to hold for')
    method = request.form.get('method', 'normal')
    if method not in ('normal', 'resample'):
        errs.append('Pick a model from the list')
    if errs:
        return simerror(errs)

    # only goes and gets prices for what wasn't typed in
    try:
        curprice, vol, hist, src = prices.resolve(stksymbol, vals['curprice'],
                                                  None if vals['vol'] is None else vals['vol'] / 100,
                                                  method == 'resample')
    except prices.PriceError as e:
        return simerror([str(e)], True)    # the message points at Advanced, so open it

    res = sim.run(curprice, vol, vals['drift'] / 100, ndays, vals['allotment'], vals['iniprice'],
                  vals['buycmm'], vals['sellcmm'], vals['cptlgain'], hist=hist)

    # a handful of days for the table view
    fan = res['fan']
    step = max(1, ndays // 6)
    fanrows = [dict(day=d, p5=fan['p5'][d], p25=fan['p25'][d], p50=fan['p50'][d],
                    p75=fan['p75'][d], p95=fan['p95'][d])
               for d in sorted(set(range(0, ndays + 1, step)) | {ndays})]

    e, c = res['bins']['edges'], res['bins']['counts']
    binrows = [dict(lo=e[i], hi=e[i + 1], share=c[i] * 100 / res['npaths']) for i in range(len(c))]

    chartdata = dict(fan=fan, bins=res['bins'], brkeven=res['brkeven'], npaths=res['npaths'])

    # so "change inputs" comes back with the form filled in
    args = {k: '{:.10g}'.format(v) for k, v in vals.items() if v is not None}
    args.update(stksymbol=stksymbol, ndays=ndays, method=method)

    return render_template('sim2.html', res=res, stksymbol=stksymbol, horizon=horizons[ndays],
                           method=method, src=src, fanrows=fanrows, binrows=binrows,
                           chartdata=chartdata, args=args)


def scoreerror(errs, code=400, openadv=False):
    return render_template('forecast.html', form=request.form, errs=errs, horizons=forecast.horizons,
                           openadv=openadv), code


@app.route('/forecast', methods=['GET', 'POST'])
def scoreboard():
    if request.method == 'GET':
        return render_template('forecast.html', form=request.args, errs=[], horizons=forecast.horizons)

    errs = []
    closes = getprices(request.form, errs)
    if closes is not None and not request.form.get('stksymbol', '').strip():
        label = 'your prices'
    else:
        label = getsym(request.form, errs)
    try:
        h = int(request.form.get('h', ''))
    except ValueError:
        h = 0
    if h not in forecast.horizons:
        errs.append('Pick how far ahead to forecast')
    if errs:
        return scoreerror(errs, openadv=bool(request.form.get('closes')))

    if closes is None:
        try:
            closes = prices.history(label, forecast.years)
        except prices.PriceError as e:
            return scoreerror([e.what + ' Try again, or paste the closing prices under Advanced.'], openadv=True)
    lp = np.log(closes)
    if len(lp) < forecast.minimum(h):
        return scoreerror(['{} has only {:,} daily prices. A {} forecast test needs at least {:,}. '
                           'Try a shorter horizon, or paste more prices under Advanced.'.format(
                               label, len(lp), forecast.horizons[h], forecast.minimum(h))], openadv=True)

    if not forecast.known(lp, h):
        fine, wait = forecasts.allowed(request.remote_addr)
        if not fine:
            return scoreerror(['That is a lot of forecasts. Each one takes a while to work out. '
                               'Try again in {} seconds.'.format(wait)], 429)
    try:
        res = forecast.board(lp, h)
    except forecast.Busy as e:
        return scoreerror([str(e)], 503)

    chartdata = dict(
        horizon=res['horizon'],
        rows=[dict(label=r['label'], skill=r['skill'], needs=r['needs'], edge=r['edge'])
              for r in res['rows'] if r['name'] != 'naive' and r['skill'] is not None],
        calibration=res['calibration'])
    args = dict(h=h)
    if label != 'your prices':
        args['stksymbol'] = label
    return render_template('forecast2.html', res=res, label=label, args=args, chartdata=chartdata,
                           win=any(r['edge'] == 'better' for r in res['rows']))


examples = [
    'I bought 100 shares of AAPL at $180 and paid $5 commission each way. With 15% tax, should I sell now or hold for 3 months?',
    "What's my break-even price on 50 shares of MSFT bought at $410 with $9.99 commission each way?",
    'How risky is holding 20 shares of TSLA that I bought at $250 for the next 6 months?',
]


@app.route('/agent')
def assistant():
    ok, why = agent.configured()
    return render_template('agent.html', ok=ok, why=why, examples=examples)


@app.route('/api/agent', methods=['POST'])
def agentapi():
    ok, why = agent.configured()
    if not ok:
        return jsonify(error=why), 503
    if not request.is_json:         # json only, so another site can't make your browser send this
        return jsonify(error='Send JSON.'), 415
    fine, wait = agent.questions.allowed(request.remote_addr)
    if not fine:
        r = jsonify(error='Slow down a little. Try again in {} seconds.'.format(wait))
        r.status_code = 429
        r.headers['Retry-After'] = str(wait)
        return r
    try:
        msgs = agent.clean(request.get_json(silent=True))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    try:
        return jsonify(agent.run(msgs))
    except agent.AgentError as e:
        return jsonify(error=str(e)), e.status


if __name__ == '__main__':
    app.run(port=8080)
