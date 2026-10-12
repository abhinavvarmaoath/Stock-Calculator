import json
import logging
import math
import re
import zlib

import numpy as np

import forecast
import prices
import sim
from calc import calc

log = logging.getLogger(__name__)


class ToolError(Exception):
    # something the model can fix or explain to the user: bad input, failed lookup
    pass


# lowest, can it be the lowest, highest. the same limits as the web forms
limits = dict(
    shares=(0, False, 1e9),
    buy_price=(0, False, 1e7),
    sell_price=(0, True, 1e7),
    buy_commission=(0, True, 1e7),
    sell_commission=(0, True, 1e7),
    tax_rate_percent=(0, True, 100),
    days=(1, True, 252),
    expected_return_percent=(-100, True, 200),
    current_price=(0, False, 1e7),
    volatility_percent=(0, True, 500),
)


def num(args, key, need=True, default=None):
    # the model's numbers are untrusted, so check every one
    low, lowok, high = limits[key]
    v = args.get(key)
    if v is None or v == '':
        if default is not None:
            return default
        if need:
            raise ToolError('{} is required'.format(key))
        return None
    if isinstance(v, bool):         # json true would sneak through as 1
        raise ToolError('{} must be a number'.format(key))
    try:
        v = float(v.replace(',', '')) if isinstance(v, str) else float(v)
    except (TypeError, ValueError):
        raise ToolError('{} must be a number'.format(key))
    if not math.isfinite(v):
        raise ToolError('{} must be a number'.format(key))
    if v < low or (v == low and not lowok):
        raise ToolError('{} must be {} {:g}'.format(key, 'at least' if lowok else 'more than', low))
    if v > high:
        raise ToolError('{} must be {:,.10g} or less'.format(key, high))
    return v


def symbol(args, need=True):
    s = args.get('symbol')
    if s is None or s == '':
        if need:
            raise ToolError('symbol is required')
        return None
    s = str(s).strip().upper()
    if not re.fullmatch(r'[A-Z0-9.\-]{1,10}', s):
        raise ToolError('symbol must be 1 to 10 letters or numbers, like AAPL or BRK-B')
    return s


def c2(x):
    return round(float(x), 2)


askuser = ' Ask the user for the current price and yearly volatility instead.'


def get_stock_info(args):
    sym = symbol(args)
    try:
        closes = prices.history(sym)
    except prices.PriceError as e:
        raise ToolError(e.what + askuser)
    last, vol, r = prices.stats(closes)
    out = dict(symbol=sym, latest_close=round(last, 4), yearly_volatility_percent=c2(vol * 100),
               days_of_history=len(closes),
               source='Yahoo Finance daily closes, adjusted for splits and dividends. Can be a day old.')
    if len(closes) > 252:
        out['change_over_last_year_percent'] = c2((last / closes[-253] - 1) * 100)
    return out


def calculate_trade(args):
    a = dict(allotment=num(args, 'shares'),
             iniprice=num(args, 'buy_price'),
             fnlprice=num(args, 'sell_price'),
             buycmm=num(args, 'buy_commission', default=0),
             sellcmm=num(args, 'sell_commission', default=0),
             cptlgain=num(args, 'tax_rate_percent', default=0))
    r = calc(**a)
    return dict(
        # what was used, so the answer can say what it assumed
        shares=a['allotment'], buy_price=a['iniprice'], sell_price=a['fnlprice'],
        buy_commission=a['buycmm'], sell_commission=a['sellcmm'], tax_rate_percent=a['cptlgain'],
        proceeds=c2(r['proceeds']), purchase_cost=c2(r['ttlshareprice']),
        total_commissions=c2(r['cmm']), gain_before_tax=c2(r['gain']), tax=c2(r['tocg']),
        net_profit=c2(r['netprft']), roi_percent=c2(r['roi']),
        break_even_price=round(r['brkeven'], 4),
        notes='Tax is only charged on a gain, no refund is assumed on a loss. '
              'ROI is net profit divided by total cost (purchase + commissions + tax).')


def simulate_holding(args):
    a = dict(allotment=num(args, 'shares'),
             iniprice=num(args, 'buy_price'),
             buycmm=num(args, 'buy_commission', default=0),
             sellcmm=num(args, 'sell_commission', default=0),
             cptlgain=num(args, 'tax_rate_percent', default=0))
    ndays = num(args, 'days')
    if ndays != int(ndays):
        raise ToolError('days must be a whole number')
    drift = num(args, 'expected_return_percent', default=0)
    curprice = num(args, 'current_price', need=False)
    volpct = num(args, 'volatility_percent', need=False)
    method = args.get('model') or 'normal'
    if method not in ('normal', 'resample'):
        raise ToolError("model must be 'normal' or 'resample'")

    # the symbol is only needed when something has to be looked up
    sym = symbol(args, need=curprice is None or volpct is None or method == 'resample')
    try:
        price, vol, hist, src = prices.resolve(sym, curprice, None if volpct is None else volpct / 100,
                                               method == 'resample')
    except prices.PriceError as e:
        raise ToolError(e.what + askuser)

    # same question, same numbers: the seed comes from the inputs
    seed = zlib.crc32(json.dumps(args, sort_keys=True, default=str).encode())
    r = sim.run(price, vol, drift / 100, int(ndays), a['allotment'], a['iniprice'],
                a['buycmm'], a['sellcmm'], a['cptlgain'], hist=hist, seed=seed)

    out = dict(
        days=int(ndays), runs=r['npaths'], model=method,
        shares=a['allotment'], buy_price=a['iniprice'], buy_commission=a['buycmm'],
        sell_commission=a['sellcmm'], tax_rate_percent=a['cptlgain'],
        price_today=round(price, 4), price_today_source=src['price'].replace('entered by you', 'given by the user'),
        volatility_percent=c2(vol * 100), volatility_source=src['vol'].replace('entered by you', 'given by the user'),
        expected_return_percent=drift,
        break_even_price=round(r['brkeven'], 4),
        if_sold_today=dict(net_profit=c2(r['now']['netprft']), roi_percent=c2(r['now']['roi'])),
        chance_of_profit_percent=round(r['pprofit'] * 100, 1),
        expected_net_profit=c2(r['expprft']), median_net_profit=c2(r['medprft']),
        net_profit_5th_percentile=c2(r['lo']), net_profit_95th_percentile=c2(r['hi']),
        average_of_worst_5_percent_of_runs=c2(r['worst']),
        price_at_end={k: c2(r['fan'][k][-1]) for k in ('p5', 'p25', 'p50', 'p75', 'p95')},
        notes='A simulation of what could happen if prices behave like the model. It is not a prediction. '
              'Tax is only charged on a gain, no refund is assumed on a loss. '
              'The numbers carry about a percentage point of random noise.')
    if sym:
        out['symbol'] = sym
    return out


def r1(x):
    return None if x is None else round(float(x), 1)


def forecast_scoreboard(args):
    sym = symbol(args)
    raw = args.get('horizon_days')
    raw = 21 if raw is None or raw == '' else raw
    try:
        h = None if isinstance(raw, bool) or float(raw) != int(float(raw)) else int(float(raw))
    except (TypeError, ValueError, OverflowError):
        h = None
    if h not in forecast.horizons:
        opts = ['{} ({})'.format(d, name) for d, name in forecast.horizons.items()]
        raise ToolError('horizon_days must be ' + ', '.join(opts[:-1]) + ' or ' + opts[-1])

    try:
        closes = prices.history(sym, forecast.years)
    except prices.PriceError as e:
        raise ToolError(e.what + " The scoreboard needs the stock's own price history, so tell the user it is not available right now.")
    lp = np.log(closes)
    if len(lp) < forecast.minimum(h):
        raise ToolError('{} has only {:,} daily prices and a {} test needs at least {:,}. Try a shorter horizon_days.'.format(
            sym, len(lp), forecast.horizons[h], forecast.minimum(h)))
    try:
        res = forecast.board(lp, h)
    except forecast.Busy as e:
        raise ToolError(str(e))

    result = dict(better='beat the benchmark', worse='worse than the benchmark',
                  tied="can't be told apart from luck", untested='too few forecasts to test')
    methods = [dict(method=r['label'], average_miss_percent=r1(r['miss']),
                    better_than_benchmark_by_percent=r1(r['skill']),
                    nearer_than_benchmark_percent=r1(r['closer']),
                    right_direction_percent=r1(r['direction']),
                    luck_test_p_value=None if r['padj'] is None else round(r['padj'], 3),
                    result=result[r['edge']])
               for r in res['rows'] if r['name'] != 'naive']
    rangecheck = {'ok': 'no sign it is off', 'narrow': 'too narrow', 'wide': 'too wide'}
    return dict(
        symbol=sym, horizon=res['horizon'], daily_prices_used=res['prices'],
        forecasts_tested=res['forecasts'], days_between_forecasts=res['every'],
        benchmark='the price stays where it is today',
        benchmark_average_miss_percent=r1(res['rows'][0]['miss']),
        summary=res['headline'], sample_size_note=res['detail'], methods=methods,
        simulator_range_check=[dict(range_percent=c['level'], held_percent=round(c['covered']),
                                    result=rangecheck[c['verdict']], separate_forecasts=c['indep'])
                               for c in res['calibration']],
        notes="Looks backwards only: how each method would have done on this stock's past prices, using only what was "
              'known on the day. It is not a prediction and has no forecast price in it. The luck test already allows '
              'for trying several methods. Better than the benchmark means a smaller average miss, and a minus means a '
              'bigger one. Forecasts closer together than the horizon overlap, so there are fewer separate tests than forecasts.')


functions = dict(get_stock_info=get_stock_info, calculate_trade=calculate_trade,
                 simulate_holding=simulate_holding, forecast_scoreboard=forecast_scoreboard)


def run(name, args):
    # gives back (result, is_error). an error result is plain text for the model to read
    fn = functions.get(name)
    if fn is None:
        return 'Unknown tool {!r}. The tools are: {}'.format(name, ', '.join(functions)), True
    if not isinstance(args, dict):
        return 'The tool input must be a JSON object', True
    try:
        return fn(args), False
    except ToolError as e:
        return str(e), True
    except Exception:
        log.exception('tool %s crashed', name)
        return 'The tool hit an unexpected error. Tell the user it failed and do not guess the numbers.', True


# what the model is told about each tool
shares = {'type': 'number', 'description': 'Number of shares (can be fractional)'}
buyprice = {'type': 'number', 'description': 'Price paid per share, in dollars'}
buycmm = {'type': 'number', 'description': 'Commission paid when buying, in dollars in total (not per share). Default 0'}
sellcmm = {'type': 'number', 'description': 'Commission paid when selling, in dollars in total (not per share). Default 0'}
taxrate = {'type': 'number', 'description': 'Capital gains tax rate in percent, for example 15 for 15%. Only charged on a gain. Default 0'}

tooldefs = [
    {
        'name': 'get_stock_info',
        'description': (
            "Look up a stock's latest closing price and its yearly volatility (how much it usually moves) "
            'from the last 2 years of daily prices on Yahoo Finance. Call this when the user names a ticker '
            'and you need its current price or how volatile it is. Prices can be a day old. If it fails, '
            'ask the user for the price and volatility instead of guessing.'),
        'input_schema': {
            'type': 'object',
            'properties': {'symbol': {'type': 'string', 'description': 'Ticker symbol, like AAPL or BRK-B'}},
            'required': ['symbol'],
            'additionalProperties': False,
        },
    },
    {
        'name': 'calculate_trade',
        'description': (
            'Work out the exact profit or loss on a trade after commissions and capital gains tax: proceeds, '
            'tax, net profit, return on investment and the break-even sell price. Call this for any question '
            'about what the user makes if they sell at a given price, including selling today at the latest '
            'price. Never work these numbers out yourself.'),
        'input_schema': {
            'type': 'object',
            'properties': {
                'shares': shares,
                'buy_price': buyprice,
                'sell_price': {'type': 'number', 'description': 'Price per share the stock is sold at, in dollars'},
                'buy_commission': buycmm,
                'sell_commission': sellcmm,
                'tax_rate_percent': taxrate,
            },
            'required': ['shares', 'buy_price', 'sell_price'],
            'additionalProperties': False,
        },
    },
    {
        'name': 'simulate_holding',
        'description': (
            'Run a Monte Carlo simulation of holding a stock for a while and then selling: 10,000 random price '
            'paths, each put through the same profit-after-commissions-and-tax maths as calculate_trade. Call '
            'this when the user asks what could happen if they hold, how risky a position is, or whether to '
            'sell now or wait. It returns the chance of profit, expected and median net profit, a likely range '
            'and the bad case. It shows what could happen if prices behave like the model, not a prediction. '
            'Pass current_price and volatility_percent if the user told you them, otherwise pass the symbol '
            'and they are looked up.'),
        'input_schema': {
            'type': 'object',
            'properties': {
                'symbol': {'type': 'string', 'description': 'Ticker symbol. Needed unless both current_price and volatility_percent are given'},
                'shares': shares,
                'buy_price': buyprice,
                'buy_commission': buycmm,
                'sell_commission': sellcmm,
                'tax_rate_percent': taxrate,
                'days': {'type': 'integer', 'description': 'Trading days until the sale, 1 to 252. About 21 a month: 63 is 3 months, 126 is 6 months, 252 is a year'},
                'expected_return_percent': {'type': 'number', 'description': 'Expected yearly return in percent. Default 0, meaning no view on which way the price goes'},
                'current_price': {'type': 'number', 'description': 'Price today per share. Leave out to use the latest close'},
                'volatility_percent': {'type': 'number', 'description': 'Yearly volatility in percent, for example 30 for 30%. Leave out to work it out from the last 2 years of prices'},
                'model': {'type': 'string', 'enum': ['normal', 'resample'], 'description': "'normal' uses random moves from a bell curve (default). 'resample' reuses the stock's own past daily moves, which keeps its big up and down days, and needs the symbol"},
            },
            'required': ['shares', 'buy_price', 'days'],
            'additionalProperties': False,
        },
    },    {
        'name': 'forecast_scoreboard',
        'description': (
            "Test how well forecasting methods would have predicted a stock's price, using its own last 10 years "
            'of daily prices. Six methods (including ARIMA and gradient boosting) each forecast the price 1 week, '
            '1 month or 3 months ahead from dozens of past days, using only what was known on that day, and are '
            'scored against the benchmark "the price stays where it is", with a statistical test of whether any '
            "gap is more than luck. It also checks whether the simulator's price range for that stock has held up. "
            "Call this when the user asks whether a stock's price can be forecast or predicted, whether a "
            "forecasting method works, or how far to trust the simulator's range. It gives no price forecast and "
            'takes 5 to 30 seconds. If it fails, say so and do not make up results.'),
        'input_schema': {
            'type': 'object',
            'properties': {
                'symbol': {'type': 'string', 'description': 'Ticker symbol, like AAPL or BRK-B'},
                'horizon_days': {'type': 'integer', 'enum': [5, 21, 63], 'description': 'How far ahead the forecasts look, in trading days: 5 is 1 week, 21 is 1 month, 63 is 3 months. Default 21'},
            },
            'required': ['symbol'],
            'additionalProperties': False,
        },
    },
]
