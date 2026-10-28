import html
import os
import re

import numpy as np
import pytest

import forecast
import prices
from app import app


REAL = list(forecast.names)         # the whole list, before the tests below make it shorter


@pytest.fixture
def client():
    app.testing = True
    return app.test_client()


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    # fast, repeatable: three methods instead of seven, and nothing remembered between tests
    monkeypatch.setattr(forecast, 'names', ['naive', 'drift', 'ridge'])
    forecast.cache.clear()
    prices.cache.clear()
    from app import forecasts
    forecasts.hits.clear()
    yield
    forecast.cache.clear()


def closes(n=1300, seed=1, up=0.0003, vol=0.015):
    return np.exp(np.log(100) + np.cumsum(np.random.default_rng(seed).normal(up, vol, n)))


def lookup(monkeypatch, c=None, boom=None):
    calls = []

    def history(sym, years=2):
        calls.append((sym, years))
        if boom:
            raise boom
        return closes() if c is None else c
    monkeypatch.setattr(prices, 'history', history)
    return calls


def post(client, **kw):
    data = dict(stksymbol='ABC', h='21')
    data.update(kw)
    return client.post('/forecast', data=data)


def text(r):
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', r.get_data(as_text=True))))


# the form

def test_the_form_page(client):
    r = client.get('/forecast')
    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert 'Forecast scoreboard' in body and 'name="closes"' in body and 'name="h"' in body
    for days, name in forecast.horizons.items():
        assert 'value="{}"'.format(days) in body and name in body
    assert 'href="/forecast"' in body                                   # in the menu


def test_the_text_counts_and_lists_the_methods_that_there_are(client, monkeypatch):
    # adding a method means saying so in the readme and on the page, and this fails until that is done
    monkeypatch.setattr(forecast, 'names', REAL)
    word = forecast.count()
    form = text(client.get('/forecast'))
    assert 'This tries {} ways of forecasting'.format(word) in form
    assert 'Testing {} methods'.format(word) in form

    inreadme = dict(naive='benchmark', drift='drift', holt='Holt', arima='ARIMA', prophet='Prophet',
                    ridge='ridge regression', gbm='gradient boosting')
    onpage = dict(naive='Price stays put', drift='Drift', holt='Damped trend (Holt)', arima='ARIMA(1,1,1)',
                  prophet='Prophet (trend only)', ridge='Momentum regression', gbm='Gradient boosting')
    assert set(inreadme) == set(onpage) == set(REAL)
    readme = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'README.md')).read()
    assert 'the app tries {} forecasting methods'.format(word) in readme
    for name in REAL:
        assert inreadme[name] in readme


def test_the_results_page_counts_and_describes_the_methods_it_ran(client, monkeypatch):
    lookup(monkeypatch)
    t = text(post(client))
    assert 'Three ways to forecast the price 1 month ahead' in t and 'What the three methods are' in t
    for heading in ('Price stays put', 'Drift', 'Damped trend (Holt)', 'ARIMA(1,1,1)', 'Prophet (trend only)',
                    'Momentum regression', 'Gradient boosting'):
        assert heading in t.split('methods are')[1]


def test_the_form_can_be_filled_in_from_the_link(client):
    body = client.get('/forecast?stksymbol=MSFT&h=63').get_data(as_text=True)
    assert 'value="MSFT"' in body and re.search(r'<option value="63"\s+selected', body)


# a good run

def test_a_run_shows_the_scoreboard(client, monkeypatch):
    calls = lookup(monkeypatch)
    r = post(client)
    assert r.status_code == 200
    assert calls == [('ABC', forecast.years)]
    t = text(r)
    assert 'Forecast scoreboard ABC' in t and '1 month' in t
    for name in ('naive', 'drift', 'ridge'):
        assert forecast.labels[name] in t
    assert forecast.labels['gbm'] not in r.get_data(as_text=True).split('methods are')[0]
    assert 'Has the simulator' in t and 'held' in t
    assert 'id="boarddata"' in r.get_data(as_text=True)


def test_the_page_says_the_luck_test_cannot_allow_for_how_many_stocks_you_try(client, monkeypatch):
    lookup(monkeypatch)
    t = text(post(client))
    assert 'allows for the {} methods tried here'.format(len(forecast.names) - 1) in t       # the benchmark is not one of them
    assert 'look at 20 and about one will show a winner by luck alone' in t


def test_a_flat_price_gets_a_plain_answer(client, monkeypatch):
    lookup(monkeypatch)
    r = post(client, stksymbol='', closes='\n'.join(['100'] * 800))
    assert r.status_code == 200
    assert 'nothing to forecast' in text(r) and 'luck zone' not in r.get_data(as_text=True)


def test_a_method_that_cannot_run_is_shown_as_failed_and_the_rest_carry_on(client, monkeypatch):
    # prophet is the most fragile one (it needs a compiled stan runtime). if it cannot load, the page says so
    monkeypatch.setattr(forecast, 'names', ['naive', 'drift', 'prophet'])
    real = forecast.make

    def broken(lp, h):
        raise ImportError('no stan here')
    monkeypatch.setattr(forecast, 'make', lambda name: broken if name == 'prophet' else real(name))
    lookup(monkeypatch)
    r = post(client)
    t = text(r)
    assert r.status_code == 200
    n = int(re.search(r'tried on (\d+) past days', t).group(1))
    assert '{} of {} forecasts failed'.format(n, n) in t
    assert 'Too few forecasts' in t and forecast.labels['drift'] in t


def test_the_chart_data_is_in_the_page(client, monkeypatch):
    import json
    lookup(monkeypatch)
    body = post(client).get_data(as_text=True)
    d = json.loads(re.search(r'id="boarddata">(.*?)</script>', body, re.S).group(1))
    assert [r['label'] for r in d['rows']] == [forecast.labels['drift'], forecast.labels['ridge']]
    assert [c['level'] for c in d['calibration']] == [80, 90]
    assert d['horizon'] == '1 month'


def test_the_page_shows_the_scoreboard_of_the_log_prices(client, monkeypatch):
    # the numbers on the page are the ones forecast.scoreboard gives for these prices
    import json
    c = closes(1300, seed=9)
    lookup(monkeypatch, c=c)
    body = post(client).get_data(as_text=True)
    shown = json.loads(re.search(r'id="boarddata">(.*?)</script>', body, re.S).group(1))
    want = forecast.scoreboard(np.log(c), 21)
    assert [(r['skill'], r['needs'], r['edge']) for r in shown['rows']] == \
        [(r['skill'], r['needs'], r['edge']) for r in want['rows'] if r['name'] != 'naive']
    assert shown['calibration'] == want['calibration']
    assert want['headline'] in html.unescape(body)


def test_pasted_prices_are_used_instead_of_a_lookup(client, monkeypatch):
    calls = lookup(monkeypatch, boom=AssertionError('should not look anything up'))
    pasted = '\n'.join('{:.2f}'.format(p) for p in closes())
    r = post(client, stksymbol='', closes=pasted)
    assert r.status_code == 200 and calls == []
    assert 'Forecast scoreboard your prices' in text(r)
    assert 'Open the simulator' not in r.get_data(as_text=True)         # there is no ticker to open it with


def test_a_symbol_given_with_pasted_prices_names_the_page(client, monkeypatch):
    lookup(monkeypatch, boom=AssertionError('no lookup'))
    r = post(client, stksymbol='xyz', closes='\n'.join('{:.2f}'.format(p) for p in closes()))
    assert 'Forecast scoreboard XYZ' in text(r) and 'href="/sim?stksymbol=XYZ"' in r.get_data(as_text=True)


def test_the_same_run_twice_is_worked_out_once(client, monkeypatch):
    lookup(monkeypatch)
    made = []
    real = forecast.scoreboard
    monkeypatch.setattr(forecast, 'scoreboard', lambda *a, **k: made.append(1) or real(*a, **k))
    assert post(client).status_code == 200 and post(client).status_code == 200
    assert len(made) == 1


# bad input

@pytest.mark.parametrize('kw, msg', [
    (dict(stksymbol=''), 'Stock symbol must be'),
    (dict(stksymbol='NOT A SYMBOL'), 'Stock symbol must be'),
    (dict(h='7'), 'Pick how far ahead'),
    (dict(h=''), 'Pick how far ahead'),
    (dict(h='abc'), 'Pick how far ahead'),
])
def test_bad_choices_are_sent_back_with_a_message(client, monkeypatch, kw, msg):
    calls = lookup(monkeypatch)
    r = post(client, **kw)
    assert r.status_code == 400 and msg in text(r) and 'name="closes"' in r.get_data(as_text=True)
    assert calls == []


@pytest.mark.parametrize('pasted, msg', [
    ('182.63\n185.64x\n184.25', "Couldn't read '185.64x'"),
    ('182.63\n-5\n184.25', "Couldn't read '-5'"),
    ('182.63\nabc', "Couldn't read 'abc'"),
    ('182.63, 185.64', "Couldn't read '182.63,'"),            # a comma between two prices is not a thousands separator
    ('182.63\n0\n184.25', 'above 0'),
    ('\n'.join(['100'] * 4001), 'at most 4,000'),
])
def test_unreadable_pasted_prices_are_refused(client, monkeypatch, pasted, msg):
    lookup(monkeypatch)
    r = post(client, closes=pasted)
    assert r.status_code == 400 and msg in text(r)
    assert pasted.split('\n')[0] in r.get_data(as_text=True)         # and what was pasted is kept, so it can be fixed


def test_dollar_signs_and_thousands_separators_are_fine(client, monkeypatch):
    lookup(monkeypatch)
    pasted = '\n'.join('${:,.2f}'.format(p * 40) for p in closes())      # prices like $4,000.00
    assert post(client, closes=pasted).status_code == 200


def test_thousands_separators_are_not_read_as_two_prices(client, monkeypatch):
    lookup(monkeypatch)
    r = post(client, closes='\n'.join(['1,234.50'] * 800))
    assert r.status_code == 200 and '800 daily closes' in text(r)       # 800 prices, not 1,600


def test_too_little_history_says_how_much_is_needed(client, monkeypatch):
    lookup(monkeypatch, c=closes(900))
    r = post(client, h='63')
    t = text(r)
    assert r.status_code == 400 and 'ABC has only 900 daily prices' in t
    assert '3 months' in t and '{:,}'.format(forecast.minimum(63)) in t


def test_lookup_trouble_is_explained_and_points_to_pasting(client, monkeypatch):
    lookup(monkeypatch, boom=prices.PriceError("Couldn't get prices for ABC right now."))
    r = post(client)
    t = text(r)
    assert r.status_code == 400 and "Couldn't get prices for ABC right now." in t
    assert 'paste the closing prices' in t and 'current price and volatility' not in t
    assert re.search(r'<details\s+open', r.get_data(as_text=True))


# limits

def test_a_busy_server_says_so(client, monkeypatch):
    lookup(monkeypatch)
    taken = [forecast.slots.acquire(), forecast.slots.acquire()]
    try:
        r = post(client)
    finally:
        for _ in taken:
            forecast.slots.release()
    assert r.status_code == 503 and 'Other forecasts are running' in text(r)


def test_new_forecasts_are_limited_but_remembered_ones_are_not(client, monkeypatch):
    lookup(monkeypatch)
    from app import forecasts
    monkeypatch.setattr(forecasts, 'limit', 2)
    assert post(client, stksymbol='A1').status_code == 200
    assert post(client, stksymbol='A1').status_code == 200              # remembered, so it did not count
    assert post(client, stksymbol='A1', h='5').status_code == 200
    r = post(client, stksymbol='A1', h='63')
    assert r.status_code == 429 and 'Try again in' in text(r)
    assert post(client, stksymbol='A1').status_code == 200              # still free to look at the old ones


def test_asking_with_bad_input_does_not_use_up_the_limit(client, monkeypatch):
    lookup(monkeypatch)
    from app import forecasts
    for _ in range(forecasts.limit + 3):
        assert post(client, h='7').status_code == 400
    assert post(client).status_code == 200


def test_only_the_form_is_allowed_to_post_and_get(client):
    assert client.put('/forecast').status_code == 405
