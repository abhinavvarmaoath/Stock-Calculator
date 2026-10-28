import json
import math
import re

import numpy as np
import pytest

import prices
from app import app

good = dict(stksymbol='abc', allotment='100', iniprice='10', buycmm='5', sellcmm='5',
            cptlgain='15', ndays='63', drift='0', curprice='12', vol='35', method='normal')


@pytest.fixture
def client():
    app.testing = True
    return app.test_client()


@pytest.fixture
def nonet(monkeypatch):
    # fails the test if the page tries to look prices up
    def boom(sym):
        raise AssertionError('should not look up prices')
    monkeypatch.setattr(prices, 'history', boom)


def closes(n=300, vol=0.30, last=50.0, seed=1):
    # made up daily prices that end at `last`
    r = np.random.default_rng(seed).normal(0, vol / np.sqrt(252), n)
    c = np.exp(np.cumsum(r))
    return c / c[-1] * last


def chartdata(html):
    return json.loads(re.search(r'<script type="application/json" id="simdata">(.*?)</script>', html, re.S).group(1))


def numbers(x):
    if isinstance(x, dict):
        return [n for v in x.values() for n in numbers(v)]
    if isinstance(x, list):
        return [n for v in x for n in numbers(v)]
    return [x]


def test_form_page(client):
    html = client.get('/sim').get_data(as_text=True)
    assert 'name="ndays"' in html
    assert '<option value="63" selected>' in html
    assert 'href="/sim"' in html        # nav link, also on the other pages
    assert 'href="/sim"' in client.get('/').get_data(as_text=True)


def test_form_is_filled_in_from_the_link(client):
    html = client.get('/sim?stksymbol=ABC&allotment=100&ndays=126').get_data(as_text=True)
    assert 'value="ABC"' in html
    assert 'name="allotment" id="allotment" value="100"' in html
    assert '<option value="126" selected>' in html


def test_report_links_to_the_simulator_with_the_trade_filled_in(client):
    trade = dict(stksymbol='abc', allotment='1,000', iniprice='10.5', fnlprice='15', buycmm='5', sellcmm='5', cptlgain='15')
    html = client.post('/post', data=trade).get_data(as_text=True)
    link = re.search(r'href="(/sim\?[^"]+)"', html).group(1).replace('&amp;', '&')
    page = client.get(link).get_data(as_text=True)
    assert 'value="ABC"' in page
    assert 'name="allotment" id="allotment" value="1000"' in page
    assert 'name="iniprice" id="iniprice" value="10.5"' in page


def test_sim_with_prices_entered_needs_no_lookup(client, nonet):
    r = client.post('/sim', data=good)
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert 'Chance of profit' in html
    assert 'Price today: $12.00 (entered by you)' in html
    assert 'Volatility: 35.00% a year (entered by you)' in html
    assert 'Normal random daily moves' in html


def test_sim_with_no_spread_is_the_calculator_answer(client, nonet):
    # price stays at 15 and never moves, so every run is what the calculator says
    html = client.post('/sim', data={**good, 'curprice': '15', 'vol': '0', 'iniprice': '10'}).get_data(as_text=True)
    assert '100%' in html
    assert '$416.50' in html
    assert '$416.50 to $416.50' in html


def test_chart_data(client, nonet):
    d = chartdata(client.post('/sim', data=good).get_data(as_text=True))
    assert d['npaths'] == 10000
    assert len(d['fan']['days']) == 64
    assert d['brkeven'] == pytest.approx(10.10)
    assert len(d['bins']['edges']) == len(d['bins']['counts']) + 1
    assert sum(d['bins']['counts']) > 9700
    assert all(math.isfinite(n) for n in numbers(d))


def test_table_views(client, nonet):
    html = client.post('/sim', data=good).get_data(as_text=True)
    assert html.count('Table view') == 2
    assert '<td>63</td>' in html          # last day is always in the fan table


def test_sim_looks_up_price_and_volatility(client, monkeypatch):
    monkeypatch.setattr(prices, 'history', lambda sym: closes(last=50.0, vol=0.30))
    r = client.post('/sim', data={**good, 'curprice': '', 'vol': ''})
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert 'Price today: $50.00 (latest close)' in html
    assert 'worked out from the last 2 years' in html
    vol = float(re.search(r'Volatility: (\d+\.\d\d)% a year', html).group(1))
    assert 25 < vol < 35


def test_only_the_missing_one_is_looked_up(client, monkeypatch):
    monkeypatch.setattr(prices, 'history', lambda sym: closes(last=50.0))
    html = client.post('/sim', data={**good, 'curprice': '', 'vol': '35'}).get_data(as_text=True)
    assert 'Price today: $50.00 (latest close)' in html
    assert 'Volatility: 35.00% a year (entered by you)' in html


def test_resampled_model(client, monkeypatch):
    monkeypatch.setattr(prices, 'history', lambda sym: closes())
    r = client.post('/sim', data={**good, 'method': 'resample'})
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert 'Daily moves resampled from the last 2 years' in html
    assert 'Price today: $12.00 (entered by you)' in html     # typed in values are kept


def test_lookup_failure_is_shown_nicely(client, monkeypatch):
    def fail(sym):
        raise prices.PriceError("Couldn't get prices for ABC right now. Try again, or enter the current price and volatility under Advanced.")
    monkeypatch.setattr(prices, 'history', fail)
    r = client.post('/sim', data={**good, 'curprice': '', 'vol': ''})
    html = r.get_data(as_text=True)
    assert r.status_code == 400
    assert 'Couldn&#39;t get prices for ABC' in html
    assert 'name="allotment" id="allotment" value="100"' in html      # form is kept
    assert '<details open' in html                                    # the message points at Advanced


def test_change_inputs_link_comes_back_filled_in(client, nonet):
    html = client.post('/sim', data={**good, 'method': 'normal'}).get_data(as_text=True)
    link = re.search(r'href="(/sim\?[^"]+)"', html.split('Change inputs')[0].rsplit('<a', 1)[1]).group(1).replace('&amp;', '&')
    page = client.get(link).get_data(as_text=True)
    assert 'value="ABC"' in page
    assert 'name="vol" id="vol" value="35"' in page
    assert 'name="curprice" id="curprice" value="12"' in page


@pytest.mark.parametrize('field, value, msg', [
    ('ndays', '7', 'Pick how long to hold for'),
    ('ndays', '²', 'Pick how long to hold for'),
    ('ndays', '', 'Pick how long to hold for'),
    ('method', 'magic', 'Pick a model from the list'),
    ('drift', '-101', 'Expected yearly return must be at least -100'),
    ('drift', '201', 'Expected yearly return must be 200 or less'),
    ('vol', '-1', 'Yearly volatility must be at least 0'),
    ('vol', '501', 'Yearly volatility must be 500 or less'),
    ('curprice', '0', 'Current price must be more than 0'),
    ('allotment', '0', 'Number of shares must be more than 0'),
    ('cptlgain', '101', 'Capital gain tax rate must be 100 or less'),
    ('stksymbol', 'no good!', 'Stock symbol must be'),
])
def test_bad_sim_input(client, nonet, field, value, msg):
    r = client.post('/sim', data={**good, field: value})
    assert r.status_code == 400
    assert msg in r.get_data(as_text=True)


def test_sim_input_is_escaped(client, nonet):
    r = client.post('/sim', data={**good, 'stksymbol': '"><script>alert(1)</script>'})
    assert r.status_code == 400
    assert b'<script>alert(1)</script>' not in r.data


@pytest.mark.parametrize('ndays, h', [('21', 21), ('63', 63), ('126', 63), ('252', 63)])
def test_the_result_links_to_the_forecast_check_for_the_nearest_horizon(client, nonet, ndays, h):
    html = client.post('/sim', data=dict(good, ndays=ndays)).get_data(as_text=True)
    link = re.search(r'href="(/forecast\?[^"]+)"[^>]*>Has this range held up\?', html).group(1).replace('&amp;', '&')
    assert 'stksymbol=ABC' in link and 'h={}'.format(h) in link
    page = client.get(link).get_data(as_text=True)                      # and the link fills the form in
    assert 'value="ABC"' in page and re.search(r'<option value="{}"\s+selected'.format(h), page)
