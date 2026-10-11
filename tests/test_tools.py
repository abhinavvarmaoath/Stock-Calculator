import numpy as np
import pytest

import app
import prices
import tools

trade = dict(shares=100, buy_price=10, sell_price=15, buy_commission=5, sell_commission=5, tax_rate_percent=15)
hold = dict(shares=100, buy_price=10, buy_commission=5, sell_commission=5, tax_rate_percent=15,
            days=63, current_price=12, volatility_percent=35)


@pytest.fixture
def nolookup(monkeypatch):
    def boom(sym):
        raise AssertionError('should not look anything up')
    monkeypatch.setattr(prices, 'history', boom)


def fakecloses(monkeypatch, n=300, vol=0.30, last=50.0, seed=1):
    r = np.random.default_rng(seed).normal(0, vol / np.sqrt(252), n)
    c = np.exp(np.cumsum(r))
    c = c / c[-1] * last
    monkeypatch.setattr(prices, 'history', lambda sym: c)
    return c


# calculate_trade

def test_calculate_trade_is_the_calculator():
    r = tools.calculate_trade(trade)
    assert r['proceeds'] == 1500 and r['purchase_cost'] == 1000
    assert r['total_commissions'] == 10 and r['gain_before_tax'] == 490
    assert r['tax'] == 73.5 and r['net_profit'] == 416.5
    assert r['roi_percent'] == 38.44 and r['break_even_price'] == 10.1


def test_calculate_trade_says_what_it_used():
    r = tools.calculate_trade(dict(shares=100, buy_price=10, sell_price=15))
    assert r['buy_commission'] == 0 and r['sell_commission'] == 0 and r['tax_rate_percent'] == 0
    assert r['net_profit'] == 500


def test_calculate_trade_loss_has_no_tax():
    r = tools.calculate_trade({**trade, 'buy_price': 15, 'sell_price': 10})
    assert r['tax'] == 0 and r['net_profit'] == -510


def test_numbers_can_come_as_text():
    r = tools.calculate_trade({**trade, 'shares': '1,000', 'buy_price': '10.5'})
    assert r['shares'] == 1000 and r['buy_price'] == 10.5


@pytest.mark.parametrize('change, msg', [
    ({'shares': 0}, 'shares must be more than 0'),
    ({'shares': -5}, 'shares must be more than 0'),
    ({'shares': 'lots'}, 'shares must be a number'),
    ({'shares': True}, 'shares must be a number'),
    ({'shares': [1]}, 'shares must be a number'),
    ({'shares': float('nan')}, 'shares must be a number'),
    ({'sell_price': float('inf')}, 'sell_price must be a number'),
    ({'shares': 5e9}, 'shares must be 1,000,000,000 or less'),
    ({'buy_price': 0}, 'buy_price must be more than 0'),
    ({'buy_price': 1e12}, 'buy_price must be 10,000,000 or less'),
    ({'sell_price': -1}, 'sell_price must be at least 0'),
    ({'sell_commission': -1}, 'sell_commission must be at least 0'),
    ({'tax_rate_percent': 150}, 'tax_rate_percent must be 100 or less'),
])
def test_calculate_trade_rejects_bad_input(change, msg):
    with pytest.raises(tools.ToolError, match=msg):
        tools.calculate_trade({**trade, **change})


@pytest.mark.parametrize('missing', ['shares', 'buy_price', 'sell_price'])
def test_calculate_trade_needs_the_basics(missing):
    args = {k: v for k, v in trade.items() if k != missing}
    with pytest.raises(tools.ToolError, match=missing + ' is required'):
        tools.calculate_trade(args)


def test_zero_sell_price_is_fine():
    assert tools.calculate_trade({**trade, 'sell_price': 0})['net_profit'] == -1010


# simulate_holding

def test_simulate_with_everything_given_needs_no_lookup(nolookup):
    r = tools.simulate_holding(hold)
    assert r['runs'] == 10000 and r['days'] == 63 and r['model'] == 'normal'
    assert r['price_today'] == 12 and r['price_today_source'] == 'given by the user'
    assert r['volatility_percent'] == 35 and r['volatility_source'] == 'given by the user'
    assert r['break_even_price'] == 10.1
    assert r['if_sold_today'] == dict(net_profit=161.5, roi_percent=15.55)
    assert 79 < r['chance_of_profit_percent'] < 84          # about 81.5 in theory
    assert r['net_profit_5th_percentile'] < r['median_net_profit'] < r['net_profit_95th_percentile']
    assert r['average_of_worst_5_percent_of_runs'] <= r['net_profit_5th_percentile']
    assert set(r['price_at_end']) == {'p5', 'p25', 'p50', 'p75', 'p95'}
    assert 'symbol' not in r


def test_simulate_without_any_spread_is_the_calculator(nolookup):
    r = tools.simulate_holding({**hold, 'current_price': 15, 'volatility_percent': 0})
    assert r['chance_of_profit_percent'] == 100
    assert r['expected_net_profit'] == r['median_net_profit'] == 416.5
    assert r['net_profit_5th_percentile'] == r['net_profit_95th_percentile'] == 416.5


def test_same_question_gives_the_same_numbers(nolookup):
    assert tools.simulate_holding(hold) == tools.simulate_holding(hold)
    assert tools.simulate_holding(hold) != tools.simulate_holding({**hold, 'days': 126})


def test_simulate_looks_up_what_is_missing(monkeypatch):
    c = fakecloses(monkeypatch, last=50.0)
    args = {k: v for k, v in hold.items() if k not in ('current_price', 'volatility_percent')}
    r = tools.simulate_holding({**args, 'symbol': 'abc'})
    assert r['symbol'] == 'ABC'
    assert r['price_today'] == pytest.approx(50.0, abs=1e-4)
    assert r['price_today_source'] == 'latest close' and 'daily closes' in r['volatility_source']
    assert 25 < r['volatility_percent'] < 35


def test_simulate_only_looks_up_the_missing_one(monkeypatch):
    fakecloses(monkeypatch, last=50.0)
    r = tools.simulate_holding({**hold, 'symbol': 'ABC', 'current_price': None})
    assert r['price_today_source'] == 'latest close'
    assert r['volatility_percent'] == 35 and r['volatility_source'] == 'given by the user'


def test_simulate_needs_a_symbol_only_when_it_has_to_look_something_up(monkeypatch):
    fakecloses(monkeypatch)
    with pytest.raises(tools.ToolError, match='symbol is required'):
        tools.simulate_holding({**hold, 'current_price': None})
    with pytest.raises(tools.ToolError, match='symbol is required'):
        tools.simulate_holding({**hold, 'model': 'resample'})
    assert tools.simulate_holding({**hold, 'model': 'resample', 'symbol': 'ABC'})['model'] == 'resample'


def test_simulate_lookup_failure_tells_the_model_what_to_do(monkeypatch):
    def fail(sym):
        raise prices.PriceError("Couldn't get prices for ABC right now.")
    monkeypatch.setattr(prices, 'history', fail)
    with pytest.raises(tools.ToolError) as e:
        tools.simulate_holding({**hold, 'current_price': None, 'symbol': 'ABC'})
    assert "Couldn't get prices for ABC right now." in str(e.value)
    assert 'Ask the user' in str(e.value)
    assert 'Advanced' not in str(e.value)      # that hint is for the web form


@pytest.mark.parametrize('change, msg', [
    ({'days': 0}, 'days must be at least 1'),
    ({'days': 253}, 'days must be 252 or less'),
    ({'days': 10.5}, 'days must be a whole number'),
    ({'days': 'soon'}, 'days must be a number'),
    ({'expected_return_percent': -101}, 'expected_return_percent must be at least -100'),
    ({'expected_return_percent': 250}, 'expected_return_percent must be 200 or less'),
    ({'volatility_percent': -1}, 'volatility_percent must be at least 0'),
    ({'volatility_percent': 501}, 'volatility_percent must be 500 or less'),
    ({'current_price': 0}, 'current_price must be more than 0'),
    ({'model': 'magic'}, "model must be 'normal' or 'resample'"),
    ({'shares': 0}, 'shares must be more than 0'),
    ({'tax_rate_percent': 101}, 'tax_rate_percent must be 100 or less'),
])
def test_simulate_rejects_bad_input(nolookup, change, msg):
    with pytest.raises(tools.ToolError, match=msg):
        tools.simulate_holding({**hold, **change})


@pytest.mark.parametrize('bad', ['no good!', '"><script>', 'WAYTOOLONGSYMBOL', '../etc', 'a b'])
def test_symbols_are_checked(monkeypatch, bad):
    fakecloses(monkeypatch)
    with pytest.raises(tools.ToolError, match='symbol must be'):
        tools.get_stock_info(dict(symbol=bad))


@pytest.mark.parametrize('args', [{}, {'symbol': ''}, {'symbol': None}])
def test_symbol_is_required(args):
    with pytest.raises(tools.ToolError, match='symbol is required'):
        tools.get_stock_info(args)


# get_stock_info

def test_get_stock_info(monkeypatch):
    c = fakecloses(monkeypatch, n=400, last=50.0)
    r = tools.get_stock_info(dict(symbol='abc'))
    assert r['symbol'] == 'ABC' and r['latest_close'] == pytest.approx(50.0, abs=1e-4)
    assert 25 < r['yearly_volatility_percent'] < 35 and r['days_of_history'] == 400
    assert r['change_over_last_year_percent'] == pytest.approx((c[-1] / c[-253] - 1) * 100, abs=0.01)
    assert 'Yahoo' in r['source']


def test_get_stock_info_short_history_skips_the_yearly_change(monkeypatch):
    fakecloses(monkeypatch, n=200)
    assert 'change_over_last_year_percent' not in tools.get_stock_info(dict(symbol='ABC'))


def test_get_stock_info_failure(monkeypatch):
    def fail(sym):
        raise prices.PriceError('No price history found for NOPE. Check the symbol.')
    monkeypatch.setattr(prices, 'history', fail)
    with pytest.raises(tools.ToolError) as e:
        tools.get_stock_info(dict(symbol='nope'))
    assert 'No price history found for NOPE' in str(e.value)
    assert 'Ask the user' in str(e.value)
    assert 'Advanced' not in str(e.value)      # that hint is for the web form


# run: what the agent calls

def test_run_gives_a_result_and_no_error():
    out, err = tools.run('calculate_trade', trade)
    assert err is False and out['net_profit'] == 416.5


def test_run_turns_bad_input_into_an_error_the_model_can_read():
    out, err = tools.run('calculate_trade', {**trade, 'shares': -1})
    assert err is True and out == 'shares must be more than 0'


def test_run_unknown_tool():
    out, err = tools.run('delete_everything', {})
    assert err is True and 'Unknown tool' in out and 'calculate_trade' in out


@pytest.mark.parametrize('bad', [None, 'text', 5, [1, 2]])
def test_run_needs_an_object(bad):
    out, err = tools.run('calculate_trade', bad)
    assert err is True and 'JSON object' in out


def test_run_hides_crashes(monkeypatch, caplog):
    def broken(args):
        raise RuntimeError('secret internal detail')
    monkeypatch.setitem(tools.functions, 'calculate_trade', broken)
    out, err = tools.run('calculate_trade', trade)
    assert err is True and 'secret' not in out and 'unexpected error' in out
    assert 'secret internal detail' in caplog.text     # but it is logged for the developer


# the definitions the model sees

def test_tool_definitions_are_well_formed():
    assert [t['name'] for t in tools.tooldefs] == list(tools.functions)
    for t in tools.tooldefs:
        s = t['input_schema']
        assert s['type'] == 'object' and s['additionalProperties'] is False
        assert t['description'].strip()
        assert set(s['required']) <= set(s['properties'])
        assert all(p['description'].strip() for p in s['properties'].values())
        assert 'strict' not in t


def test_the_schemas_ask_for_exactly_what_the_functions_read():
    props = {t['name']: set(t['input_schema']['properties']) for t in tools.tooldefs}
    assert props['calculate_trade'] == set(trade)
    assert props['simulate_holding'] == set(hold) | {'symbol', 'expected_return_percent', 'model'}
    assert props['get_stock_info'] == {'symbol'}


def test_limits_match_the_web_forms():
    # same numbers whether you type them in or the assistant sends them
    pairs = dict(shares='allotment', buy_price='iniprice', sell_price='fnlprice', buy_commission='buycmm',
                 sell_commission='sellcmm', tax_rate_percent='cptlgain', expected_return_percent='drift',
                 current_price='curprice', volatility_percent='vol')
    fields = {f[0]: f[2:5] for f in app.simfields + app.calcfields}
    for tool, form in pairs.items():
        assert tools.limits[tool] == fields[form], tool
