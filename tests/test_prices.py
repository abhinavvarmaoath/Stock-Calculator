import types

import numpy as np
import pandas as pd
import pytest

import prices


def fakeyf(monkeypatch, df=None, boom=None):
    # stand in for yfinance so the tests don't need the internet
    calls = []

    class Ticker:
        def __init__(self, sym):
            calls.append(sym)

        def history(self, **kw):
            if boom:
                raise boom
            return df

    monkeypatch.setitem(__import__('sys').modules, 'yfinance', types.SimpleNamespace(Ticker=Ticker))
    return calls


@pytest.fixture(autouse=True)
def fresh_cache():
    prices.cache.clear()


def closes(n, nan_at=None):
    c = 100 + np.arange(n, dtype=float)
    if nan_at is not None:
        c[nan_at] = np.nan
    return pd.DataFrame({'Close': c})


def test_history_gives_clean_closes(monkeypatch):
    fakeyf(monkeypatch, df=closes(100, nan_at=5))
    c = prices.history('ABC')
    assert len(c) == 99
    assert not np.isnan(c).any()
    assert c[-1] == 199


def test_history_is_cached(monkeypatch):
    calls = fakeyf(monkeypatch, df=closes(100))
    prices.history('ABC')
    prices.history('ABC')
    assert calls == ['ABC']
    prices.history('XYZ')
    assert calls == ['ABC', 'XYZ']


def test_cache_expires(monkeypatch):
    calls = fakeyf(monkeypatch, df=closes(100))
    prices.history('ABC')
    when, c = prices.cache['ABC']
    prices.cache['ABC'] = (when - prices.ttl - 1, c)
    prices.history('ABC')
    assert calls == ['ABC', 'ABC']


def test_network_error(monkeypatch):
    fakeyf(monkeypatch, boom=ConnectionError('no route to host'))
    with pytest.raises(prices.PriceError) as e:
        prices.history('ABC')
    assert "Couldn't get prices for ABC" in str(e.value)
    assert 'no route' not in str(e.value)       # raw error text isn't shown to the user


def test_history_asks_yfinance_to_raise_its_errors(monkeypatch):
    seen = {}

    class Ticker:
        def __init__(self, sym):
            pass

        def history(self, **kw):
            seen.update(kw)
            return closes(100)

    monkeypatch.setitem(__import__('sys').modules, 'yfinance', types.SimpleNamespace(Ticker=Ticker))
    prices.history('ABC')
    assert seen['raise_errors'] is True


def test_yahoo_not_knowing_the_symbol_is_not_a_network_problem(monkeypatch):
    class YFTickerMissingError(Exception):
        pass

    class YFPricesMissingError(Exception):
        pass

    for err in (YFTickerMissingError('possibly delisted'), YFPricesMissingError('no data')):
        fakeyf(monkeypatch, boom=err)
        with pytest.raises(prices.PriceError, match='No price history found for NOPE'):
            prices.history('NOPE')
        prices.cache.clear()


def test_unknown_symbol(monkeypatch):
    fakeyf(monkeypatch, df=pd.DataFrame())
    with pytest.raises(prices.PriceError) as e:
        prices.history('NOPE')
    assert 'No price history found for NOPE' in str(e.value)


def test_not_enough_history(monkeypatch):
    fakeyf(monkeypatch, df=closes(30))
    with pytest.raises(prices.PriceError) as e:
        prices.history('NEW')
    assert 'Not enough price history' in str(e.value)


def test_failed_lookups_are_not_cached(monkeypatch):
    fakeyf(monkeypatch, boom=ConnectionError())
    with pytest.raises(prices.PriceError):
        prices.history('ABC')
    assert 'ABC' not in prices.cache


def test_error_has_a_short_form_for_the_agent():
    e = prices.PriceError("Couldn't get prices for ABC right now.")
    assert e.what == "Couldn't get prices for ABC right now."
    assert 'Advanced' in str(e) and 'Advanced' not in e.what


def history_of(c):
    return lambda sym: np.asarray(c, dtype=float)


def test_resolve_needs_no_lookup_when_everything_is_typed_in(monkeypatch):
    def boom(sym):
        raise AssertionError('should not look anything up')
    monkeypatch.setattr(prices, 'history', boom)
    price, vol, hist, src = prices.resolve('ABC', 12.0, 0.35, False)
    assert (price, vol, hist) == (12.0, 0.35, None)
    assert src == dict(price='entered by you', vol='entered by you')


def test_resolve_fills_in_only_what_is_missing(monkeypatch):
    c = 50 * np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.02, 300)))
    monkeypatch.setattr(prices, 'history', history_of(c))
    price, vol, hist, src = prices.resolve('ABC', None, 0.35, False)
    assert price == pytest.approx(c[-1]) and vol == 0.35
    assert src == dict(price='latest close', vol='entered by you')
    price, vol, hist, src = prices.resolve('ABC', 12.0, None, False)
    assert price == 12.0 and vol == pytest.approx(0.02 * np.sqrt(252), rel=0.15)
    assert src['price'] == 'entered by you' and 'daily closes' in src['vol']
    assert hist is None


def test_resolve_gives_past_returns_when_resampling(monkeypatch):
    c = 50 * np.exp(np.cumsum(np.random.default_rng(2).normal(0, 0.02, 300)))
    monkeypatch.setattr(prices, 'history', history_of(c))
    price, vol, hist, src = prices.resolve('ABC', 12.0, 0.35, True)    # both typed in, still looks up
    assert price == 12.0 and vol == 0.35
    assert len(hist) == len(c) - 1


def test_resolve_passes_lookup_errors_on(monkeypatch):
    def fail(sym):
        raise prices.PriceError('nope')
    monkeypatch.setattr(prices, 'history', fail)
    with pytest.raises(prices.PriceError):
        prices.resolve('ABC', None, None, False)


def test_stats():
    # made up prices with a known 30% yearly vol
    r = np.random.default_rng(4).normal(0, 0.30 / np.sqrt(252), 5000)
    c = 50 * np.exp(np.cumsum(r))
    last, vol, hist = prices.stats(c)
    assert last == pytest.approx(c[-1])
    assert vol == pytest.approx(0.30, rel=0.05)
    assert len(hist) == len(c) - 1
