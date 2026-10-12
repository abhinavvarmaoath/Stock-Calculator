import json
import threading
import time
import warnings

import numpy as np
import pytest

import forecast as fc

fast = ['drift', 'ridge', 'gbm']        # holt and arima take about 0.1 s a forecast, so they are tested on their own


def walk(seed=0, n=1000, drift=0.0003, vol=0.015):
    # log prices of a stock that moves at random
    rng = np.random.default_rng(seed)
    return np.log(100) + np.cumsum(rng.normal(drift, vol, n))


# the models

def test_naive_says_the_price_stays_put():
    lp = walk()
    assert fc.naive(lp, 21) == lp[-1]


def test_drift_keeps_the_pace_of_the_last_year():
    lp = 0.001 * np.arange(600.0)
    assert fc.drift(lp, 10) == pytest.approx(lp[-1] + 0.010)


def test_drift_ignores_what_happened_before_the_last_year():
    # slow for the first 300 days, 3 times faster for the last 300
    lp = np.concatenate([0.001 * np.arange(300.0), 0.3 + 0.003 * np.arange(1, 301.0)])
    assert fc.drift(lp, 10) == pytest.approx(lp[-1] + 0.030)


@pytest.mark.parametrize('model', ['holt', 'arima'])
def test_trend_models_carry_a_steady_climb_forward(model):
    rng = np.random.default_rng(5)
    lp = 0.001 * np.arange(600.0) + np.cumsum(rng.normal(0, 0.0005, 600))
    got = getattr(fc, model)(lp, 10) - lp[-1]
    assert 0.004 < got < 0.016              # a straight line of 0.001 a day is 0.010 in 10 days


def test_fits_that_complain_are_quiet():
    # some windows make arima complain about its starting values. none of that should reach the log
    from statsmodels.tsa.arima.model import ARIMA
    lp = walk(5, 900)
    noisy = []
    for t in range(504, 800, 6):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ARIMA(lp[:t + 1][-504:], order=(1, 1, 1), trend='t').fit()
        if w:
            noisy.append(t)
            with warnings.catch_warnings(record=True) as again:
                warnings.simplefilter('always')
                fc.arima(lp[:t + 1], 21)
            assert again == []
        if len(noisy) == 2:
            break
    assert noisy                                                # there was something to be quiet about


@pytest.mark.parametrize('model, where, cls', [('holt', 'statsmodels.tsa.holtwinters', 'ExponentialSmoothing'),
                                              ('arima', 'statsmodels.tsa.arima.model', 'ARIMA')])
def test_forecasts_running_together_leave_the_warning_filters_as_they_were(monkeypatch, model, where, cls):
    # catch_warnings changes the filters for everyone, so fits that overlap can leave them changed for good.
    # the real fits are swapped for slow ones, and the three start 30 ms apart, so each is certain to start
    # inside the one before and to end in the same order: the worst case
    import importlib

    class Slow:
        def __init__(self, *a, **k):
            pass

        def fit(self, *a, **k):
            time.sleep(0.15)
            return self

        def forecast(self, h):
            return np.zeros(h)
    monkeypatch.setattr(importlib.import_module(where), cls, Slow)

    lp = walk(2, 700)
    before = list(warnings.filters)

    def fit(i):
        time.sleep(0.03 * i)
        getattr(fc, model)(lp, 5)
    threads = [threading.Thread(target=fit, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert list(warnings.filters) == before


def test_features_only_use_the_past():
    lp = walk()
    other = lp.copy()
    other[401:] += np.random.default_rng(1).normal(0, 1, len(lp) - 401)
    assert fc.features(lp, 400) == fc.features(other, 400)
    assert len(fc.features(lp, 400)) == 7


def test_ridge_matches_a_plain_loop_version():
    # the learner is vectorised. this builds the same training set the slow obvious way:
    # a day only counts if its outcome (the price h days later) is already known on day t
    lp, h = walk(3, 700), 21
    t = 600
    L = fc.Learner('ridge')
    got = L(lp[:t + 1], h)

    X, y = [], []
    for s in range(fc.first, t + 1):
        if s + h <= t:
            X.append(fc.features(lp, s))
            y.append(lp[s + h] - lp[s])
    X, y = np.array(X), np.array(y)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + len(y) * np.eye(7), Z.T @ (y - y.mean()))
    want = lp[t] + y.mean() + ((np.array(fc.features(lp, t)) - mu) / sd) @ w
    assert got == pytest.approx(want)


def test_learner_only_goes_forward():
    lp = walk(2, 700)
    L = fc.Learner('ridge')
    L(lp[:601], 5)
    with pytest.raises(ValueError):
        L(lp[:501], 5)


def test_learner_needs_history():
    with pytest.raises(ValueError):
        fc.Learner('ridge')(walk(2, 250), 21)


@pytest.mark.parametrize('kind', ['ridge', 'gbm'])
def test_learner_finds_a_relationship_that_is_really_there(kind):
    # each month mostly repeats the one before: a return is 0.9 of the return 21 days earlier, plus noise.
    # if the training days and their outcomes were lined up wrongly, the forecasts would not follow this
    rng = np.random.default_rng(1)
    e = rng.normal(0, 0.01, 2500)
    r = np.zeros(2500)
    for i in range(21, 2500):
        r[i] = 0.9 * r[i - 21] + e[i]
    lp = np.cumsum(r)

    L = fc.Learner(kind)
    days = range(1500, 2400, 30)
    said = np.array([L(lp[:t + 1], 21) - lp[t] for t in days])
    moved = np.array([lp[t + 21] - lp[t] for t in days])
    assert np.corrcoef(said, moved)[0, 1] > 0.4


# the test itself

def test_a_short_history_is_refused_with_how_much_is_needed():
    for h in fc.horizons:
        lp = walk(1, fc.minimum(h) - 1)
        with pytest.raises(fc.TooShort) as e:
            fc.backtest(lp, h, ['naive'])
        assert '{:,}'.format(fc.minimum(h)) in str(e.value)


@pytest.mark.parametrize('h', sorted(fc.horizons))
def test_the_least_history_still_gives_30_forecasts(h):
    bt = fc.backtest(walk(1, fc.minimum(h)), h, ['naive'])
    assert len(bt['origins']) >= 30


@pytest.mark.parametrize('h', sorted(fc.horizons))
def test_forecasts_start_after_the_warm_up_and_stay_inside_the_history(h):
    lp = walk(4, 1258)
    bt = fc.backtest(lp, h, ['naive'])
    o = bt['origins']
    assert o[0] == fc.warm
    assert o[-1] + h <= len(lp) - 1
    assert (np.diff(o) == bt['step']).all()
    assert bt['step'] >= h // 3
    assert np.array_equal(bt['actual'], lp[o + h]) and np.array_equal(bt['last'], lp[o])


def test_naive_is_always_in_and_scores_zero():
    bt = fc.backtest(walk(1, 800), 5, ['drift'])
    assert list(bt['preds']) == ['naive', 'drift']
    assert np.array_equal(bt['preds']['naive'], bt['last'])
    row = fc.score(bt)[0]
    assert row['name'] == 'naive' and row['skill'] == 0.0
    assert row['closer'] is None and row['direction'] is None


def test_prices_that_are_not_numbers_are_refused():
    lp = walk(1, 800)
    lp[10] = np.nan
    with pytest.raises(ValueError):
        fc.backtest(lp, 5, ['naive'])


def test_same_prices_same_answers():
    lp = walk(6, 800)
    a, b = fc.backtest(lp, 21, fast), fc.backtest(lp, 21, fast)
    for name in a['preds']:
        assert np.array_equal(a['preds'][name], b['preds'][name])


# no look-ahead

def leaks(lp, h, cut, which=fast):
    # models whose forecasts from day cut or earlier change when every price after cut is replaced by junk
    junk = lp.copy()
    junk[cut + 1:] = lp[cut] + np.cumsum(np.random.default_rng(99).normal(0, 0.3, len(lp) - cut - 1))
    a, b = fc.backtest(lp, h, which), fc.backtest(junk, h, which)
    early = a['origins'] <= cut
    assert early.sum() >= 5
    return [n for n in a['preds']
            if not np.array_equal(a['preds'][n][early], b['preds'][n][early], equal_nan=True)]


@pytest.mark.parametrize('h', [5, 21])
def test_no_model_can_see_past_the_day_it_forecasts_from(h):
    assert leaks(walk(7, 800), h, cut=650) == []


def test_the_check_notices_a_model_that_peeks(monkeypatch):
    # prove the check above can fail: a model that reads the real future has to be caught
    seen = {}

    def peek(lp, h):
        return seen['full'][len(lp) - 1 + h]        # the price h days on, from the whole history

    real_make, real_backtest = fc.make, fc.backtest
    monkeypatch.setattr(fc, 'make', lambda name: peek if name == 'peek' else real_make(name))

    def recording(lp, h, which=None):
        seen['full'] = np.asarray(lp)
        return real_backtest(lp, h, which)
    monkeypatch.setattr(fc, 'backtest', recording)

    assert leaks(walk(7, 800), 5, cut=650, which=['peek']) == ['peek']


# failures

def test_a_forecast_that_fails_is_counted_and_not_scored(monkeypatch):
    calls = {'n': 0}

    def flaky(lp, h):
        calls['n'] += 1
        if calls['n'] % 4 == 0:
            raise RuntimeError('did not converge')
        if calls['n'] % 4 == 1:
            return float('inf')
        return lp[-1] + 0.01

    real = fc.make
    monkeypatch.setattr(fc, 'make', lambda name: flaky if name == 'flaky' else real(name))
    bt = fc.backtest(walk(1, 800), 5, ['flaky'])
    n = len(bt['origins'])
    assert bt['failed']['flaky'] == sum(1 for i in range(1, n + 1) if i % 4 in (0, 1))      # the raises and the infs
    assert bt['failed']['naive'] == 0
    row = [r for r in fc.score(bt) if r['name'] == 'flaky'][0]
    assert row['n'] == n - bt['failed']['flaky']
    assert row['failed'] == bt['failed']['flaky']


def test_a_model_that_never_forecasts_gets_an_empty_row(monkeypatch):
    def broken(lp, h):
        raise RuntimeError('no')

    real = fc.make
    monkeypatch.setattr(fc, 'make', lambda name: broken if name == 'broken' else real(name))
    row = [r for r in fc.score(fc.backtest(walk(1, 800), 5, ['broken'])) if r['name'] == 'broken'][0]
    assert row['n'] == 0 and row['miss'] is None and row['skill'] is None


# scoring

def result(preds, actual, last=None):
    n = len(actual)
    last = np.zeros(n) if last is None else np.array(last, dtype=float)
    p = {k: np.array(v, dtype=float) for k, v in preds.items()}
    p['naive'] = last.copy()
    return dict(h=5, step=5, origins=np.arange(n), last=last, actual=np.array(actual, dtype=float), preds=p,
                failed={k: int(np.isnan(v).sum()) for k, v in p.items()})


def row(rows, name):
    return [r for r in rows if r['name'] == name][0]


def test_score_numbers_worked_by_hand():
    # naive misses by .10 .10 .05 .05 (average .075). model a misses by .02 .08 .00 .10 (average .05)
    r = row(fc.score(result(dict(a=[0.08, -0.02, 0.05, 0.05]), [0.10, -0.10, 0.05, -0.05])), 'a')
    assert r['miss'] == pytest.approx(5.0)
    assert r['skill'] == pytest.approx(100 * (1 - 0.05 / 0.075))
    assert r['closer'] == pytest.approx(75.0)               # nearer than naive on 3 of 4
    assert r['direction'] == pytest.approx(75.0)            # up, down, up, up against up, down, up, down
    assert r['n'] == 4 and r['failed'] == 0


def test_score_compares_only_the_days_a_model_forecast():
    # the missing forecast is dropped from the model and from naive's side of the comparison
    r = row(fc.score(result(dict(a=[0.08, np.nan, 0.05, 0.05]), [0.10, -0.10, 0.05, -0.05])), 'a')
    assert r['n'] == 3 and r['failed'] == 1
    assert r['skill'] == pytest.approx(100 * (1 - (0.02 + 0.0 + 0.10) / 3 / ((0.10 + 0.05 + 0.05) / 3)))


def test_direction_is_up_or_down_from_where_the_price_is_today():
    # the price is 2.0 today. a is below it where the price then rose, and above it where it fell
    r = row(fc.score(result(dict(a=[1.95, 2.05]), [2.1, 1.9], last=[2.0, 2.0])), 'a')
    assert r['direction'] == 0.0
    r = row(fc.score(result(dict(a=[2.05, 1.95]), [2.1, 1.9], last=[2.0, 2.0])), 'a')
    assert r['direction'] == 100.0


def test_a_model_that_calls_no_direction_has_no_direction_score():
    r = row(fc.score(result(dict(a=[0.0, 0.0, 0.0, 0.0]), [0.1, -0.1, 0.1, -0.1])), 'a')
    assert r['direction'] is None
    assert r['skill'] == pytest.approx(0.0)


def test_a_model_exactly_as_close_as_naive_is_not_closer():
    r = row(fc.score(result(dict(a=[0.0, 0.0]), [0.1, -0.1])), 'a')
    assert r['closer'] == 0.0


def test_a_flat_price_has_no_skill_score():
    r = row(fc.score(result(dict(a=[0.01, 0.01]), [0.0, 0.0])), 'a')
    assert r['skill'] is None and r['miss'] == pytest.approx(1.0)


# the scoreboard

def test_the_scoreboard_is_plain_data_the_page_can_use():
    lp = walk(1, 800)
    out = fc.scoreboard(lp, 21, ['drift', 'ridge'])
    json.dumps(out)                                     # fails on a numpy number
    assert out['h'] == 21 and out['horizon'] == '1 month' and out['prices'] == 800
    assert [r['name'] for r in out['rows']] == ['naive', 'drift', 'ridge']
    assert out['forecasts'] == out['rows'][0]['n'] and out['every'] >= 7
    assert [c['level'] for c in out['calibration']] == [80, 90]
    assert out['headline'] and out['rows'][1]['edge'] in ('better', 'worse', 'tied')


def test_the_scoreboard_takes_a_plain_list_of_prices():
    lp = walk(1, 800)
    assert fc.scoreboard(list(lp), 5, ['drift']) == fc.scoreboard(lp, 5, ['drift'])


# the memory and the limit on how many run at once

@pytest.fixture
def counted(monkeypatch):
    # scoreboard() replaced by something instant that counts how often it ran
    fc.cache.clear()
    calls = []

    def fake(lp, h, which=None):
        calls.append(h)
        return dict(h=h, n=len(lp))
    monkeypatch.setattr(fc, 'scoreboard', fake)
    yield calls
    fc.cache.clear()


def test_the_same_question_is_answered_from_memory(counted):
    lp = walk(1, 800)
    assert fc.board(lp, 21) == fc.board(lp.copy(), 21) == dict(h=21, n=800)
    assert counted == [21]


def test_known_says_whether_asking_is_free(counted):
    lp = walk(1, 800)
    assert not fc.known(lp, 21)
    fc.board(lp, 21)
    assert fc.known(lp, 21) and not fc.known(lp, 5)


def test_other_prices_or_another_horizon_are_a_new_question(counted):
    lp = walk(1, 800)
    other = lp.copy()
    other[-1] += 0.01
    fc.board(lp, 21)
    fc.board(other, 21)
    fc.board(lp, 5)
    assert counted == [21, 21, 5]


def test_the_oldest_answer_is_forgotten_first(counted, monkeypatch):
    monkeypatch.setattr(fc, 'keep', 3)
    prices = [walk(i, 800) for i in range(4)]
    for p in prices:
        fc.board(p, 5)
    assert len(fc.cache) == 3
    fc.board(prices[3], 5)                      # still remembered
    fc.board(prices[0], 5)                      # not any more
    assert counted == [5] * 5


def test_when_all_the_places_are_taken_a_new_question_is_turned_away(counted):
    lp = walk(1, 800)
    fc.board(lp, 21)
    taken = [fc.slots.acquire(), fc.slots.acquire()]
    try:
        with pytest.raises(fc.Busy):
            fc.board(walk(2, 800), 21)
        assert fc.board(lp, 21) == dict(h=21, n=800)        # a remembered answer needs no place
    finally:
        for _ in taken:
            fc.slots.release()
    assert fc.board(walk(2, 800), 21)


def test_the_place_is_given_back_when_a_scoreboard_fails():
    fc.cache.clear()
    with pytest.raises(fc.TooShort):
        fc.board(walk(1, 300), 21)
    with pytest.raises(fc.TooShort):
        fc.board(walk(1, 300), 21)
    with pytest.raises(fc.TooShort):
        fc.board(walk(1, 300), 21)              # a third failure would be Busy if the places were not returned


def test_at_most_two_run_at_once(monkeypatch):
    fc.cache.clear()
    running, peak = [0], [0]
    lock = threading.Lock()

    def slow(lp, h, which=None):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        time.sleep(0.3)
        with lock:
            running[0] -= 1
        return dict(h=h)
    monkeypatch.setattr(fc, 'scoreboard', slow)

    outcome = []
    go = threading.Barrier(5)

    def ask(i):
        go.wait()
        try:
            fc.board(walk(i, 800), 21)
            outcome.append('ran')
        except fc.Busy:
            outcome.append('busy')
    threads = [threading.Thread(target=ask, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcome) == ['busy', 'busy', 'busy', 'ran', 'ran'] and peak[0] == 2
    fc.cache.clear()


def test_every_method_runs_from_start_to_finish():
    # arima and holt are slow, so this is the one place they run inside a whole scoreboard
    out = fc.scoreboard(walk(5, fc.minimum(21) + 5), 21)
    assert [r['name'] for r in out['rows']] == fc.names
    assert all(r['n'] == out['forecasts'] and r['failed'] == 0 for r in out['rows'])
    json.dumps(out)
