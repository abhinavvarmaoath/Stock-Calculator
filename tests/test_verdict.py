import numpy as np
import pytest

import forecast as fc
import verdict as vd


def windows(h, step, n, rng, shift=0.0):
    # how much better than naive each forecast did, when every day's luck is independent. a forecast
    # covers h days, so forecasts less than h days apart share some days and their results move together
    o = np.arange(n) * step
    z = np.concatenate([[0], np.cumsum(rng.standard_normal(o[-1] + h))])
    return (z[o + h] - z[o]) / np.sqrt(h) + shift, o


# the test

def test_no_edge_gives_a_large_p_and_a_clear_edge_a_small_one():
    rng = np.random.default_rng(1)
    d = rng.normal(0, 1, 60)
    o = np.arange(60) * 10
    assert vd.dm(d - d.mean(), o, 5)['p'] > 0.9
    assert vd.dm(d + 2, o, 5)['p'] < 1e-6


def test_the_sign_of_t_says_who_was_better_and_p_does_not_care():
    d = np.random.default_rng(2).normal(0.3, 1, 60)
    o = np.arange(60) * 10
    a, b = vd.dm(d, o, 5), vd.dm(-d, o, 5)
    assert a['t'] > 0 > b['t'] and a['t'] == pytest.approx(-b['t'])
    assert a['p'] == pytest.approx(b['p'])


def test_too_few_forecasts_are_not_tested():
    assert vd.dm(np.ones(vd.fewest - 1), np.arange(vd.fewest - 1) * 10, 5) is None


def test_a_model_that_is_naive_is_exactly_tied():
    r = vd.dm(np.zeros(30), np.arange(30) * 10, 5)
    assert r['p'] == 1.0 and r['mean'] == 0.0


def test_the_same_gap_every_single_time_is_certain():
    o = np.arange(30) * 10
    better, worse = vd.dm(np.full(30, 0.01), o, 5), vd.dm(np.full(30, -0.01), o, 5)
    assert better['p'] == 0.0 and better['mean'] > 0
    assert worse['p'] == 0.0 and worse['mean'] < 0


def test_more_evidence_gives_a_smaller_p():
    rng = np.random.default_rng(3)
    d = rng.normal(0.4, 1, 400)
    ps = [vd.dm(d[:n], np.arange(n) * 10, 5)['p'] for n in (25, 100, 400)]
    assert ps[0] > ps[1] > ps[2]


def test_forecasts_that_overlap_count_for_less():
    # the same numbers, once as forecasts 21 days apart and once as forecasts 3 days apart that
    # share most of their prices. the second set is much less evidence
    d = 0.3 + np.sin(np.arange(60) / 4)
    apart = vd.dm(d, np.arange(60) * 21, 21)
    close = vd.dm(d, np.arange(60) * 3, 21)
    assert close['p'] > 5 * apart['p']
    assert close['dof'] < apart['dof']


@pytest.mark.parametrize('h,step,n', [(5, 10, 75), (21, 10, 74), (63, 21, 33), (21, 3, 120)])
def test_luck_is_called_a_result_about_as_often_as_it_says(h, step, n):
    # nothing is going on, so p under .05 should happen about 5% of the time even when forecasts overlap.
    # the plain version of this test fails the overlapping cases at 10-15%
    rng = np.random.default_rng(h * 1000 + step)
    hits = 0
    for _ in range(600):
        d, o = windows(h, step, n, rng)
        hits += vd.dm(d, o, h)['p'] < 0.05
    assert 0.025 < hits / 600 < 0.10


@pytest.mark.parametrize('h,step,n,least', [(5, 10, 75, 0.9), (21, 10, 74, 0.6)])
def test_a_real_edge_is_usually_found(h, step, n, least):
    rng = np.random.default_rng(h)
    hits = 0
    for _ in range(300):
        d, o = windows(h, step, n, rng, shift=0.5)
        hits += vd.dm(d, o, h)['p'] < 0.05
    assert hits / 300 > least


# trying several models

def test_holm_worked_by_hand():
    # sorted: .005 x4, .01 x3, .03 x2, .04 x1 = .02, .03, .06, .04 (which is raised to the .06 before it)
    got = vd.holm({'a': 0.01, 'b': 0.04, 'c': 0.03, 'd': 0.005})
    assert got == pytest.approx({'d': 0.02, 'a': 0.03, 'c': 0.06, 'b': 0.06})


def test_holm_never_lowers_a_p_and_never_goes_past_one():
    ps = {'a': 0.2, 'b': 0.5, 'c': 0.9, 'd': 0.02}
    got = vd.holm(ps)
    assert got == pytest.approx({'d': 0.08, 'a': 0.6, 'b': 1.0, 'c': 1.0})    # b would be 1.0 and c .9, but c stays above b
    assert all(got[k] >= ps[k] for k in ps)


def test_holm_stops_at_one():
    assert vd.holm({'a': 0.6, 'b': 0.7}) == pytest.approx({'a': 1.0, 'b': 1.0})     # .6 x 2 is 1.2


def test_holm_with_one_model_changes_nothing():
    assert vd.holm({'a': 0.03}) == {'a': 0.03}
    assert vd.holm({}) == {}


# judging a scoreboard

def board(n=40, seed=0):
    # a made-up backtest of 40 forecasts. the price is 0 today and the real move is e (up or down).
    # naive misses by e. a misses by half of that every time, c by double, b by about the same
    rng = np.random.default_rng(seed)
    e = rng.uniform(0.02, 0.10, n) * rng.choice([-1, 1], n)
    last = np.zeros(n)
    miss_b = np.abs(e) * (1 + 0.3 * rng.choice([-1, 1], n))
    preds = dict(naive=last, a=0.5 * e, b=e - np.sign(e) * miss_b, c=-e)
    return dict(h=5, step=10, origins=np.arange(n) * 10 + 500, last=last, actual=e, preds=preds,
                failed={k: 0 for k in preds})


def judged(bt):
    rows = fc.score(bt)
    out = vd.judge(bt, rows)
    return {r['name']: r for r in rows}, out


def test_a_clear_winner_a_clear_loser_and_a_tie():
    rows, out = judged(board())
    assert rows['a']['edge'] == 'better' and rows['c']['edge'] == 'worse' and rows['b']['edge'] == 'tied'
    assert rows['naive']['edge'] is None and rows['naive']['p'] is None
    for k in 'abc':
        assert 0 <= rows[k]['p'] <= rows[k]['padj'] <= 1
        assert rows[k]['needs'] > 0


def test_the_winner_is_named_in_the_headline():
    rows, out = judged(board())
    assert out['headline'].startswith('a cut the average miss by 50% compared with "price stays put"')
    assert 'No model' not in out['headline']


def test_with_two_winners_the_better_one_comes_first_and_the_other_is_mentioned():
    bt = board()
    bt['preds']['best'] = 0.7 * bt['actual']                        # misses by .3 of naive's, a by .5
    bt['failed']['best'] = 0
    _, out = judged(bt)
    assert out['headline'].startswith('best cut the average miss by 70%')
    assert out['headline'].endswith(' So did a.')


def test_without_a_winner_the_headline_says_so_and_names_the_losers():
    bt = board()
    del bt['preds']['a']
    rows, out = judged(bt)
    assert out['headline'].startswith('No model beat "price stays put"')
    assert 'c did measurably worse' in out['headline']
    assert out['detail'].startswith('With 40 forecasts a model had to cut the average miss by at least about ')
    assert out['detail'].endswith('% to be told apart from luck. Smaller real edges would not show up here.')


def test_needs_is_the_gap_that_would_just_have_counted():
    # if a model had been this much better than naive it would have sat right at the line
    bt = board(80)
    rows, _ = judged(bt)
    base = np.abs(bt['actual'])
    need = rows['b']['needs'] / 100 * base.mean()                   # in price terms
    d = base - np.abs(bt['preds']['b'] - bt['actual'])
    shifted = d - d.mean() + need
    r = vd.dm(shifted, bt['origins'], bt['h'])
    m = 3                                                            # models tested
    assert r['p'] == pytest.approx(0.05 / m, rel=0.01)


def test_when_no_gap_could_ever_count_it_says_that_instead_of_a_percent_over_100():
    rows = [dict(name='naive', edge=None, needs=None, n=40),
            dict(name='x', label='X', edge='tied', needs=140.0, n=40, skill=-3.0),
            dict(name='y', label='Y', edge='tied', needs=210.0, n=33, skill=1.0)]
    out = vd.summary(rows)
    assert out['detail'] == 'With only 33 forecasts none of these comparisons can tell a real edge from luck.'


def test_models_with_no_forecasts_or_too_few_are_left_out_of_the_family():
    bt = board()
    bt['preds']['empty'] = np.full(40, np.nan)
    bt['failed']['empty'] = 40
    bt['preds']['few'] = np.where(np.arange(40) < 5, bt['preds']['a'], np.nan)
    bt['failed']['few'] = 35
    rows, _ = judged(bt)
    assert rows['empty']['edge'] == 'untested' and rows['few']['edge'] == 'untested'
    assert rows['empty']['p'] is None and rows['few']['padj'] is None
    # and the others are adjusted for 3 models, not 5. a has the smallest p, so it is multiplied by 3
    assert rows['a']['padj'] == pytest.approx(3 * rows['a']['p'])


def test_a_model_is_only_better_if_it_survives_trying_several(monkeypatch):
    # p of .02 and .03 each look good alone. with 3 models tried they are really .06 and .06, which is luck
    given = iter([0.02, 0.03, 0.9])
    monkeypatch.setattr(vd, 'dm', lambda d, o, h: dict(mean=0.01, se=0.005, t=2.0, p=next(given), dof=30.0))
    rows, _ = judged(board())
    assert [rows[k]['p'] for k in 'abc'] == [0.02, 0.03, 0.9]
    assert rows['a']['padj'] == pytest.approx(0.06) and rows['b']['padj'] == pytest.approx(0.06)
    assert [rows[k]['edge'] for k in 'abc'] == ['tied', 'tied', 'tied']


def test_a_method_that_is_right_every_time_beats_the_benchmark():
    # the price moves by exactly .05 each time. naive misses by .05 every time and this method never misses
    n = 40
    e = 0.05 * np.where(np.arange(n) % 2 == 0, 1, -1)
    bt = dict(h=5, step=10, origins=np.arange(n) * 10 + 500, last=np.zeros(n), actual=e,
              preds=dict(naive=np.zeros(n), perfect=e), failed=dict(naive=0, perfect=0))
    rows, out = judged(bt)
    assert rows['perfect']['edge'] == 'better' and rows['perfect']['padj'] == 0.0
    assert out['headline'].startswith('perfect cut the average miss by 100%')


def test_a_price_that_never_moved_says_there_is_nothing_to_forecast():
    n = 40
    bt = dict(h=5, step=10, origins=np.arange(n) * 10 + 500, last=np.zeros(n), actual=np.zeros(n),
              preds=dict(naive=np.zeros(n), a=np.zeros(n)), failed=dict(naive=0, a=0))
    rows, out = judged(bt)
    assert out['headline'] == 'The price never moved on any of the forecast days, so there is nothing to forecast.'
    assert rows['a']['edge'] == 'untested' and rows['a']['skill'] is None


def test_a_scoreboard_where_nothing_can_be_tested_says_so():
    bt = board()
    for k in ('a', 'b', 'c'):
        bt['preds'][k] = np.full(40, np.nan)
        bt['failed'][k] = 40
    _, out = judged(bt)
    assert 'too few' in out['headline']


# the science: does the whole thing give the right answer on data where we know it

def climb(seed, n=1258, up=0.0015, vol=0.006):
    # a stock that really does climb about 0.15% a day
    return np.log(100) + np.cumsum(np.random.default_rng(seed).normal(up, vol, n))


def wander(seed, n=1258):
    # a stock with no pattern at all: random moves
    return np.log(100) + np.cumsum(np.random.default_rng(seed).normal(0.0003, 0.015, n))


def edges(lp, h):
    bt = fc.backtest(lp, h, ['drift', 'ridge'])
    rows = fc.score(bt)
    vd.judge(bt, rows)
    return {r['name']: r['edge'] for r in rows}


@pytest.mark.parametrize('h', [5, 21])
def test_a_random_walk_has_no_winner(h):
    wins = sum(e == 'better' for seed in range(14) for e in edges(wander(seed), h).values())
    assert wins == 0


def test_a_real_trend_is_found_over_a_month():
    # 74 overlapping forecasts is not much evidence, so even a 30% better model is not always
    # enough. most are found, every series has a winner, and nothing is called worse
    found = [edges(climb(seed), 21) for seed in range(6)]
    flat = [e for f in found for e in f.values()]
    assert flat.count('better') >= 9 and 'worse' not in flat
    assert all('better' in f.values() for f in found)


# the simulator's range

def gbm(seed, n=1700, vol=0.25, scale=None):
    # prices that move the way the simulator assumes. scale changes the volatility from day to day
    rng = np.random.default_rng(seed)
    sd = vol / np.sqrt(252) * (np.ones(n) if scale is None else scale)
    return np.log(100) + np.cumsum(-0.5 * sd ** 2 + sd * rng.standard_normal(n))


def ranges(lp, h=21):
    bt = fc.backtest(lp, h, ['naive'])
    return {c['level']: c for c in vd.calibrate(lp, bt)}


def test_ranges_hold_the_real_price_as_often_as_claimed_when_the_model_is_true():
    got = [ranges(gbm(seed)) for seed in range(16)]
    for lvl in (80, 90):
        assert np.mean([g[lvl]['covered'] for g in got]) == pytest.approx(lvl, abs=3)
    assert all(g[lvl]['p'] > 0.001 for g in got for lvl in (80, 90))


def test_ranges_that_are_too_narrow_are_called_narrow():
    # the simulator expects 0% a year, so a stock that keeps climbing keeps landing above its range
    r = ranges(climb(1))
    assert r[80]['covered'] < 66 and r[80]['verdict'] == 'narrow'
    assert r[90]['covered'] < r[90]['lo'] and r[90]['verdict'] == 'narrow'


def test_ranges_that_are_too_wide_are_called_wide():
    # wild for the 2 years the volatility is worked out from, then calm
    n = 1000
    r = ranges(gbm(1, n=n, scale=np.where(np.arange(n) < 500, 4.0, 1.0)))
    assert r[80]['covered'] > 95 and r[80]['verdict'] == 'wide'


def test_calibration_numbers_and_how_many_forecasts_are_really_separate():
    lp = gbm(2)
    for h in (5, 21, 63):
        bt = fc.backtest(lp, h, ['naive'])
        n = len(bt['origins'])
        c = vd.calibrate(lp, bt)[0]
        assert c['n'] == n and c['level'] == 80
        assert c['indep'] == (n if bt['step'] >= h else round(n * bt['step'] / h))
        assert c['lo'] <= c['hi'] and 0 <= c['covered'] <= 100


def test_volatility_on_a_day_only_uses_earlier_prices():
    lp = gbm(4)
    days = np.arange(600, 1200, 50)
    junk = lp.copy()
    junk[900 + 1:] += np.random.default_rng(5).normal(0, 1, len(lp) - 901)
    early = days <= 900
    assert early.sum() >= 5
    assert np.array_equal(vd.vols(lp, days)[early], vd.vols(junk, days)[early])
    assert not np.array_equal(vd.vols(lp, days)[~early], vd.vols(junk, days)[~early])


def test_volatility_is_the_yearly_figure_the_simulator_uses():
    assert vd.vols(gbm(3, n=3000, vol=0.25), [2500])[0] == pytest.approx(0.25, rel=0.1)
    a = 0.01                                                         # moves of exactly +1% and -1%
    lp = np.cumsum(np.tile([a, -a], 400))
    assert vd.vols(lp, [600])[0] == pytest.approx(a * (504 / 503) ** 0.5 * 252 ** 0.5)


def test_the_edges_of_the_range_worked_by_hand():
    # daily moves of +-1% give a known volatility, so the range edges are known. for 21 days the 80% range
    # is the middle -.5 vol^2 h/252 plus or minus 1.2816 standard deviations
    a = 0.01
    lp = np.cumsum(np.tile([a, -a], 400))
    h, t = 21, 600
    vol = a * (504 / 503) ** 0.5 * 252 ** 0.5
    sd, mid = vol * (h / 252) ** 0.5, -0.5 * vol ** 2 * h / 252
    z = 1.2815515655446004

    def inside(move):
        bt = dict(h=h, step=h, origins=np.array([t]), last=np.array([0.0]), actual=np.array([move]))
        return vd.calibrate(lp, bt, levels=(80,))[0]['covered'] == 100.0

    assert inside(mid + 0.998 * z * sd) and inside(mid - 0.998 * z * sd)
    assert not inside(mid + 1.002 * z * sd) and not inside(mid - 1.002 * z * sd)
