import numpy as np
from scipy import stats

# Is a better score real, or luck? A model that beats "price stays put" on 70 forecasts
# can easily be flattered by which months happened to be in the test.

alpha = 0.05
fewest = 8          # fewer forecasts than this and there is nothing to test


def dm(d, o, h):
    # Diebold-Mariano test that the average of d is not zero. d is how much better a model did than
    # naive on each forecast (naive's miss minus its miss), o the day each forecast was made and h
    # how many days ahead it looked.
    # forecasts made less than h days apart share some of the same prices, so their misses move
    # together. the variance allows for that with weights that fade to nothing 2h days apart.
    # the plain h under-counted it and called luck a result too often (10-15% of the time in
    # simulations, where 5% is the aim). 2h gets that down to 5-8%
    d, o = np.asarray(d, dtype=float), np.asarray(o, dtype=float)
    n = len(d)
    if n < fewest:
        return None
    mean = float(d.mean())
    z = d - mean
    w = np.clip(1 - np.abs(o[:, None] - o[None, :]) / (2 * h), 0, None)
    var = float(z @ w @ z) / n ** 2
    if var < 1e-24:
        # every forecast was the same distance from naive's: exactly as good as naive every time (p of 1),
        # or better or worse by the same amount every single time (p of 0)
        return dict(mean=mean, se=0.0, t=0.0, p=1.0 if abs(mean) < 1e-12 else 0.0, dof=float(n - 1))
    se = var ** 0.5
    dof = max(n * n / float(w.sum()) - 1, 2)        # forecasts that overlap count for less than one each
    t = mean / se
    return dict(mean=mean, se=se, t=float(t), p=float(2 * stats.t.sf(abs(t), dof)), dof=float(dof))


def holm(ps):
    # p values after allowing for having tried several models. the chance that any of them looks
    # better than naive by luck stays under alpha. ps is {name: p}
    order = sorted(ps, key=ps.get)
    out, top = {}, 0.0
    for i, k in enumerate(order):
        top = max(top, min(1.0, (len(order) - i) * ps[k]))
        out[k] = top
    return out


def judge(bt, rows):
    # adds to each row of score(): p (the chance of a gap this big by luck), padj (the same after
    # allowing for the number of models tried), edge (better, worse or tied with naive) and needs
    # (how much less than naive's miss it would have taken to count)
    actual, o, h = bt['actual'], bt['origins'], bt['h']
    base = np.abs(bt['preds']['naive'] - actual)

    tests, basemean = {}, {}
    for r in rows:
        r.update(p=None, padj=None, edge=None, needs=None)
        if r['name'] == 'naive':
            continue
        r['edge'] = 'untested'
        if not r['n']:
            continue
        p = bt['preds'][r['name']]
        ok = np.isfinite(p)
        t = dm(base[ok] - np.abs(p[ok] - actual[ok]), o[ok], h)
        if t and base[ok].mean() > 0:
            tests[r['name']], basemean[r['name']] = t, base[ok].mean()

    if not tests and not (base > 0).any():
        return dict(headline='The price never moved on any of the forecast days, so there is nothing to forecast.', detail='')

    adj = holm({k: t['p'] for k, t in tests.items()})
    for r in rows:
        t = tests.get(r['name'])
        if not t:
            continue
        r['p'], r['padj'] = t['p'], adj[r['name']]
        r['edge'] = 'tied'
        if r['padj'] < alpha:
            r['edge'] = 'better' if t['mean'] > 0 else 'worse'
        # the gap needed to beat the strictest bar, which is the one the best model faces
        need = stats.t.ppf(1 - alpha / len(tests) / 2, t['dof']) * t['se']
        r['needs'] = float(100 * need / basemean[r['name']])
    return summary(rows)


def summary(rows):
    # what to tell people, in plain words
    tested = [r for r in rows if r['needs'] is not None]
    if not tested:
        return dict(headline='There were too few forecasts to compare the models.', detail='')

    better = sorted((r for r in rows if r['edge'] == 'better'), key=lambda r: -r['skill'])
    worse = [r for r in rows if r['edge'] == 'worse']
    if better:
        b = better[0]
        head = '{} cut the average miss by {:.0f}% compared with "price stays put", more than luck is likely to explain.'.format(
            b['label'], b['skill'])
        if len(better) > 1:
            head += ' So did {}.'.format(', '.join(r['label'] for r in better[1:]))
    else:
        head = 'No model beat "price stays put" by more than luck can explain.'
        if worse:
            head += ' {} did measurably worse.'.format(', '.join(r['label'] for r in worse))

    least, n = min(r['needs'] for r in tested), min(r['n'] for r in tested)
    if least >= 100:
        # a method cannot cut its miss by more than all of it, so no gap would have counted
        detail = 'With only {} forecasts none of these comparisons can tell a real edge from luck.'.format(n)
    else:
        detail = ('With {} forecasts a model had to cut the average miss by at least about {:.0f}% to be told apart from '
                  'luck. Smaller real edges would not show up here.').format(n, least)
    return dict(headline=head, detail=detail)


def vols(lp, days):
    # yearly volatility as the simulator works it out on each of these days: the 2 years of daily
    # closes up to and including that day
    lp = np.asarray(lp, dtype=float)
    return np.array([np.diff(lp[t - 504:t + 1]).std(ddof=1) for t in days]) * 252 ** 0.5


def calibrate(lp, bt, levels=(80, 90)):
    # the simulator draws a range for where the price could be after h days. did the real price land
    # inside it as often as the range says? this uses the simulator's own normal-curve option, which
    # has a closed form: expected return 0% a year, volatility from the last 2 years of daily closes
    h, o = bt['h'], bt['origins']
    vol = vols(lp, o)
    sd = np.maximum(vol * (h / 252) ** 0.5, 1e-12)
    mid = -0.5 * vol ** 2 * h / 252                 # the middle of the log price sits a little under today's
    away = np.abs(bt['actual'] - bt['last'] - mid) / sd

    # forecasts closer together than h days overlap, so they are not that many separate tests
    n = len(o)
    indep = max(1, int(round(n * min(bt['step'] / h, 1))))
    out = []
    for lvl in levels:
        want = lvl / 100
        got = float((away <= stats.norm.ppf(0.5 + want / 2)).mean())
        p = float(stats.binomtest(int(round(got * indep)), indep, want).pvalue)
        lo, hi = stats.binom.ppf([0.025, 0.975], indep, want) / indep
        out.append(dict(level=lvl, covered=100 * got, n=n, indep=indep, p=p,
                        lo=float(100 * lo), hi=float(100 * hi),
                        verdict='ok' if p >= alpha else ('narrow' if got < want else 'wide')))
    return out
