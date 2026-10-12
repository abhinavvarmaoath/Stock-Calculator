import warnings

import numpy as np

import verdict

# Walk-forward test of price forecasts. Everything works on log prices, so an error of 0.05
# is about 5% of the price. A model is only ever handed the prices up to the day it
# forecasts from, never anything after it.

warm = 504          # two years of prices before the first test forecast
target = 70         # about this many test forecasts, spread over the history
first = 199         # days of history the features need

names = ['naive', 'drift', 'holt', 'arima', 'ridge', 'gbm']
labels = dict(
    naive='Naive (price stays put)',
    drift='Drift (average past growth)',
    holt='Damped trend (Holt)',
    arima='ARIMA(1,1,1)',
    ridge='Momentum regression (ridge)',
    gbm='Gradient boosting',
)
horizons = {5: '1 week', 21: '1 month', 63: '3 months'}


class TooShort(Exception):
    pass


def minimum(h):
    # fewest closing prices that still gives 30 test forecasts
    return warm + h + 30 * max(h // 3, 1)


# the models. each gets the log prices up to today and says where the log price will be h days on

def naive(lp, h):
    return lp[-1]


def drift(lp, h):
    # keep growing at the average pace of the last year
    return lp[-1] + (lp[-1] - lp[-253]) / 252 * h


def holt(lp, h):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = ExponentialSmoothing(lp[-504:], trend='add', damped_trend=True,
                                   initialization_method='estimated').fit()
        return float(fit.forecast(h)[-1])


def arima(lp, h):
    from statsmodels.tsa.arima.model import ARIMA
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = ARIMA(lp[-504:], order=(1, 1, 1), trend='t').fit()
        return float(fit.forecast(h)[-1])


def features(lp, s):
    # what can be known on day s: only prices up to and including day s
    d = np.diff(lp[s - 63:s + 1])           # the last 63 daily moves
    return [lp[s] - lp[s - 5], lp[s] - lp[s - 21], lp[s] - lp[s - 63], lp[s] - lp[s - 126],
            d[-21:].std(), d.std(), lp[s] - lp[s - 199:s + 1].mean()]


class Learner:
    # ridge regression or gradient boosting on those features. it learns what happened over
    # the next h days after each past day, using only days whose next h days are already over.
    # use one per test run, and give it later and later days
    def __init__(self, kind):
        self.kind = kind
        self.rows, self.model, self.calls = [], None, 0

    def __call__(self, lp, h):
        t = len(lp) - 1
        if len(self.rows) > t - first + 1:
            raise ValueError('days have to go forward')
        for s in range(first + len(self.rows), t + 1):      # features for the days not seen yet
            self.rows.append(features(lp, s))
        X = np.array(self.rows)
        k = t - h - first + 1                               # days whose next h days are over
        if k < 100:
            raise ValueError('not enough history to learn from')
        y = lp[first + h:t + 1] - lp[first:t - h + 1]
        x = X[-1]

        if self.kind == 'ridge':
            mu, sd = X[:k].mean(0), X[:k].std(0) + 1e-12
            Z = (X[:k] - mu) / sd
            ym = y.mean()
            # a penalty of k pulls each weight about halfway to zero. the h day targets overlap,
            # so there is less real information here than there are rows
            w = np.linalg.solve(Z.T @ Z + k * np.eye(Z.shape[1]), Z.T @ (y - ym))
            move = ym + ((x - mu) / sd) @ w
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            if self.model is None or self.calls % 3 == 0:   # refit every third forecast
                # small and slow on purpose: the targets are mostly noise, so big trees only fit it
                self.model = HistGradientBoostingRegressor(
                    max_depth=2, learning_rate=0.02, max_iter=60, min_samples_leaf=100,
                    l2_regularization=10.0, random_state=0).fit(X[:k], y)
            move = self.model.predict(x[None])[0]
        self.calls += 1
        return lp[-1] + float(move)


def make(name):
    if name in ('ridge', 'gbm'):
        return Learner(name)
    return dict(naive=naive, drift=drift, holt=holt, arima=arima)[name]


# the test

def backtest(lp, h, which=None):
    # forecast h days ahead from many past days and keep what actually happened
    which = ['naive'] + [w for w in (which or names) if w != 'naive']     # everything is scored against naive
    lp = np.asarray(lp, dtype=float)
    if lp.ndim != 1 or not np.isfinite(lp).all():
        raise ValueError('the prices have to be a list of numbers')
    n = len(lp)
    if n < minimum(h):
        raise TooShort('need at least {:,} prices for a {} day forecast, got {:,}'.format(minimum(h), h, n))
    span = n - warm - h
    step = max(span // target, h // 3, 1)
    origins = np.arange(warm, n - h, step)

    preds, failed = {}, {}
    for name in which:
        f = make(name)
        out = np.full(len(origins), np.nan)
        for i, t in enumerate(origins):
            try:
                v = float(f(lp[:t + 1], h))         # a slice: the model cannot see past day t
            except Exception:
                continue
            if np.isfinite(v):
                out[i] = v
        preds[name] = out
        failed[name] = int(np.isnan(out).sum())
    return dict(h=h, step=int(step), origins=origins, last=lp[origins], actual=lp[origins + h],
                preds=preds, failed=failed)


def score(bt):
    # one row per model: how far off it was, and how that compares with the naive forecast
    actual, last = bt['actual'], bt['last']
    base = np.abs(bt['preds']['naive'] - actual)
    rows = []
    for name, p in bt['preds'].items():
        ok = np.isfinite(p)
        if not ok.any():
            rows.append(dict(name=name, label=labels.get(name, name), n=0, failed=bt['failed'][name],
                             miss=None, skill=None, closer=None, direction=None))
            continue
        err, b = np.abs(p - actual)[ok], base[ok]
        row = dict(name=name, label=labels.get(name, name), n=int(ok.sum()), failed=bt['failed'][name],
                   miss=float(100 * err.mean()), skill=None, closer=None, direction=None)
        if name == 'naive':
            row['skill'] = 0.0
        else:
            if b.mean() > 0:
                row['skill'] = float(100 * (1 - err.mean() / b.mean()))
            row['closer'] = float(100 * (err < b).mean())
            up, moved = np.sign(p[ok] - last[ok]), np.sign(actual[ok] - last[ok])
            has = up != 0
            if has.any():
                row['direction'] = float(100 * (up[has] == moved[has]).mean())
        rows.append(row)
    return rows


def scoreboard(lp, h, which=None):
    # everything the page and the assistant show for one horizon. plain numbers and text only
    lp = np.asarray(lp, dtype=float)
    bt = backtest(lp, h, which)
    rows = score(bt)
    said = verdict.judge(bt, rows)
    return dict(h=h, horizon=horizons.get(h, '{} days'.format(h)), prices=int(len(lp)),
                forecasts=int(len(bt['origins'])), every=bt['step'], rows=rows,
                calibration=verdict.calibrate(lp, bt), **said)
