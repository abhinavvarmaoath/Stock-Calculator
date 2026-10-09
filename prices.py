import time

import numpy as np

cache = {}          # symbol -> (time fetched, closes)
ttl = 15 * 60       # keep prices for 15 minutes so we don't keep hitting yahoo

tryagain = ' Try again, or enter the current price and volatility under Advanced.'


class PriceError(Exception):
    def __init__(self, what):
        # what went wrong, plus a hint for people using the web form
        super().__init__(what + tryagain)
        self.what = what


def history(sym, years=2):
    # daily closes (adjusted for splits and dividends) as a numpy array
    hit = cache.get(sym)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]

    try:
        import yfinance as yf
        df = yf.Ticker(sym).history(period='{}y'.format(years), auto_adjust=True, timeout=10, raise_errors=True)
    except Exception as e:
        # raise_errors because otherwise yfinance sometimes hides a failed fetch as an empty
        # result, which would read as "no such symbol". yahoo not knowing the symbol is
        # different from us not being able to ask
        if type(e).__name__ in ('YFTickerMissingError', 'YFPricesMissingError'):
            raise PriceError('No price history found for {}. Check the symbol.'.format(sym))
        raise PriceError("Couldn't get prices for {} right now.".format(sym))

    if df is None or df.empty or 'Close' not in df:
        raise PriceError('No price history found for {}. Check the symbol.'.format(sym))

    closes = df['Close'].dropna().to_numpy(dtype=float)
    closes = closes[closes > 0]
    if len(closes) < 60:
        raise PriceError('Not enough price history for {} to work out volatility.'.format(sym))

    cache[sym] = (time.time(), closes)
    return closes


def stats(closes):
    # last price, yearly volatility and the daily log returns
    r = np.diff(np.log(closes))
    return float(closes[-1]), float(r.std(ddof=1) * np.sqrt(252)), r


def resolve(sym, curprice, vol, resample):
    # use what was typed in and only go to the price history for what is missing.
    # gives back the price, the volatility, the past daily returns (only when
    # resampling) and where the price and volatility came from
    src = dict(price='entered by you', vol='entered by you')
    hist = None
    if curprice is None or vol is None or resample:
        last, histvol, r = stats(history(sym))
        if curprice is None:
            curprice, src['price'] = last, 'latest close'
        if vol is None:
            vol, src['vol'] = histvol, 'worked out from the last 2 years of daily closes'
        if resample:
            hist = r
    return curprice, vol, hist, src
