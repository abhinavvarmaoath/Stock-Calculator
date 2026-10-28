import os
import sqlite3
import sys
from contextlib import contextmanager
from datetime import datetime, timezone

import notify
import prices
from calc import calc

# The watchlist: positions you hold, checked against the latest price. You are told when a price
# crosses your break-even (the sell price that covers both commissions), in either direction, and
# when it reaches a target you set. A crossing is a change of side since the last check, so
# you get one alert for it and not one every day it stays there.
#
# Run `python watch.py` once a day after the market closes (see the README), or press Check now on
# the page. Alerts are always kept on the page. Email and a webhook are extra (see notify.py).

maxwatch = 50           # positions on the list
keepalerts = 200        # alerts kept

schema = """
create table if not exists positions (
    id integer primary key,
    symbol text not null,
    allotment real not null,
    iniprice real not null,
    buycmm real not null,
    sellcmm real not null,
    cptlgain real not null,
    target real,
    added text not null,
    beside text,            -- above or below the break-even at the last check
    tgtside text,           -- hit or miss against the target at the last check
    lastprice real,
    lastcheck text,
    lasterr text            -- why the last check could not be done, if it could not
);
create table if not exists alerts (
    id integer primary key,
    symbol text not null,
    kind text not null,     -- up, down or target
    price real not null,
    title text not null,
    body text not null,
    made text not null,
    sent text not null      -- what happened to it: page, email ok, webhook failed...
);
"""


def dbpath():
    return os.environ.get('WATCHLIST_DB') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'watchlist.db')


@contextmanager
def opened():
    # one transaction. the first thing it does is take the write lock, so a check and an add (or two
    # checks) cannot both read the list and then both act on what they read
    db = sqlite3.connect(dbpath(), timeout=10, isolation_level=None)
    db.row_factory = sqlite3.Row
    try:
        db.executescript(schema)
        db.execute('begin immediate')
        yield db
        db.execute('commit')
    finally:
        db.close()              # an error before the commit leaves nothing changed


def stamp():
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')


def num(x):
    # 100 -> 100, 12.5 -> 12.5, 1000000 -> 1,000,000
    return '{:,.4f}'.format(x).rstrip('0').rstrip('.')


def breakeven(r):
    return calc(r['allotment'], r['iniprice'], r['iniprice'], r['buycmm'], r['sellcmm'], r['cptlgain'])['brkeven']


def sides(price, be, target):
    # above or below the break-even, and whether the target has been reached (None if there is no target).
    # exactly on the break-even is not above, since nothing is made there. it is worked out in floating
    # point (12.7 + 0.3 / 3 comes out a hair under 12.8), so exactly has a hair of room
    return ('above' if price > be + 1e-9 else 'below'), (None if target is None else ('hit' if price >= target else 'miss'))


def look(r):
    # a position as the page shows it: the row, and what the calculator says about it
    out = dict(r, brkeven=breakeven(r))
    if r['lastprice'] is not None:
        c = calc(r['allotment'], r['iniprice'], r['lastprice'], r['buycmm'], r['sellcmm'], r['cptlgain'])
        out.update(netprft=c['netprft'], roi=c['roi'])
    return out


def add(sym, v, price):
    # v has allotment, iniprice, buycmm, sellcmm, cptlgain and target (None for no target). price is
    # the latest price. that is where it starts: you are told about crossings after this, not where
    # it is now
    side, tgt = sides(price, breakeven(v), v['target'])
    with opened() as db:
        if db.execute('select count(*) from positions').fetchone()[0] >= maxwatch:
            raise ValueError('The watchlist is full. It holds {} positions at most.'.format(maxwatch))
        cur = db.execute(
            'insert into positions (symbol, allotment, iniprice, buycmm, sellcmm, cptlgain, target, added,'
            ' beside, tgtside, lastprice, lastcheck) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (sym, v['allotment'], v['iniprice'], v['buycmm'], v['sellcmm'], v['cptlgain'], v['target'],
             stamp(), side, tgt, price, stamp()))
        return cur.lastrowid


def remove(pid):
    with opened() as db:
        return db.execute('delete from positions where id = ?', (pid,)).rowcount == 1


def positions():
    with opened() as db:
        return [look(dict(r)) for r in db.execute('select * from positions order by symbol, id')]


def recent(n=20):
    with opened() as db:
        return [dict(r) for r in db.execute('select * from alerts order by id desc limit ?', (n,))]


def message(kind, r, price):
    # what an alert says. the numbers are the calculator's, for this position at this price
    c = calc(r['allotment'], r['iniprice'], price, r['buycmm'], r['sellcmm'], r['cptlgain'])
    sym, money = r['symbol'], notify.money
    if kind == 'up':
        title = '{} is above your break-even'.format(sym)
        first = '{} is at {}, above your break-even sell price of {}.'.format(sym, money(price), money(c['brkeven']))
    elif kind == 'down':
        title = '{} fell below your break-even'.format(sym)
        first = '{} is at {}, below your break-even sell price of {}.'.format(sym, money(price), money(c['brkeven']))
    else:
        title = '{} reached your target'.format(sym)
        first = '{} is at {}, at or above your target of {}. Your break-even sell price is {}.'.format(
            sym, money(price), money(r['target']), money(c['brkeven']))
    return title, '\n'.join([
        first,
        'Your position: {} shares bought at {}, with {} commission to buy and {} to sell.'.format(
            num(r['allotment']), money(r['iniprice']), money(r['buycmm']), money(r['sellcmm'])),
        'Selling at {} would net {} after commissions and {}% tax on a gain ({:.1f}% return).'.format(
            money(price), money(c['netprft']), num(r['cptlgain']), c['roi']),
        'Prices are the latest Yahoo Finance has and can be a day old. Not financial advice.'])


def how(out):
    # what happened to an alert, in words
    return ', '.join(['page'] + [k + (' ok' if v == 'ok' else ' failed: ' + v) for k, v in out.items()])


def check(send=None):
    # asks for the latest price of everything on the list and tells you about each crossing.
    # gives back one dict per position: its id and symbol, the price or the error, and its alerts
    send = send or notify.send
    with opened() as db:
        rows = [dict(r) for r in db.execute('select * from positions order by symbol, id')]

    latest = {}
    for sym in sorted({r['symbol'] for r in rows}):         # one lookup for each symbol, however many lots
        try:
            latest[sym] = float(prices.history(sym, 1)[-1])
        except prices.PriceError as e:
            latest[sym] = e.what

    results = []
    for r in rows:
        got = latest[r['symbol']]
        if isinstance(got, str):
            with opened() as db:
                db.execute('update positions set lastcheck = ?, lasterr = ? where id = ?', (stamp(), got, r['id']))
            results.append(dict(id=r['id'], symbol=r['symbol'], error=got, alerts=[]))
            continue

        side, tgt = sides(got, breakeven(r), r['target'])
        made = []
        with opened() as db:
            db.execute('update positions set lastprice = ?, lastcheck = ?, lasterr = null where id = ?',
                       (got, stamp(), r['id']))
            # moving to the new side is claimed here. a second check that read the list at the same
            # time finds it already moved, so only one of them gets to say so
            kinds = []
            if db.execute('update positions set beside = ? where id = ? and beside is ?',
                          (side, r['id'], r['beside'])).rowcount and r['beside'] and r['beside'] != side:
                kinds.append('up' if side == 'above' else 'down')
            if tgt and db.execute('update positions set tgtside = ? where id = ? and tgtside is ?',
                                  (tgt, r['id'], r['tgtside'])).rowcount and r['tgtside'] == 'miss' and tgt == 'hit':
                kinds.append('target')
            for kind in kinds:
                title, body = message(kind, r, got)
                aid = db.execute('insert into alerts (symbol, kind, price, title, body, made, sent) values (?, ?, ?, ?, ?, ?, ?)',
                                 (r['symbol'], kind, got, title, body, stamp(), 'not sent yet')).lastrowid
                made.append((aid, kind, title, body))
            if made:
                db.execute('delete from alerts where id not in (select id from alerts order by id desc limit ?)', (keepalerts,))

        # sent after the transaction, because a slow mail server must not hold the database
        told = []
        for aid, kind, title, body in made:
            out = send(title, body)
            with opened() as db:
                db.execute('update alerts set sent = ? where id = ?', (how(out), aid))
            told.append(dict(id=aid, kind=kind, title=title, sent=how(out), failed=[k for k, v in out.items() if v != 'ok']))
        results.append(dict(id=r['id'], symbol=r['symbol'], price=got, side=side, alerts=told))
    return results


def run(argv):
    # python watch.py [-v]: quiet unless something happened, so a cron job only speaks when it has to.
    # exit status 1 when a price could not be looked up or an alert could not be sent
    verbose = '-v' in argv or '--verbose' in argv
    bad = 0
    results = check()
    for r in results:
        if r.get('error'):
            print('{}: could not check it ({})'.format(r['symbol'], r['error']), file=sys.stderr)
            bad += 1
        elif verbose:
            print('{}: {}, {} your break-even'.format(r['symbol'], notify.money(r['price']), 'above' if r['side'] == 'above' else 'not above'))
        for a in r['alerts']:
            print('ALERT {}: {} ({})'.format(r['symbol'], a['title'], a['sent']))
            bad += len(a['failed'])
    if verbose:
        print('checked {} positions'.format(len(results)))
    return 1 if bad else 0


def main():
    from dotenv import load_dotenv
    load_dotenv()               # where the email and webhook settings live
    return run(sys.argv[1:])


if __name__ == '__main__':
    sys.exit(main())
