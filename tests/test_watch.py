import os
import subprocess
import sys
import threading

import pytest

import prices
import watch

buy = dict(allotment=100, iniprice=180, buycmm=5, sellcmm=5, cptlgain=15, target=None)      # break-even is 180.10


class Told:
    # what would have gone out by email or webhook
    def __init__(self, result=None):
        self.sent, self.result = [], result or {}

    def __call__(self, title, body):
        self.sent.append((title, body))
        return self.result


def new(sym='AAPL', price=175.0, **kw):
    return watch.add(sym, dict(buy, **kw), price)


def kinds(results):
    return [a['kind'] for r in results for a in r['alerts']]


# the list

def test_a_position_is_kept_with_its_break_even():
    pid = new()
    [p] = watch.positions()
    assert p['id'] == pid and p['symbol'] == 'AAPL' and p['allotment'] == 100 and p['iniprice'] == 180
    assert p['brkeven'] == pytest.approx(180.10) and p['target'] is None
    assert p['lastprice'] == 175 and p['beside'] == 'below'
    assert p['netprft'] == pytest.approx(-510)               # the same maths as the calculator, at 175


def test_the_list_is_in_symbol_order():
    for s in ('MSFT', 'AAPL', 'TSLA'):
        new(s)
    assert [p['symbol'] for p in watch.positions()] == ['AAPL', 'MSFT', 'TSLA']


def test_a_position_can_be_taken_off():
    a, b = new('AAPL'), new('MSFT')
    assert watch.remove(a) is True
    assert [p['id'] for p in watch.positions()] == [b]
    assert watch.remove(a) is False and watch.remove(9999) is False


def test_the_list_survives_between_runs():
    new()
    assert os.path.exists(watch.dbpath())
    assert len(watch.positions()) == 1                       # a new connection, same file


def test_where_the_file_is_kept(monkeypatch, tmp_path):
    assert watch.dbpath() == str(tmp_path / 'watchlist.db')
    monkeypatch.delenv('WATCHLIST_DB')
    assert watch.dbpath() == os.path.join(os.path.dirname(os.path.abspath(watch.__file__)), 'watchlist.db')


def test_there_is_a_limit_on_the_list():
    for i in range(watch.maxwatch):
        new('S%d' % i)
    with pytest.raises(ValueError, match='holds 50 positions at most'):
        new('ONEMORE')
    assert len(watch.positions()) == watch.maxwatch


def test_two_adds_at_once_cannot_both_take_the_last_place(monkeypatch):
    for i in range(watch.maxwatch - 1):
        new('S%d' % i)
    go, outcome = threading.Barrier(6), []

    def add(i):
        go.wait()
        try:
            new('T%d' % i)
            outcome.append('added')
        except ValueError:
            outcome.append('full')
    threads = [threading.Thread(target=add, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcome) == ['added'] + ['full'] * 5 and len(watch.positions()) == watch.maxwatch


def test_a_change_that_goes_wrong_halfway_leaves_nothing_behind():
    new('AAPL')
    with pytest.raises(RuntimeError):
        with watch.opened() as db:
            db.execute("update positions set symbol = 'CHANGED'")
            db.execute("insert into alerts (symbol, kind, price, title, body, made, sent) values ('X', 'up', 1, 't', 'b', 'm', 's')")
            raise RuntimeError('something broke')
    assert [p['symbol'] for p in watch.positions()] == ['AAPL'] and watch.recent() == []


# crossings

@pytest.mark.parametrize('start, prices_after, want', [
    (175, [175], [[]]),                                               # nothing moves, nothing is said
    (175, [185], [['up']]),
    (175, [185, 186, 190], [['up'], [], []]),                         # it stays there, so it is said once
    (175, [185, 170], [['up'], ['down']]),
    (175, [185, 170, 171, 172], [['up'], ['down'], [], []]),
    (175, [185, 170, 190], [['up'], ['down'], ['up']]),               # and again each time it crosses
    (190, [175], [['down']]),
    (190, [195, 200], [[], []]),
    (175, [180.10], [[]]),                                            # exactly on it is not above
    (190, [180.10], [['down']]),                                      # and leaving the top means down
    (175, [180.11], [['up']]),
])
def test_you_are_told_once_for_each_crossing_of_the_break_even(market, start, prices_after, want):
    new(price=start)
    told = Told()
    got = []
    for p in prices_after:
        market.price['AAPL'] = p
        got.append(kinds(watch.check(send=told)))
    assert got == want
    assert len(told.sent) == sum(len(w) for w in want)


@pytest.mark.parametrize('start, prices_after, want', [
    (190, [195], [[]]),
    (190, [200], [['target']]),                                       # reached it
    (190, [205, 210], [['target'], []]),                              # it stays up there
    (190, [205, 190, 200], [['target'], [], ['target']]),             # it fell back, so it counts again
    (205, [210], [[]]),                                               # already past it when it was added
    (205, [190, 202], [[], ['target']]),
])
def test_the_target_is_told_once_each_time_it_is_reached(market, start, prices_after, want):
    new(price=start, target=200)
    got = []
    for p in prices_after:
        market.price['AAPL'] = p
        got.append(kinds(watch.check(send=Told())))
    assert got == want


def test_a_price_on_the_break_even_is_not_above_it_even_when_the_maths_is_a_hair_out(market):
    # 3 shares at $12.70 with $0.30 of commission break even at $12.80, but 12.7 + 0.3 / 3 is 12.799999999999999
    assert 12.7 + 0.3 / 3 < 12.8
    new(price=12.70, allotment=3, iniprice=12.70, buycmm=0.15, sellcmm=0.15, cptlgain=0)
    market.price['AAPL'] = 12.80
    told = Told()
    assert kinds(watch.check(send=told)) == [] and watch.positions()[0]['beside'] == 'below'
    market.price['AAPL'] = 12.81
    assert kinds(watch.check(send=told)) == ['up']
    market.price['AAPL'] = 12.80
    assert kinds(watch.check(send=told)) == ['down']                 # and leaving the top means it is back to even


def test_a_jump_over_both_says_both_in_that_order(market):
    new(price=175, target=200)
    market.price['AAPL'] = 210
    assert kinds(watch.check(send=Told())) == ['up', 'target']


def test_where_it_is_when_added_is_not_news(market):
    new(price=175)
    market.price['AAPL'] = 175
    told = Told()
    watch.check(send=told)
    assert told.sent == [] and watch.recent() == []


def test_a_position_with_no_side_recorded_yet_is_set_silently(market):
    # not something the page can make, but a row with nothing recorded must not alert on its first check
    with watch.opened() as db:
        db.execute("insert into positions (symbol, allotment, iniprice, buycmm, sellcmm, cptlgain, target, added)"
                   " values ('AAPL', 100, 180, 5, 5, 15, 200, '2026-01-01 00:00')")
    told = Told()
    market.price['AAPL'] = 210
    assert kinds(watch.check(send=told)) == [] and told.sent == []
    p = watch.positions()[0]
    assert p['beside'] == 'above' and p['tgtside'] == 'hit'
    market.price['AAPL'] = 170
    assert kinds(watch.check(send=told)) == ['down']                 # and from then on it says so


def test_no_target_means_no_target_alert(market):
    new(price=190)
    market.price['AAPL'] = 500
    assert kinds(watch.check(send=Told())) == []


# what an alert says

def test_what_an_up_alert_says(market):
    new(price=175)
    market.price['AAPL'] = 185
    told = Told()
    watch.check(send=told)
    [(title, body)] = told.sent
    assert title == 'AAPL is above your break-even'
    assert body.split('\n') == [
        'AAPL is at $185.00, above your break-even sell price of $180.10.',
        'Your position: 100 shares bought at $180.00, with $5.00 commission to buy and $5.00 to sell.',
        'Selling at $185.00 would net $416.50 after commissions and 15% tax on a gain (2.3% return).',
        'Prices are the latest Yahoo Finance has and can be a day old. Not financial advice.']


def test_what_a_down_alert_says(market):
    new(price=190)
    market.price['AAPL'] = 170
    told = Told()
    watch.check(send=told)
    [(title, body)] = told.sent
    assert title == 'AAPL fell below your break-even'
    lines = body.split('\n')
    assert lines[0] == 'AAPL is at $170.00, below your break-even sell price of $180.10.'
    assert lines[2] == 'Selling at $170.00 would net -$1,010.00 after commissions and 15% tax on a gain (-5.6% return).'


def test_what_a_target_alert_says(market):
    new(price=190, target=200, allotment=12.5, cptlgain=0.5)
    market.price['AAPL'] = 201
    told = Told()
    watch.check(send=told)
    [(title, body)] = told.sent
    assert title == 'AAPL reached your target'
    lines = body.split('\n')
    assert lines[0] == 'AAPL is at $201.00, at or above your target of $200.00. Your break-even sell price is $180.80.'
    assert lines[1] == 'Your position: 12.5 shares bought at $180.00, with $5.00 commission to buy and $5.00 to sell.'
    assert '0.5% tax' in lines[2]


def test_alerts_are_kept_for_the_page_with_what_happened_to_them(market):
    new(price=175, target=200)
    market.price['AAPL'] = 210
    told = Told({'email': 'ok', 'webhook': 'HTTPError: HTTP Error 500: Server Error'})
    watch.check(send=told)
    got = watch.recent()
    assert [a['kind'] for a in got] == ['target', 'up']                                  # newest first
    assert all(a['symbol'] == 'AAPL' and a['price'] == 210 for a in got)
    assert got[0]['sent'] == 'page, email ok, webhook failed: HTTPError: HTTP Error 500: Server Error'
    assert got[0]['title'] == 'AAPL reached your target' and 'Not financial advice' in got[0]['body']


def test_with_nothing_set_up_an_alert_is_only_on_the_page(market):
    new(price=175)
    market.price['AAPL'] = 185
    watch.check()                                  # the real notify: no channel is set up in the tests
    assert watch.recent()[0]['sent'] == 'page'


def test_alerts_go_out_through_the_real_channels(market, chan):
    hook, smtp = chan.hook(), chan.smtp()
    new(price=175)
    market.price['AAPL'] = 185
    [r] = watch.check()
    assert r['alerts'][0]['sent'] == 'page, email ok, webhook ok' and r['alerts'][0]['failed'] == []
    assert hook.posts[0]['json']['text'].startswith('AAPL is above your break-even\nAAPL is at $185.00')
    assert smtp.mails[0]['msg']['Subject'] == 'AAPL is above your break-even'


def test_a_channel_that_fails_is_reported_and_the_alert_is_still_kept(market, chan):
    chan.hook(status=500)
    new(price=175)
    market.price['AAPL'] = 185
    [r] = watch.check()
    assert r['alerts'][0]['failed'] == ['webhook'] and 'webhook failed: HTTPError' in r['alerts'][0]['sent']
    assert len(watch.recent()) == 1


def test_the_oldest_alerts_are_forgotten(market, monkeypatch):
    monkeypatch.setattr(watch, 'keepalerts', 3)
    new(price=175)
    for i in range(6):
        market.price['AAPL'] = 185 if i % 2 == 0 else 170
        watch.check(send=Told())
    got = watch.recent(10)
    assert [a['kind'] for a in got] == ['down', 'up', 'down']                            # the last three of six


# two checks at once

def test_two_checks_at_the_same_moment_say_it_once(market, monkeypatch):
    new(price=175)
    market.price['AAPL'] = 185
    both_asked = threading.Barrier(2)
    real = market.history

    def history(sym, years=2):
        both_asked.wait(timeout=10)                 # each has read the list before either has acted on it
        return real(sym, years)
    monkeypatch.setattr(prices, 'history', history)
    told, found = Told(), []

    def go():
        found.append(kinds(watch.check(send=told)))
    threads = [threading.Thread(target=go) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(map(len, found)) == [0, 1] and len(told.sent) == 1 and len(watch.recent()) == 1


# when prices cannot be had

def test_a_failed_lookup_is_noted_and_changes_nothing_else(market):
    new('AAPL', price=175)
    new('MSFT', price=175)
    market.price['AAPL'] = 185
    market.down['MSFT'] = "Couldn't get prices for MSFT right now."
    told = Told()
    results = watch.check(send=told)
    by = {r['symbol']: r for r in results}
    assert by['MSFT']['error'] == "Couldn't get prices for MSFT right now." and by['MSFT']['alerts'] == []
    assert kinds(results) == ['up'] and len(told.sent) == 1                              # the other one carried on
    p = {x['symbol']: x for x in watch.positions()}
    assert p['MSFT']['lasterr'] == "Couldn't get prices for MSFT right now." and p['MSFT']['beside'] == 'below'
    assert p['MSFT']['lastprice'] == 175 and p['AAPL']['lasterr'] is None


def test_a_later_good_check_clears_the_error_and_still_catches_the_crossing(market):
    new(price=175)
    market.down['AAPL'] = 'No price history found for AAPL.'
    watch.check(send=Told())
    del market.down['AAPL']
    market.price['AAPL'] = 185
    assert kinds(watch.check(send=Told())) == ['up']
    assert watch.positions()[0]['lasterr'] is None


def test_several_lots_of_one_symbol_cost_one_lookup(market):
    new(price=175)
    new(price=175, iniprice=170, buycmm=0, sellcmm=0)                 # break-even $170
    market.price['AAPL'] = 185
    told = Told()
    watch.check(send=told)
    assert market.asked == [('AAPL', 1)]
    assert [t for t, b in told.sent] == ['AAPL is above your break-even']                # only the first lot crossed


def test_each_lot_has_its_own_break_even(market):
    new(price=175)
    new(price=175, iniprice=170, buycmm=0, sellcmm=0)
    market.price['AAPL'] = 175
    assert kinds(watch.check(send=Told())) == []                      # the second was already above 170 when added
    market.price['AAPL'] = 165
    assert kinds(watch.check(send=Told())) == ['down']                # and only that one crossed


def test_an_empty_list_asks_for_nothing(market):
    assert watch.check(send=Told()) == [] and market.asked == []


def test_a_position_removed_while_checking_is_not_resurrected(market, monkeypatch):
    pid = new(price=175)
    market.price['AAPL'] = 185
    real = market.history

    def history(sym, years=2):
        watch.remove(pid)                           # taken off the list between the look-up and the update
        return real(sym, years)
    monkeypatch.setattr(prices, 'history', history)
    told = Told()
    assert kinds(watch.check(send=told)) == []
    assert watch.positions() == [] and told.sent == []


# the command line

def test_it_says_nothing_when_nothing_happens(market, capsys):
    new(price=175)
    market.price['AAPL'] = 176
    assert watch.run([]) == 0
    assert capsys.readouterr() == ('', '')


def test_it_says_what_crossed(market, capsys):
    new(price=175)
    market.price['AAPL'] = 185
    assert watch.run([]) == 0
    out, err = capsys.readouterr()
    assert out == 'ALERT AAPL: AAPL is above your break-even (page)\n' and err == ''


def test_verbose_says_where_each_one_is(market, capsys):
    new('AAPL', price=175)
    new('MSFT', price=175)
    market.price.update(AAPL=185, MSFT=170)
    assert watch.run(['-v']) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ['AAPL: $185.00, above your break-even', 'ALERT AAPL: AAPL is above your break-even (page)',
                   'MSFT: $170.00, not above your break-even', 'checked 2 positions']


def test_a_price_that_cannot_be_had_is_an_error_on_stderr_and_exit_1(market, capsys):
    new(price=175)
    market.down['AAPL'] = "Couldn't get prices for AAPL right now."
    assert watch.run([]) == 1
    out, err = capsys.readouterr()
    assert out == '' and err == "AAPL: could not check it (Couldn't get prices for AAPL right now.)\n"


def test_an_alert_that_could_not_be_sent_is_exit_1_but_still_printed(market, chan, capsys):
    chan.hook(status=500)
    new(price=175)
    market.price['AAPL'] = 185
    assert watch.run([]) == 1
    out = capsys.readouterr().out
    assert out.startswith('ALERT AAPL: AAPL is above your break-even (page, webhook failed: HTTPError')


def test_the_script_reads_the_settings_file_before_it_checks(monkeypatch):
    import dotenv
    order = []
    monkeypatch.setattr(dotenv, 'load_dotenv', lambda *a, **k: order.append('settings'))
    monkeypatch.setattr(watch, 'check', lambda *a, **k: order.append('check') or [])
    monkeypatch.setattr(sys, 'argv', ['watch.py'])
    assert watch.main() == 0 and order == ['settings', 'check']


def test_the_script_runs_on_its_own(tmp_path):
    # the way cron runs it: a fresh python, the settings from the environment, a real exit status
    new(price=175)
    code = ("import numpy as np, prices, sys, watch\n"
            "prices.history = lambda sym, years=2: np.array([100.0, 185.0])\n"
            "sys.exit(watch.main())")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, WATCHLIST_DB=watch.dbpath(), PYTHONPATH=here, PYTHONDONTWRITEBYTECODE='1',
               ALERT_EMAIL_TO='', SMTP_HOST='', ALERT_WEBHOOK_URL='')
    r = subprocess.run([sys.executable, '-c', code, '-v'], cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == ['AAPL: $185.00, above your break-even', 'ALERT AAPL: AAPL is above your break-even (page)',
                                     'checked 1 positions']
    assert watch.recent()[0]['kind'] == 'up'                    # in the same file the tests use
