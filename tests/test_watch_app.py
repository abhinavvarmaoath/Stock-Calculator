import html
import re

import pytest

import prices
import watch
from app import app

form = dict(stksymbol='aapl', allotment='100', iniprice='180', buycmm='5', sellcmm='5', cptlgain='15', target='')
buy = dict(allotment=100, iniprice=180, buycmm=5, sellcmm=5, cptlgain=15, target=None)       # break-even is $180.10


@pytest.fixture
def client():
    app.testing = True
    return app.test_client()


@pytest.fixture(autouse=True)
def fresh_limits():
    from app import adds, checks
    adds.hits.clear()
    checks.hits.clear()


def text(r):
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', r.get_data(as_text=True))))


def add(client, **kw):
    return client.post('/watch/add', data=dict(form, **kw))


def watching(price=175.0, sym='AAPL', **kw):
    return watch.add(sym, dict(buy, **kw), price)


# the page

def test_an_empty_watchlist_says_so_and_asks_for_nothing(client, market):
    r = client.get('/watch')
    t = text(r)
    assert r.status_code == 200 and 'Nothing on the list yet' in t and 'None yet.' in t
    for name in ('stksymbol', 'allotment', 'iniprice', 'buycmm', 'sellcmm', 'cptlgain', 'target'):
        assert 'name="{}"'.format(name) in r.get_data(as_text=True)
    assert market.asked == []                                         # looking at the page asks Yahoo for nothing


def test_the_menu_has_it_on_every_page(client):
    for url in ('/', '/sim', '/forecast', '/agent', '/watch'):
        assert 'href="/watch"' in client.get(url).get_data(as_text=True)


def test_the_form_is_filled_in_from_the_link(client):
    body = client.get('/watch?stksymbol=MSFT&allotment=12.5&iniprice=410&buycmm=9.99&sellcmm=9.99&cptlgain=15').get_data(as_text=True)
    assert 'value="MSFT"' in body and 'name="allotment" id="allotment" value="12.5"' in body
    assert 'name="iniprice" id="iniprice" value="410"' in body and 'name="cptlgain" id="cptlgain" value="15"' in body


def test_the_calculator_report_links_here_with_the_trade_filled_in(client):
    r = client.post('/post', data=dict(stksymbol='abc', allotment='100', iniprice='10', fnlprice='15', buycmm='5', sellcmm='5', cptlgain='15'))
    link = re.search(r'href="(/watch\?[^"]+)"[^>]*>Watch this position', r.get_data(as_text=True)).group(1).replace('&amp;', '&')
    assert 'stksymbol=ABC' in link and 'allotment=100' in link and 'iniprice=10' in link and 'cptlgain=15' in link
    page = client.get(link).get_data(as_text=True)
    assert 'value="ABC"' in page and 'name="allotment" id="allotment" value="100"' in page
    assert 'fnlprice' not in link                                     # the sell price is not part of a position


def test_a_notice_about_what_was_added_only_for_a_real_symbol(client):
    assert 'Added AAPL to the watchlist.' in text(client.get('/watch?added=AAPL'))
    t = text(client.get('/watch?added=<script>alert(1)</script>'))
    assert 'Added' not in t and 'alert(1)' not in t


# adding

def test_a_position_is_added_and_shown(client, market):
    market.price['AAPL'] = 175
    r = add(client)
    assert r.status_code == 302 and r.headers['Location'].endswith('/watch?added=AAPL')
    assert market.asked == [('AAPL', 1)]
    t = text(client.get(r.headers['Location']))
    assert 'Added AAPL to the watchlist.' in t
    assert 'AAPL 100 shares at $180.00' in t
    assert '$180.10' in t and '$175.00' in t and '-$510.00' in t                    # break-even, latest price, net if sold now
    assert 'Not above break-even' in t and 'UTC' in t and '1 of 50 places used.' in t
    [p] = watch.positions()
    assert (p['symbol'], p['allotment'], p['iniprice'], p['buycmm'], p['sellcmm'], p['cptlgain'], p['target']) == ('AAPL', 100, 180, 5, 5, 15, None)
    assert p['beside'] == 'below' and p['lastprice'] == 175


def test_a_target_is_kept_and_shown(client, market):
    market.price['AAPL'] = 175
    add(client, target='200')
    assert watch.positions()[0]['target'] == 200
    assert '$200.00' in text(client.get('/watch'))


def test_a_position_added_above_its_break_even_starts_there_without_an_alert(client, market):
    market.price['AAPL'] = 190
    add(client)
    t = text(client.get('/watch'))
    assert 'Above break-even' in t and 'None yet.' in t and watch.recent() == []


def test_a_position_added_past_its_target_shows_it_as_reached(client, market):
    market.price['AAPL'] = 210
    add(client, target='200')
    assert 'reached' in text(client.get('/watch'))


@pytest.mark.parametrize('change, message', [
    (dict(stksymbol=''), 'Stock symbol must be 1 to 10 letters or numbers'),
    (dict(stksymbol='NOT A SYMBOL'), 'Stock symbol must be 1 to 10 letters or numbers'),
    (dict(allotment='0'), 'Number of shares must be more than 0'),
    (dict(allotment='abc'), 'Number of shares must be a number'),
    (dict(iniprice='-1'), 'Initial share price must be more than 0'),
    (dict(buycmm='-1'), 'Buy commission must be at least 0'),
    (dict(sellcmm=''), 'Sell commission is required'),
    (dict(cptlgain='101'), 'Capital gain tax rate must be 100 or less'),
    (dict(target='0'), 'Target sell price must be more than 0'),
    (dict(target='abc'), 'Target sell price must be a number'),
    (dict(target='1e99'), 'Target sell price must be 10,000,000 or less'),
])
def test_bad_input_is_sent_back_with_a_message_and_nothing_is_looked_up_or_kept(client, market, change, message):
    market.price['AAPL'] = 175
    r = add(client, **change)
    assert r.status_code == 400 and message in text(r)
    assert market.asked == [] and watch.positions() == []


def test_what_was_typed_comes_back_so_it_can_be_fixed(client, market):
    r = add(client, allotment='0', iniprice='123.45', target='250')
    body = r.get_data(as_text=True)
    assert 'value="aapl"' in body and 'name="iniprice" id="iniprice" value="123.45"' in body and 'value="250"' in body


def test_a_price_that_cannot_be_looked_up_means_it_is_not_added(client, market):
    market.down['AAPL'] = "Couldn't get prices for AAPL right now."
    r = add(client)
    t = text(r)
    assert r.status_code == 400 and "Couldn't get prices for AAPL right now." in t
    assert 'can only be added when its price can be looked up' in t and 'volatility' not in t
    assert watch.positions() == [] and 'value="aapl"' in r.get_data(as_text=True)        # as it was typed


def test_a_full_list_says_so(client, market):
    market.price['AAPL'] = 175
    for i in range(watch.maxwatch):
        watching(sym='S%d' % i)
    r = add(client)
    assert r.status_code == 400 and 'The watchlist is full. It holds 50 positions at most.' in text(r)
    assert len(watch.positions()) == watch.maxwatch


def test_adds_are_limited_but_bad_input_does_not_count(client, market, monkeypatch):
    from app import adds
    monkeypatch.setattr(adds, 'limit', 2)
    market.price['AAPL'] = 175
    for _ in range(5):
        assert add(client, allotment='0').status_code == 400
    assert add(client).status_code == 302 and add(client).status_code == 302
    r = add(client)
    assert r.status_code == 429 and 'Try again in' in text(r) and len(watch.positions()) == 2


# taking one off

def test_a_position_is_taken_off_with_a_post(client, market):
    a, b = watching(sym='AAPL'), watching(sym='MSFT')
    r = client.post('/watch/remove/{}'.format(a))
    assert r.status_code == 302 and r.headers['Location'].endswith('/watch')
    assert [p['id'] for p in watch.positions()] == [b]


def test_taking_off_one_that_is_not_there_is_harmless(client):
    r = client.post('/watch/remove/999')
    assert r.status_code == 302


def test_a_link_cannot_take_one_off(client):
    pid = watching()
    assert client.get('/watch/remove/{}'.format(pid)).status_code == 405
    assert len(watch.positions()) == 1


def test_the_page_has_a_button_to_take_each_one_off(client):
    a, b = watching(sym='AAPL'), watching(sym='MSFT')
    body = client.get('/watch').get_data(as_text=True)
    assert 'action="/watch/remove/{}"'.format(a) in body and 'action="/watch/remove/{}"'.format(b) in body
    assert 'aria-label="Remove AAPL from the watchlist"' in body


# check now

def test_check_now_says_what_crossed_and_keeps_the_alert(client, market):
    watching(price=175)
    market.price['AAPL'] = 185
    r = client.post('/watch/check')
    t = text(r)
    assert r.status_code == 200 and 'Checked 1 position: 1 alert.' in t
    assert 'AAPL is above your break-even' in t and 'AAPL is at $185.00, above your break-even sell price of $180.10.' in t
    assert 'Sent: page' in t and 'Above break-even' in t and '$185.00' in t and '$416.50' in t
    assert 'None yet.' not in t


def test_check_now_with_nothing_to_say(client, market):
    watching(price=175)
    market.price['AAPL'] = 176
    t = text(client.post('/watch/check'))
    assert 'Checked 1 position: 0 alerts.' in t and 'None yet.' in t


def test_check_now_says_which_could_not_be_checked(client, market):
    watching(sym='AAPL')
    watching(sym='MSFT')
    market.price['AAPL'] = 185
    market.down['MSFT'] = "Couldn't get prices for MSFT right now."
    t = text(client.post('/watch/check'))
    assert 'Checked 2 positions: 1 alert, 1 could not be checked.' in t
    assert "Couldn't get prices for MSFT right now." in t


def test_check_now_sends_alerts_out_through_the_channels_that_are_set_up(client, market, chan):
    hook = chan.hook()
    watching(price=175)
    market.price['AAPL'] = 185
    t = text(client.post('/watch/check'))
    assert 'Sent: page, webhook ok' in t and 'By webhook, for Slack, Discord, Mattermost and similar: set up.' in t
    assert hook.posts[0]['json']['text'].startswith('AAPL is above your break-even')


def test_check_now_shows_when_a_channel_failed(client, market, chan):
    chan.hook(status=500)
    watching(price=175)
    market.price['AAPL'] = 185
    assert 'webhook failed: HTTPError: HTTP Error 500' in text(client.post('/watch/check'))


def test_the_page_says_which_ways_of_being_told_are_set_up(client, chan):
    t = text(client.get('/watch'))
    assert 'By email: not set up.' in t and 'By webhook, for Slack, Discord, Mattermost and similar: not set up.' in t
    chan.smtp()
    chan.hook()
    t = text(client.get('/watch'))
    assert 'By email: set up.' in t and 'similar: set up.' in t


def test_checks_are_limited(client, market, monkeypatch):
    from app import checks
    monkeypatch.setattr(checks, 'limit', 1)
    watching()
    market.price['AAPL'] = 176
    assert client.post('/watch/check').status_code == 200
    r = client.post('/watch/check')
    assert r.status_code == 429 and 'That is a lot of checks' in text(r) and 'Try again in' in text(r)


def test_the_error_shown_for_a_position_is_taken_from_the_last_check_only(client, market):
    watching()
    market.down['AAPL'] = 'No price history found for AAPL.'
    client.post('/watch/check')
    assert 'No price history found for AAPL.' in text(client.get('/watch'))
    del market.down['AAPL']
    market.price['AAPL'] = 176
    client.post('/watch/check')
    assert 'No price history found' not in text(client.get('/watch'))


# other sites

CHANGES = [('/watch/add', dict(form)), ('/watch/remove/1', {}), ('/watch/check', {})]


@pytest.mark.parametrize('url, data', CHANGES)
@pytest.mark.parametrize('headers', [
    {'Sec-Fetch-Site': 'cross-site'},
    {'Sec-Fetch-Site': 'same-site'},                       # another subdomain is another site for this
    {'Origin': 'https://evil.example'},
    {'Origin': 'http://localhost.evil.example'},
    {'Origin': 'null'},
])
def test_a_form_from_another_site_changes_nothing(client, market, url, data, headers):
    market.price['AAPL'] = 175
    watching()
    r = client.post(url, data=data, headers=headers)
    assert r.status_code == 403 and 'another site' in r.get_data(as_text=True)
    assert market.asked == [] and len(watch.positions()) == 1 and watch.recent() == []


@pytest.mark.parametrize('headers', [
    {}, {'Sec-Fetch-Site': 'same-origin'}, {'Sec-Fetch-Site': 'none'}, {'Origin': 'http://localhost'},
    {'Sec-Fetch-Site': 'same-origin', 'Origin': 'https://anything.example'},      # the browser's own word wins
])
def test_forms_from_this_site_work(client, market, headers):
    market.price['AAPL'] = 175
    assert client.post('/watch/add', data=form, headers=headers).status_code == 302


# what is shown is never taken as html

def test_nothing_stored_is_shown_as_html(client):
    with watch.opened() as db:
        db.execute("insert into alerts (symbol, kind, price, title, body, made, sent) values (?, 'up', 1, ?, ?, '2026-01-01 00:00', ?)",
                   ('AAPL', '<script>alert("title")</script>', '<img src=x onerror=alert(1)>\nsecond', 'webhook failed: <b>bold</b>'))
        db.execute("insert into positions (symbol, allotment, iniprice, buycmm, sellcmm, cptlgain, added, lasterr)"
                   " values ('AAPL', 1, 1, 0, 0, 0, '2026-01-01', '<script>alert(\"err\")</script>')")
    body = client.get('/watch').get_data(as_text=True)
    assert '<script>alert' not in body and '<img src=x' not in body and '<b>bold</b>' not in body
    assert '&lt;script&gt;alert(&#34;title&#34;)&lt;/script&gt;' in body
