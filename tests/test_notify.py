import smtplib
import ssl

import pytest

import notify

title, body = 'AAPL is above your break-even', 'AAPL is at $190.50.\nSelling now would net $1,016.55.'


# which ways are set up

def test_nothing_is_set_up_until_the_environment_says_so(chan):
    assert notify.channels() == []
    assert notify.send(title, body) == {}


def test_a_webhook_needs_only_its_address(chan):
    chan.hook()
    assert notify.channels() == ['webhook']


def test_email_needs_both_a_server_and_an_address(chan, monkeypatch):
    monkeypatch.setenv('SMTP_HOST', 'mail.example.com')
    assert notify.channels() == []
    monkeypatch.delenv('SMTP_HOST')
    monkeypatch.setenv('ALERT_EMAIL_TO', 'me@example.com')
    assert notify.channels() == []
    monkeypatch.setenv('SMTP_HOST', 'mail.example.com')
    assert notify.channels() == ['email']


def test_both_can_be_set_up(chan):
    chan.hook()
    chan.smtp()
    assert notify.channels() == ['email', 'webhook']


# webhook

def test_the_webhook_gets_the_alert_as_json_text(chan):
    hook = chan.hook()
    assert notify.send(title, body) == {'webhook': 'ok'}
    assert len(hook.posts) == 1
    post = hook.posts[0]
    assert post['path'] == '/services/s3cr3t-token' and post['type'] == 'application/json'
    assert post['json'] == {'text': title + '\n' + body}


@pytest.mark.parametrize('url, key', [
    ('https://discord.com/api/webhooks/123/abc', 'content'),
    ('https://discordapp.com/api/webhooks/123/abc', 'content'),
    ('https://hooks.slack.com/services/T0/B0/xyz', 'text'),
    ('https://example.com/discord.com/api/webhooks/1', 'text'),          # only the real host counts
    ('https://discord.com.evil.example/api/webhooks/1', 'text'),
])
def test_discord_wants_content_and_everyone_else_text(chan, monkeypatch, url, key):
    seen = []

    def fake(req, timeout):
        seen.append(req)
        raise OSError('stop here')
    monkeypatch.setattr(notify.urllib.request, 'urlopen', fake)
    monkeypatch.setenv('ALERT_WEBHOOK_URL', url)
    notify.send(title, body)
    import json
    assert list(json.loads(seen[0].data)) == [key]


@pytest.mark.parametrize('status', [400, 404, 500])
def test_a_webhook_that_says_no_is_reported(chan, status):
    chan.hook(status=status)
    out = notify.send(title, body)
    assert out['webhook'].startswith('HTTPError: HTTP Error {}'.format(status))


def test_the_webhook_address_is_never_in_what_is_reported(chan, monkeypatch):
    hook = chan.hook(status=500, token='TOKEN-abc123-xyz')
    out = notify.send(title, body)
    assert 'TOKEN-abc123-xyz' not in str(out) and hook.url not in str(out)

    # an address with a mistake in it makes python quote part of it in its own message
    monkeypatch.setenv('ALERT_WEBHOOK_URL', 'https://hooks.example.com/services/T0/B0/TOKEN-abc123-xyz oops')
    out = notify.send(title, body)
    assert out['webhook'].startswith('InvalidURL') and 'TOKEN-abc123-xyz' not in str(out)


def test_a_webhook_nobody_answers_is_reported(chan, monkeypatch):
    monkeypatch.setenv('ALERT_WEBHOOK_URL', 'http://127.0.0.1:9/hook/TOKEN-abc123-xyz')
    out = notify.send(title, body)
    assert 'webhook' in out and out['webhook'] != 'ok' and 'TOKEN-abc123-xyz' not in out['webhook']


@pytest.mark.parametrize('url', ['ftp://example.com/x', 'file:///etc/passwd', 'example.com/hook', 'javascript:alert(1)'])
def test_only_http_and_https_addresses_are_used(chan, monkeypatch, url):
    monkeypatch.setenv('ALERT_WEBHOOK_URL', url)
    out = notify.send(title, body)
    assert out == {'webhook': 'ValueError: the webhook address has to start with http:// or https://'}


# email

def test_the_email_arrives_as_written(chan):
    smtp = chan.smtp()
    assert notify.send(title, body) == {'email': 'ok'}
    assert len(smtp.mails) == 1
    mail = smtp.mails[0]
    assert mail['to'] == ['me@example.com'] and mail['sender'] == 'watchlist@localhost'
    msg = mail['msg']
    assert msg['Subject'] == title and msg['To'] == 'me@example.com' and msg['From'] == 'watchlist@localhost'
    assert msg.get_content().replace('\r\n', '\n').strip() == body            # lines end in crlf on the wire


def test_the_from_address_can_be_set(chan, monkeypatch):
    smtp = chan.smtp()
    monkeypatch.setenv('SMTP_FROM', 'Watchlist <alerts@example.com>')
    notify.send(title, body)
    assert smtp.mails[0]['sender'] == 'alerts@example.com'
    assert smtp.mails[0]['msg']['From'] == 'Watchlist <alerts@example.com>'


def test_several_addresses_all_get_it(chan, monkeypatch):
    smtp = chan.smtp()
    monkeypatch.setenv('ALERT_EMAIL_TO', 'a@example.com, b@example.org')
    notify.send(title, body)
    assert smtp.mails[0]['to'] == ['a@example.com', 'b@example.org']


def test_other_alphabets_and_symbols_survive(chan):
    smtp = chan.smtp()
    notify.send('Café: 100€ → 120€', 'naïve $ and a line\nwith a break ✓')
    msg = smtp.mails[0]['msg']
    assert msg['Subject'] == 'Café: 100€ → 120€' and 'with a break ✓' in msg.get_content()


def test_a_refused_address_is_reported(chan):
    chan.smtp(refuse='no such user here')
    out = notify.send(title, body)
    assert out['email'].startswith('SMTPRecipientsRefused')


def test_a_mail_server_nobody_answers_is_reported(chan, monkeypatch):
    chan.smtp()
    monkeypatch.setenv('SMTP_PORT', '9')
    out = notify.send(title, body)
    assert out['email'] != 'ok' and out['email'].split(':')[0] in ('ConnectionRefusedError', 'OSError', 'TimeoutError')


def test_a_password_is_never_sent_without_tls(chan, monkeypatch):
    smtp = chan.smtp()
    monkeypatch.setenv('SMTP_USER', 'me')
    monkeypatch.setenv('SMTP_PASSWORD', 'hunter2-hunter2')
    out = notify.send(title, body)
    assert out == {'email': 'ValueError: refusing to send a password without TLS'}
    assert smtp.mails == []


def test_an_unknown_tls_setting_is_an_error(chan, monkeypatch):
    chan.smtp()
    monkeypatch.setenv('SMTP_TLS', 'maybe')
    assert notify.send(title, body)['email'].startswith('ValueError')


def checks(context):
    # whether a tls connection would check the server's certificate and name
    return None if context is None else (context.verify_mode, context.check_hostname)


class Recorder:
    # stands in for smtplib's classes, to see the order things are done in when there is tls
    log = []
    fail_login = None

    def __init__(self, *args, **kw):
        Recorder.log.append(('connect', type(self).__name__, args, {k: v for k, v in kw.items() if k != 'context'}, checks(kw.get('context'))))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        Recorder.log.append(('close',))

    def starttls(self, context=None):
        Recorder.log.append(('starttls', checks(context)))

    def login(self, user, password):
        Recorder.log.append(('login', user, password))
        if Recorder.fail_login:
            raise smtplib.SMTPException(Recorder.fail_login)

    def send_message(self, msg):
        Recorder.log.append(('send', msg['To']))


class RecSMTP(Recorder):
    pass


class RecSSL(Recorder):
    pass


@pytest.fixture
def rec(chan, monkeypatch):
    Recorder.log, Recorder.fail_login = [], None
    monkeypatch.setattr(smtplib, 'SMTP', RecSMTP)
    monkeypatch.setattr(smtplib, 'SMTP_SSL', RecSSL)
    monkeypatch.setenv('SMTP_HOST', 'mail.example.com')
    monkeypatch.setenv('ALERT_EMAIL_TO', 'me@example.com')
    monkeypatch.setenv('SMTP_USER', 'me@example.com')
    monkeypatch.setenv('SMTP_PASSWORD', 'hunter2-hunter2')
    return Recorder


def test_starttls_comes_before_the_password(rec, monkeypatch):
    monkeypatch.setenv('SMTP_PORT', '587')
    assert notify.send(title, body) == {'email': 'ok'}
    assert [x[0] for x in rec.log] == ['connect', 'starttls', 'login', 'send', 'close']
    assert rec.log[0][1] == 'RecSMTP' and rec.log[0][2] == ('mail.example.com', 587) and rec.log[1] == ('starttls', (ssl.CERT_REQUIRED, True))
    assert rec.log[2] == ('login', 'me@example.com', 'hunter2-hunter2')


def test_port_465_means_tls_from_the_start(rec, monkeypatch):
    monkeypatch.setenv('SMTP_PORT', '465')
    assert notify.send(title, body) == {'email': 'ok'}
    assert [x[0] for x in rec.log] == ['connect', 'login', 'send', 'close']
    assert rec.log[0][1] == 'RecSSL' and rec.log[0][2] == ('mail.example.com', 465) and rec.log[0][4] == (ssl.CERT_REQUIRED, True)


def test_tls_can_be_chosen_whatever_the_port(rec, monkeypatch):
    monkeypatch.setenv('SMTP_PORT', '2525')
    monkeypatch.setenv('SMTP_TLS', 'ssl')
    notify.send(title, body)
    assert rec.log[0][1] == 'RecSSL'


def test_the_port_defaults_to_587(rec):
    notify.send(title, body)
    assert rec.log[0][2] == ('mail.example.com', 587)


def test_no_login_without_a_user(rec, monkeypatch):
    monkeypatch.delenv('SMTP_USER')
    monkeypatch.delenv('SMTP_PASSWORD')
    notify.send(title, body)
    assert 'login' not in [x[0] for x in rec.log]


def test_a_rejected_login_is_reported_without_the_password(rec):
    rec.fail_login = 'login refused for hunter2-hunter2'
    out = notify.send(title, body)
    assert out['email'].startswith('SMTPException') and 'hunter2' not in out['email'] and '***' in out['email']


# send

def test_one_way_failing_does_not_stop_the_other(chan):
    chan.hook(status=500)
    smtp = chan.smtp()
    out = notify.send(title, body)
    assert out['email'] == 'ok' and out['webhook'].startswith('HTTPError') and len(smtp.mails) == 1


def test_money_is_written_the_way_the_pages_write_it():
    assert [notify.money(x) for x in (1234.5, -0.004, -2.5, 0, 1e6)] == ['$1,234.50', '$0.00', '-$2.50', '$0.00', '$1,000,000.00']
