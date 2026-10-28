import json
import os
import smtplib
import ssl
import urllib.request
from email.message import EmailMessage
from urllib.parse import urlparse

# The ways of being told that a price has crossed something. The settings live in .env (see
# .env.example). Nothing here is needed for the watchlist to work: every alert is always
# shown on the watchlist page, and these only add email or a message in a chat app.


def money(x):
    x = round(float(x), 2)
    return ('-' if x < 0 else '') + '${:,.2f}'.format(abs(x))


def channels():
    # the ways of being told that are set up
    out = []
    if os.environ.get('ALERT_EMAIL_TO') and os.environ.get('SMTP_HOST'):
        out.append('email')
    if os.environ.get('ALERT_WEBHOOK_URL'):
        out.append('webhook')
    return out


def email(title, body):
    host = os.environ['SMTP_HOST']
    port = int(os.environ.get('SMTP_PORT') or 587)
    user, password = os.environ.get('SMTP_USER'), os.environ.get('SMTP_PASSWORD', '')
    tls = (os.environ.get('SMTP_TLS') or ('ssl' if port == 465 else 'starttls')).lower()
    if tls not in ('starttls', 'ssl', 'none'):
        raise ValueError("SMTP_TLS has to be starttls, ssl or none")
    if user and tls == 'none':
        raise ValueError('refusing to send a password without TLS')

    msg = EmailMessage()
    msg['Subject'] = title
    msg['From'] = os.environ.get('SMTP_FROM') or user or 'watchlist@localhost'
    msg['To'] = os.environ['ALERT_EMAIL_TO']
    msg.set_content(body)

    ctx = ssl.create_default_context()
    smtp = smtplib.SMTP_SSL(host, port, timeout=20, context=ctx) if tls == 'ssl' else smtplib.SMTP(host, port, timeout=20)
    with smtp as s:
        if tls == 'starttls':
            s.starttls(context=ctx)
        if user:
            s.login(user, password)
        s.send_message(msg)


def webhook(title, body):
    # posts {"text": "..."}, which Slack, Mattermost and most automation tools take. Discord wants "content"
    url = os.environ['ALERT_WEBHOOK_URL']
    u = urlparse(url)
    if u.scheme not in ('http', 'https'):
        raise ValueError('the webhook address has to start with http:// or https://')
    key = 'content' if u.hostname in ('discord.com', 'discordapp.com') and u.path.startswith('/api/webhooks') else 'text'
    req = urllib.request.Request(url, data=json.dumps({key: title + '\n' + body}).encode(),
                                 headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=15) as r:     # anything but 2xx raises
        r.read(1000)


ways = dict(email=email, webhook=webhook)


def secrets():
    # what must never show up in an error: the webhook address (its path is the token), and the password
    out = []
    url = os.environ.get('ALERT_WEBHOOK_URL')
    if url:
        p = urlparse(url)
        out += [url, p.path, p.query]
    out.append(os.environ.get('SMTP_PASSWORD'))
    return sorted({x for x in out if x and len(x) > 3}, key=len, reverse=True)


def send(title, body):
    # tries every way that is set up and gives back {way: 'ok' or what went wrong}. the answer gets shown
    # on the page and stored, so the secrets are taken out of it first
    out = {}
    for name in channels():
        try:
            ways[name](title, body)
            out[name] = 'ok'
        except Exception as e:
            why = ' '.join(str(e).split())
            for secret in secrets():
                why = why.replace(secret, '***')
            out[name] = '{}: {}'.format(type(e).__name__, why[:120])
    return out
