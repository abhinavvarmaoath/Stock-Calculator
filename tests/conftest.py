import pytest

import fakeapi
import fakechan


@pytest.fixture(autouse=True)
def noapi(request, monkeypatch, tmp_path):
    # tests never talk to the real api, whatever is in .env or the shell. they never send a real alert
    # or touch the real watchlist either. anything a test needs it sets up itself. the live tests are the exception
    if request.node.get_closest_marker('live'):
        return
    for k in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL',
              'ANTHROPIC_MODEL', 'ANTHROPIC_PROFILE'):
        monkeypatch.delenv(k, raising=False)
    for k in ('ALERT_EMAIL_TO', 'SMTP_HOST', 'SMTP_PORT', 'SMTP_USER', 'SMTP_PASSWORD', 'SMTP_FROM', 'SMTP_TLS',
              'ALERT_WEBHOOK_URL'):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv('WATCHLIST_DB', str(tmp_path / 'watchlist.db'))


@pytest.fixture
def fake():
    # a fake Anthropic api that replays the replies you give it. it stops when the test ends
    made = []

    def make(*replies):
        s = fakeapi.FakeAPI(replies).start()
        made.append(s)
        return s
    yield make
    for s in made:
        s.stop()


@pytest.fixture
def chan(monkeypatch):
    # the places alerts go, as real servers on this machine. settings come from the environment, like a real .env.
    # nothing is set up until the test asks for it
    made = []

    class Chan:
        def hook(self, **kw):
            h = fakechan.FakeHook(**kw).start()
            made.append(h)
            monkeypatch.setenv('ALERT_WEBHOOK_URL', h.url)
            return h

        def smtp(self, **kw):
            s = fakechan.FakeSmtp(**kw).start()
            made.append(s)
            monkeypatch.setenv('SMTP_HOST', '127.0.0.1')
            monkeypatch.setenv('SMTP_PORT', str(s.port))
            monkeypatch.setenv('SMTP_TLS', 'none')
            monkeypatch.setenv('ALERT_EMAIL_TO', 'me@example.com')
            return s
    yield Chan()
    for c in made:
        c.stop()
