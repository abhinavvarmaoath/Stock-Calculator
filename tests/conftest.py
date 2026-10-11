import pytest

import fakeapi


@pytest.fixture(autouse=True)
def noapi(request, monkeypatch):
    # tests never talk to the real api, whatever is in .env or the shell.
    # anything a test needs it sets up itself. the live tests are the exception
    if request.node.get_closest_marker('live'):
        return
    for k in ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL',
              'ANTHROPIC_MODEL', 'ANTHROPIC_PROFILE'):
        monkeypatch.delenv(k, raising=False)


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
