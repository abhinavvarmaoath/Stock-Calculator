import threading
import time

from limiter import Limiter


class Slow(dict):
    def get(self, key, default=None):
        value = super().get(key, default)
        time.sleep(0.0005)          # lets another request in between the check and the update
        return value


def test_limit_per_address():
    lim = Limiter(10, 60)
    for i in range(10):
        assert lim.allowed('1.1.1.1', now=100 + i) == (True, 0)
    ok, wait = lim.allowed('1.1.1.1', now=110)
    assert ok is False and 40 < wait <= 61
    assert lim.allowed('2.2.2.2', now=110) == (True, 0)         # someone else is fine
    assert lim.allowed('1.1.1.1', now=100 + 61) == (True, 0)    # a minute later it is fine again


def test_the_window_is_the_one_asked_for():
    lim = Limiter(2, 600)
    assert lim.allowed('a', now=0) == (True, 0) and lim.allowed('a', now=10) == (True, 0)
    ok, wait = lim.allowed('a', now=20)
    assert ok is False and wait == 581                          # until the first hit is 10 minutes old
    assert lim.allowed('a', now=599) == (False, 2)
    assert lim.allowed('a', now=600) == (True, 0)               # the first hit is exactly 10 minutes old, so it is out


def test_a_refused_hit_does_not_extend_the_wait():
    lim = Limiter(1, 60)
    lim.allowed('a', now=0)
    for t in range(1, 50):
        assert lim.allowed('a', now=t)[0] is False
    assert lim.allowed('a', now=61) == (True, 0)


def test_forgets_quiet_addresses():
    lim = Limiter(10, 60)
    for i in range(1200):
        lim.allowed('10.0.%d.%d' % (i // 250, i % 250), now=0)
    lim.allowed('9.9.9.9', now=500)
    assert len(lim.hits) < 1200


def test_two_requests_at_once_cannot_both_take_the_last_place():
    lim = Limiter(50, 60)
    lim.hits = Slow()
    results = []
    go = threading.Barrier(16)

    def hammer():
        go.wait()
        for _ in range(5):
            results.append(lim.allowed('a', now=1)[0])

    threads = [threading.Thread(target=hammer) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 80 and results.count(True) == 50
