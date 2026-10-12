import threading
import time


class Limiter:
    # at most `limit` hits from one address in any `window` seconds
    def __init__(self, limit, window):
        self.limit, self.window = limit, window
        self.hits = {}                  # address -> times of its recent hits
        self.lock = threading.Lock()

    def allowed(self, who, now=None):
        # (ok, seconds to wait)
        now = time.time() if now is None else now
        with self.lock:
            recent = [t for t in self.hits.get(who, []) if now - t < self.window]
            if len(recent) >= self.limit:
                self.hits[who] = recent
                return False, int(self.window - (now - recent[0])) + 1
            self.hits[who] = recent + [now]
            if len(self.hits) > 1000:       # forget addresses that have gone quiet
                for k in [k for k, v in self.hits.items() if now - v[-1] >= self.window]:
                    del self.hits[k]
            return True, 0
