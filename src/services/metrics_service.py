import time


class Timer:
    """Simple perf-counter based timer."""

    def __init__(self):
        self._start: float | None = None

    def start(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        if self._start is None:
            return 0.0
        elapsed = time.perf_counter() - self._start
        self._start = None
        return round(elapsed, 3)
