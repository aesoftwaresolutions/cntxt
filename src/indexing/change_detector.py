import threading
import time
from typing import Any, Optional


class ChangeDetector:
    def __init__(self, indexer: Any, poll_interval: int = 300) -> None:
        self.indexer = indexer
        self.poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.indexer.index_incremental()
            except Exception as e:
                # Log exception but continue running
                pass
            self._stop_event.wait(self.poll_interval)
