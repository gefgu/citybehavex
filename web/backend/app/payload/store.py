"""Process-local cache for shared comparison payload artifacts."""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Callable, Hashable, TypeVar

T = TypeVar("T")


class ArtifactStore:
    def __init__(self, max_items: int = 16):
        self._max_items = max(1, max_items)
        self._values: OrderedDict[Hashable, object] = OrderedDict()
        self._locks: dict[Hashable, threading.Lock] = {}
        self._guard = threading.Lock()

    def get_or_build(self, key: Hashable, build: Callable[[], T]) -> T:
        with self._guard:
            if key in self._values:
                value = self._values.pop(key)
                self._values[key] = value
                return value  # type: ignore[return-value]
            lock = self._locks.setdefault(key, threading.Lock())

        with lock:
            with self._guard:
                if key in self._values:
                    value = self._values.pop(key)
                    self._values[key] = value
                    return value  # type: ignore[return-value]
            value = build()
            with self._guard:
                self._values[key] = value
                while len(self._values) > self._max_items:
                    old_key, _old_value = self._values.popitem(last=False)
                    self._locks.pop(old_key, None)
                self._locks.pop(key, None)
            return value

    def clear(self) -> None:
        with self._guard:
            self._values.clear()
            self._locks.clear()


def _default_max_items() -> int:
    value = os.environ.get("CBX_WEB_ARTIFACT_CACHE_ITEMS")
    if value:
        return max(1, int(value))
    return 16


artifact_store = ArtifactStore(_default_max_items())
