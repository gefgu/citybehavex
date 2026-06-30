from __future__ import annotations

from . import comparison as _comparison
from .config import ComparisonConfig

for _name in dir(_comparison):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_comparison, _name)

__all__ = [
    "ComparisonConfig",
    *[
        name
        for name in globals()
        if not name.startswith("__") and name not in {"ComparisonConfig", "_comparison", "_name"}
    ],
]

del _name
