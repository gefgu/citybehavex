"""Filesystem locations the backend reads from.

The repo layout is fixed: this file is ``web/backend/app/config.py`` so the repo
root is three parents up. Configs live in ``configs/``, simulation outputs under
``data/``, and the on-disk chart-payload cache under ``data/.web_cache/``.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"
CACHE_DIR = DATA_DIR / ".web_cache"
