from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel

from .root import CityBehavExConfig

# os.path.expandvars leaves ${VAR} as a literal string when VAR is unset,
# which config fields relying on env-var-means-configured (e.g. llm.base_url)
# then treat as a truthy, garbage "configured" value. Substitute unset vars
# with "" instead, so an intentionally-unconfigured field (like the optional
# diary LLM) actually reads as empty.
_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)\}|\$(\w+)")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(
            lambda m: os.environ.get(m.group(1) or m.group(2), ""), value
        )
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def load_config(path: Optional[str]) -> CityBehavExConfig:
    if path is None:
        return CityBehavExConfig()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("config file must contain a YAML mapping")
    return CityBehavExConfig.model_validate(_expand_env(raw))


def apply_overrides(model: BaseModel, overrides: dict[str, Any]) -> BaseModel:
    clean = {key: value for key, value in overrides.items() if value is not None}
    if not clean:
        return model
    data = model.model_dump()
    data.update(clean)
    return model.__class__.model_validate(data)
