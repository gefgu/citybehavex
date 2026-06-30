from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict


class ComparisonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Optional[str] = None
    label: str = "observed"
    html: str = "comparison.html"
