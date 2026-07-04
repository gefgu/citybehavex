from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator

from .comparison import ALL_REPORT_SECTIONS


class ComparisonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Optional[str] = None
    label: str = "observed"
    # Deprecated standalone metrics export; None = skip it.
    json_output: Optional[str] = None
    # Which report sections to compute; None (default) = run all of them.
    # Wasserstein/CPC summary metrics and the ECDF charts always run.
    sections: Optional[list[str]] = None

    @field_validator("sections")
    @classmethod
    def valid_sections(cls, value: Optional[list[str]]) -> Optional[list[str]]:
        if value is None:
            return value
        unknown = set(value) - ALL_REPORT_SECTIONS
        if unknown:
            raise ValueError(
                f"Unknown comparison report section(s): {sorted(unknown)}. "
                f"Valid sections: {sorted(ALL_REPORT_SECTIONS)}"
            )
        return value
