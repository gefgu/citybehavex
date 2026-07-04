from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AgentProfilesConfig(BaseModel):
    """Configuration for agent demographic profile generation."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    profiles_path: Optional[str] = None
    output: str = "agent_profiles.parquet"
    llm_override: bool = False
    home_anchors_path: Optional[str] = None
    home_anchors_output: Optional[str] = None
    home_anchor_relevance: float = Field(default=1.0, gt=0)
    home_anchor_h3_resolution: int = Field(default=9, ge=0, le=15)
    location_inference_method: Literal["poi_building", "legacy_poi"] = "poi_building"
    overture_building_features_path: Optional[str] = None
    overture_building_features_output: Optional[str] = None
    overture_feature_h3_resolution: Optional[int] = Field(default=None, ge=0, le=15)
    home_poi_inverse_weight: float = Field(default=0.5, ge=0)
    home_building_weight: float = Field(default=1.0, ge=0)
    work_poi_weight: float = Field(default=0.75, ge=0)
    work_building_weight: float = Field(default=1.0, ge=0)

    age_beta_a: float = Field(default=2.0, gt=0)
    age_beta_b: float = Field(default=5.0, gt=0)
    age_min: int = Field(default=16, ge=0)
    age_max: int = Field(default=80, ge=0)

    education_weights: list[float] = Field(
        default_factory=lambda: [0.08, 0.32, 0.23, 0.27, 0.10]
    )
    health_weights: list[float] = Field(
        default_factory=lambda: [0.02, 0.07, 0.21, 0.45, 0.25]
    )
    household_weights: list[float] = Field(
        default_factory=lambda: [0.08, 0.21, 0.14, 0.05, 0.06, 0.12, 0.34]
    )
    job_weights: list[float] = Field(
        default_factory=lambda: [0.10, 0.22, 0.16, 0.12, 0.18, 0.03, 0.08, 0.06, 0.05]
    )

    car_probability: float = Field(default=0.55, ge=0.0, le=1.0)
    bike_probability: float = Field(default=0.35, ge=0.0, le=1.0)

    male_names: list[str] = Field(
        default_factory=lambda: [
            "James", "John", "Robert", "Michael", "William", "David",
            "Richard", "Joseph", "Thomas", "Charles", "Daniel", "Matthew",
            "Lucas", "Hugo", "Théo", "Nathan", "Maxime", "Pierre", "Antoine",
            "Louis", "Julien", "Nicolas", "Clément", "Alexandre", "Thomas",
        ]
    )
    female_names: list[str] = Field(
        default_factory=lambda: [
            "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Susan",
            "Jessica", "Sarah", "Karen", "Emma", "Léa", "Clara", "Chloé",
            "Camille", "Manon", "Inès", "Lucie", "Anaïs", "Juliette", "Marie",
            "Zoé", "Alice", "Océane", "Pauline", "Charlotte",
        ]
    )

    @field_validator("education_weights", "health_weights", "household_weights", "job_weights")
    @classmethod
    def positive_weights(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("weights list must not be empty")
        if any(w < 0 for w in v):
            raise ValueError("weights must be non-negative")
        if sum(v) <= 0:
            raise ValueError("weights must sum to a positive value")
        return v

    @model_validator(mode="after")
    def validate_age_range(self) -> AgentProfilesConfig:
        if self.age_min >= self.age_max:
            raise ValueError("age_min must be less than age_max")
        return self
