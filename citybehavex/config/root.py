from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from citybehavex.activities.config import ActivitiesConfig
from citybehavex.embedding.config import EmbeddingConfig
from citybehavex.llm.config import LLMConfig
from citybehavex.llm_diaries.config import DiariesConfig
from citybehavex.profiles.config import AgentProfilesConfig
from citybehavex.reports.config import ComparisonConfig
from citybehavex.schedules.config import ScheduleConfig
from citybehavex.simulation.config import SimulationConfig
from citybehavex.tessellation.config import TessellationConfig


class CityBehavExConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tessellation: TessellationConfig = Field(default_factory=TessellationConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    diaries: DiariesConfig = Field(default_factory=DiariesConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)
    profiles: AgentProfilesConfig = Field(default_factory=AgentProfilesConfig)
    activities: ActivitiesConfig = Field(default_factory=ActivitiesConfig)
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)
