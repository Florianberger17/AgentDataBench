"""NoiseConfiguration domain object: describes deterministic error injection
for a benchmark package's NoiseEngine step."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from agentdatabench.domain.common import StrictBaseModel


class NoiseTypeConfig(BaseModel):
    """Generic noise-type spec. Only `type`/`probability` are fixed; other
    keys are noise-type-specific extras interpreted by a NoiseModel registry.
    """

    model_config = ConfigDict(extra="allow")

    type: str
    probability: float


class NoiseConfiguration(StrictBaseModel):
    seed: int
    excluded_columns: list[str] = []
    noise_types: list[NoiseTypeConfig]
