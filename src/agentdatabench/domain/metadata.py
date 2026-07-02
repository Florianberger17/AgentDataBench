"""Metadata domain object: bookkeeping information about a benchmark package."""

from __future__ import annotations

from typing import Literal

from pydantic import field_validator

from agentdatabench.domain.common import StrictBaseModel


class Metadata(StrictBaseModel):
    scenario_id: str
    task_id: str
    version: str
    author: str
    created: str
    difficulty: Literal["easy", "medium", "hard"]
    seed: int | None = None

    @field_validator("version", mode="before")
    @classmethod
    def _coerce_version_to_str(cls, value: object) -> object:
        if isinstance(value, (int, float)):
            return str(value)
        return value
