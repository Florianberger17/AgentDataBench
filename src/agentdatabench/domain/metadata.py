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
    # Two independent axes rather than one flat difficulty label: how
    # complex/realistic the underlying scenario is, and how much of the
    # task specification (schemas, mapping rules) is actually given to the
    # agent vs. left for it to infer. The four combinations are the
    # project's four benchmark difficulty levels (basic+explicit = level 1,
    # basic+underspecified = level 2, realistic+explicit = level 3,
    # realistic+underspecified = level 4) - kept as two fields instead of a
    # single "level" enum so scores can later be aggregated/filtered by
    # either axis independently.
    task_complexity: Literal["basic", "realistic"]
    specification_completeness: Literal["explicit", "underspecified"]
    # A third independent axis, same rationale as the two above: whether the
    # source data has injected data-quality issues (typos, missing values,
    # duplicates - see noise_configuration.yaml) the agent must correct
    # before mapping. Kept explicit rather than inferring it from whether
    # noise_configuration.yaml happens to exist, so a package with the file
    # present but a real intent of "clean" (or vice versa) is a checkable
    # authoring error - see MetadataConsistencyCheck.
    data_quality: Literal["clean", "noisy"]
    seed: int | None = None

    @field_validator("version", mode="before")
    @classmethod
    def _coerce_version_to_str(cls, value: object) -> object:
        if isinstance(value, (int, float)):
            return str(value)
        return value
