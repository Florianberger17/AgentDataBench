"""Scenario domain object: the business/fachliches scenario a benchmark task is embedded in."""

from __future__ import annotations

from migratebench.domain.common import StrictBaseModel


class SystemInfo(StrictBaseModel):
    name: str
    description: str


class Systems(StrictBaseModel):
    source: SystemInfo
    target: SystemInfo


class Scenario(StrictBaseModel):
    scenario_id: str
    name: str
    domain: str
    description: str
    business_objects: list[str]
    systems: Systems
