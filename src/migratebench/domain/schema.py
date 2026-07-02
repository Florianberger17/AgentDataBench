"""Schema domain object: structural schema definition for a source or target dataset."""

from __future__ import annotations

from migratebench.domain.common import StrictBaseModel


class Attribute(StrictBaseModel):
    name: str
    type: str
    required: bool
    unique: bool = False
    description: str | None = None
    format: str | None = None
    allowed_values: list[str] | None = None


class SchemaConstraint(StrictBaseModel):
    id: str
    description: str


class Schema(StrictBaseModel):
    table: str
    entity: str | None = None
    description: str
    attributes: list[Attribute]
    constraints: list[SchemaConstraint] | None = None
    used_by: list[str] | None = None
