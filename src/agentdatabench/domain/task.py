"""Task domain object: the migration task definition, including business rules.

The ``business_rules.filtering`` block is observed in two shapes across existing
benchmark packages (an inline single rule, or an explicit ``rules`` list) and is
normalized to a single internal shape. ``TransformationSpec`` is deliberately
generic (``type`` + arbitrary extra fields) rather than a hardcoded union of
transformation classes, so that new transformation types can be introduced by a
future ``GroundTruthCreator`` registry without changing this core domain model.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from agentdatabench.domain.common import StrictBaseModel


class FilterRule(StrictBaseModel):
    field: str
    operator: str
    value: Any
    # Date-token format (e.g. "DD.MM.YYYY") of `field` in the source data,
    # when it isn't in the ISO format `value` is authored in.
    field_format: str | None = None


class FilteringRules(StrictBaseModel):
    description: str | None = None
    rules: list[FilterRule]

    @model_validator(mode="before")
    @classmethod
    def _normalize_inline_rule(cls, data: Any) -> Any:
        if isinstance(data, dict) and "rules" not in data and "field" in data:
            data = {
                "description": data.get("description"),
                "rules": [
                    {
                        "field": data["field"],
                        "operator": data["operator"],
                        "value": data["value"],
                        "field_format": data.get("field_format"),
                    }
                ],
            }
        return data


class TransformationSpec(BaseModel):
    """Generic transformation specification.

    Only ``type`` is fixed; all other keys (e.g. ``start``/``increment``/``digits``
    for ``sequential_number``, ``separator`` for ``concatenate``, ``mapping`` for
    ``value_mapping``, ``input_format``/``output_format`` for ``date_format``) are
    passed through as extras and interpreted by a transformation-type registry.
    """

    model_config = ConfigDict(extra="allow")

    type: str


class MappingRule(StrictBaseModel):
    source_field: str | None = None
    source_fields: list[str] | None = None
    target_field: str
    transformation: TransformationSpec
    description: str | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "MappingRule":
        has_single = self.source_field is not None
        has_multi = self.source_fields is not None
        if has_single == has_multi:
            raise ValueError(
                "MappingRule requires exactly one of 'source_field' or 'source_fields'"
            )
        return self


class BusinessRules(StrictBaseModel):
    filtering: FilteringRules | None = None
    mappings: list[MappingRule] | None = None


class TaskInput(StrictBaseModel):
    source_dataset: str
    # Formal schemas, or a small (e.g. 1-2 row) target_example the agent must
    # infer the mapping/target structure from instead - mutually exclusive,
    # see _schema_xor_target_example. Underspecified tasks
    # (Metadata.specification_completeness == "underspecified") use
    # target_example; the package can still keep a real target_schema.yaml
    # on disk for internal tooling (e.g. authoring ground_truth.csv) even
    # when it isn't referenced here, since only this reference controls
    # what AgentAdapter exposes to the agent.
    source_schema: str | None = None
    target_schema: str | None = None
    target_example: str | None = None
    # Paths (relative to the package root) to supplementary files an agent
    # may need to consult for information missing from source_dataset (e.g.
    # a PDF order confirmation carrying an address the CSV lacks). Optional
    # since most tasks are self-contained within the CSV/schemas alone.
    additional_documents: list[str] | None = None

    @model_validator(mode="after")
    def _schema_xor_target_example(self) -> "TaskInput":
        has_source_schema = self.source_schema is not None
        has_target_schema = self.target_schema is not None
        if has_source_schema != has_target_schema:
            raise ValueError(
                "TaskInput requires source_schema and target_schema together, or neither"
            )
        has_schemas = has_source_schema and has_target_schema
        has_example = self.target_example is not None
        if has_schemas == has_example:
            raise ValueError(
                "TaskInput requires exactly one of (source_schema and target_schema) "
                "or target_example"
            )
        return self


class TaskOutput(StrictBaseModel):
    format: str
    # None for underspecified tasks (see TaskInput.target_example) - there
    # is no formal schema to reference.
    schema_reference: str | None = None


class Task(StrictBaseModel):
    task_id: str
    objective: str
    input: TaskInput
    # None for an underspecified task (see TaskInput.target_example): naming
    # the operation categories (filtering, value_mapping, ...) up front would
    # itself leak part of what the agent is supposed to infer.
    required_operations: list[str] | None = None
    business_rules: BusinessRules
    ignored_source_fields: list[str] | None = None
    output: TaskOutput
    constraints: list[str]
