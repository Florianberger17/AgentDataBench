"""Task domain object: the migration task definition, including business rules.

The ``business_rules.filtering`` block is observed in two shapes across existing
benchmark packages (an inline single rule, or an explicit ``rules`` list) and is
normalized to a single internal shape. ``TransformationSpec`` is deliberately
generic (``type`` + arbitrary extra fields) rather than a hardcoded union of
transformation classes, so that new transformation types can be introduced by a
future ``GroundTruthCreator`` registry without changing this core domain model.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from agentdatabench.domain.common import StrictBaseModel


class FilterRule(StrictBaseModel):
    """One condition a record has to satisfy to be migrated.

    ``value`` is optional because not every operator compares against one:
    ``is empty``/``is not empty`` test the field itself, and ``in``/``not in``
    test membership in a reference table named by ``reference``/``key_field``
    (see ``ReferenceData``).
    """

    # A list for the `in`/`not in` operators when the reference table is keyed
    # on more than one column; `field` and `key_field` pair up positionally.
    field: str | list[str]
    operator: str
    value: Any = None
    # Date-token format (e.g. "DD.MM.YYYY") of `field` in the source data,
    # when it isn't in the ISO format `value` is authored in.
    field_format: str | None = None
    # For the `in`/`not in` operators: the reference table to test against and
    # the column of it holding the keys.
    reference: str | None = None
    key_field: str | list[str] | None = None


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
                        key: value
                        for key, value in data.items()
                        if key != "description"
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
    """One target field and where its value comes from.

    ``source_field`` names a single source column, ``source_fields`` several
    (e.g. ``concatenate``). Both may be omitted: a source-independent
    transformation such as ``constant`` or ``sequence`` derives the target
    value from the row count alone, not from any source column. Setting both
    at once stays an error, since the transformation handlers read exactly
    one of them.
    """

    source_field: str | None = None
    source_fields: list[str] | None = None
    target_field: str
    transformation: TransformationSpec
    description: str | None = None

    @model_validator(mode="after")
    def _at_most_one_source(self) -> "MappingRule":
        if self.source_field is not None and self.source_fields is not None:
            raise ValueError(
                "MappingRule accepts 'source_field' or 'source_fields', not both"
            )
        return self


class RecordOrder(StrictBaseModel):
    """The order the surviving records are processed in.

    Only meaningful when the output depends on position - a
    ``sequential_number`` target field assigns its numbers in this order, so
    without it the expected result would not be defined for a source file in
    arbitrary order.
    """

    description: str | None = None
    fields: list[str]
    direction: Literal["ascending", "descending"] = "ascending"


class Grouping(StrictBaseModel):
    """The records that belong together as one target document.

    Several source rows can map to one document of the target system - all
    items of one sales order become one sales order there. A target field
    numbered ``assigned_per: group`` then draws one number per group instead
    of one per row.
    """

    description: str | None = None
    key_fields: list[str]


class Aggregate(StrictBaseModel):
    """One figure accumulated over the records of a group.

    ``name`` is the source column the result is written to, so a mapping rule
    reads it like any other field of the dataset. It is deliberately not a
    target field: a credit limit derived from a historical revenue needs that
    revenue as an intermediate, and the target schema has no column for it.

    ``where`` restricts the records the figure is accumulated over, using the
    same rule vocabulary as ``business_rules.filtering``. Filtering cannot do
    this job: two figures of the same group can rest on *different* records -
    actual costs on the cost postings, the planned revenue on the order-value
    postings - so a rule that dropped either set would destroy the other
    figure. A group in which no record matches contributes nothing, which for
    ``sum`` and ``count`` is zero rather than an empty cell.

    ``factor`` is multiplied into the result, for the sign and unit
    conventions two systems rarely share - a revenue posted as a credit read
    back as a positive limit.
    """

    name: str
    source_field: str
    function: Literal["sum", "count", "min", "max", "first", "last"]
    # Decimal places of the result. Without it a summed amount is rendered
    # with whatever precision the addition happened to produce.
    decimals: int | None = None
    where: FilteringRules | None = None
    factor: float | None = None
    description: str | None = None


class Aggregation(StrictBaseModel):
    """Collapses the records of a group into one record.

    Unlike ``Grouping``, which only names what belongs together while every
    source row still becomes one target row, aggregation changes the
    cardinality: many source records become one target record. A credit master
    record per customer is derived from every order that customer ever placed.

    Columns that are neither a key nor an aggregate keep the value of the
    group's first record - they are constant within the group by construction
    (a customer's name) or irrelevant to the mapping.
    """

    description: str | None = None
    key_fields: list[str]
    aggregates: list[Aggregate]


class BusinessRules(StrictBaseModel):
    filtering: FilteringRules | None = None
    # Applied after filtering and after unmappable records are dropped, and
    # before record_order - see GroundTruthCreator.
    aggregation: Aggregation | None = None
    record_order: RecordOrder | None = None
    grouping: Grouping | None = None
    mappings: list[MappingRule] | None = None


class ReferenceData(StrictBaseModel):
    """A lookup table the task resolves values against (e.g. a supplier
    mapping list assigning legacy supplier numbers to target ones).

    Unlike ``TaskInput.additional_documents``, which are free-form
    attachments an agent reads for itself, a reference table is structured:
    ``key_field``/``value_field`` name the columns, so a ``lookup``
    transformation can derive the expected result from it as well.

    ``key_field`` is a list when the table is keyed on a combination of
    columns - a customer cross reference resolving customer number *and*
    address number, because one legacy customer can hold several addresses
    that became separate customers in the target system.

    ``value_field`` is a list when one table feeds several target fields (a
    material cross reference carrying both the new number and its
    description). Which column a given rule reads is decided by that rule's
    ``return_field``; the declaration here is what the agent is told about.
    """

    name: str
    file: str
    key_field: str | list[str]
    value_field: str | list[str]
    description: str | None = None


class TaskInput(StrictBaseModel):
    source_dataset: str
    # Formal schemas, or a small (e.g. 1-2 row) target_example the agent must
    # infer the mapping/target structure from instead - mutually exclusive,
    # see _schema_xor_target_example. Underspecified tasks
    # (Metadata.specification_style == "implicit") use
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
    # Lookup tables the task resolves values against - see ReferenceData.
    reference_data: list[ReferenceData] | None = None

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
    # None for implicit tasks (see TaskInput.target_example) - there
    # is no formal schema to reference.
    schema_reference: str | None = None


class Task(StrictBaseModel):
    task_id: str
    objective: str
    input: TaskInput
    # None for an implicit task (see TaskInput.target_example): naming
    # the operation categories (filtering, value_mapping, ...) up front would
    # itself leak part of what the agent is supposed to infer.
    required_operations: list[str] | None = None
    business_rules: BusinessRules
    ignored_source_fields: list[str] | None = None
    output: TaskOutput
    constraints: list[str]
