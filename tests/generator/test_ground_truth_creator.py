import pandas as pd
import pytest

from agentdatabench.domain.schema import Schema
from agentdatabench.domain.task import Task
from agentdatabench.generator.ground_truth_creator import GroundTruthCreator


def _task(**business_rules_overrides):
    business_rules = {
        "mappings": [
            {
                "source_field": "src_a",
                "target_field": "a",
                "transformation": {"type": "copy"},
            }
        ],
        **business_rules_overrides,
    }
    return Task(
        task_id="T1",
        objective="test",
        input={
            "source_dataset": "data/dataset.csv",
            "source_schema": "schemas/source_schema.yaml",
            "target_schema": "schemas/target_schema.yaml",
        },
        required_operations=["schema_mapping"],
        business_rules=business_rules,
        output={"format": "csv", "schema_reference": "schemas/target_schema.yaml"},
        constraints=[],
    )


def _schema(attributes):
    return Schema(table="t", description="d", attributes=attributes)


def test_optional_unmapped_column_is_skipped():
    df = pd.DataFrame({"src_a": ["x", "y"]})
    task = _task()
    schema = _schema(
        [
            {"name": "a", "type": "string", "required": True},
            {"name": "b", "type": "string", "required": False},
        ]
    )
    result = GroundTruthCreator().create_ground_truth(df, task, schema)
    assert list(result.columns) == ["a"]


def test_missing_required_mapping_raises():
    df = pd.DataFrame({"src_a": ["x", "y"]})
    task = _task()
    schema = _schema(
        [
            {"name": "a", "type": "string", "required": True},
            {"name": "b", "type": "string", "required": True},
        ]
    )
    with pytest.raises(ValueError, match="b"):
        GroundTruthCreator().create_ground_truth(df, task, schema)


def test_unknown_mapping_target_field_raises():
    df = pd.DataFrame({"src_a": ["x", "y"]})
    task = _task(
        mappings=[
            {
                "source_field": "src_a",
                "target_field": "not_in_schema",
                "transformation": {"type": "copy"},
            }
        ]
    )
    schema = _schema([{"name": "a", "type": "string", "required": True}])
    with pytest.raises(ValueError, match="not_in_schema"):
        GroundTruthCreator().create_ground_truth(df, task, schema)


def test_unregistered_transformation_type_raises():
    df = pd.DataFrame({"src_a": ["x", "y"]})
    task = _task(
        mappings=[
            {
                "source_field": "src_a",
                "target_field": "a",
                "transformation": {"type": "does_not_exist"},
            }
        ]
    )
    schema = _schema([{"name": "a", "type": "string", "required": True}])
    with pytest.raises(ValueError, match="does_not_exist"):
        GroundTruthCreator().create_ground_truth(df, task, schema)


def test_lookup_exclude_record_drops_row_from_every_column():
    """An unresolvable lookup key removes the whole record, so the other
    columns must not keep a value for it either."""
    df = pd.DataFrame({"src_a": ["x", "y", "z"], "supplier": ["1", "unknown", "2"]})
    task = _task(
        mappings=[
            {
                "source_field": "src_a",
                "target_field": "a",
                "transformation": {"type": "copy"},
            },
            {
                "source_field": "supplier",
                "target_field": "b",
                "transformation": {
                    "type": "lookup",
                    "reference": "suppliers",
                    "lookup_key": "legacy",
                    "return_field": "new",
                    "on_missing": "exclude_record",
                },
            },
        ]
    )
    schema = _schema(
        [
            {"name": "a", "type": "string", "required": True},
            {"name": "b", "type": "string", "required": True},
        ]
    )
    reference = {"suppliers": pd.DataFrame({"legacy": ["1", "2"], "new": ["100", "200"]})}

    result = GroundTruthCreator().create_ground_truth(df, task, schema, reference)

    assert len(result) == 2
    assert list(result["a"]) == ["x", "z"]
    assert list(result["b"]) == ["100", "200"]


def test_filtering_runs_before_lookup_exclusion():
    """A record the filter already removed must not make the lookup fail on
    a key that only exists in filtered-out rows."""
    df = pd.DataFrame(
        {"src_a": ["x", "y"], "supplier": ["1", "unknown"], "status": ["keep", "drop"]}
    )
    task = _task(
        filtering={"field": "status", "operator": "!=", "value": "drop"},
        mappings=[
            {
                "source_field": "src_a",
                "target_field": "a",
                "transformation": {"type": "copy"},
            },
            {
                "source_field": "supplier",
                "target_field": "b",
                "transformation": {
                    "type": "lookup",
                    "reference": "suppliers",
                    "lookup_key": "legacy",
                    "return_field": "new",
                    "on_missing": "exclude_record",
                },
            },
        ],
    )
    schema = _schema(
        [
            {"name": "a", "type": "string", "required": True},
            {"name": "b", "type": "string", "required": True},
        ]
    )
    reference = {"suppliers": pd.DataFrame({"legacy": ["1"], "new": ["100"]})}

    result = GroundTruthCreator().create_ground_truth(df, task, schema, reference)

    assert list(result["a"]) == ["x"]
    assert list(result["b"]) == ["100"]


def test_record_order_sorts_before_sequential_numbers_are_assigned():
    """The numbers follow the configured order, not the source file's."""
    df = pd.DataFrame({"src_a": ["c", "a", "b"], "seq": ["3", "1", "2"]})
    task = _task(
        record_order={"fields": ["seq"], "direction": "ascending"},
        mappings=[
            {
                "source_field": "src_a",
                "target_field": "a",
                "transformation": {"type": "copy"},
            },
            {
                "target_field": "b",
                "transformation": {"type": "sequential_number", "start": 100},
            },
        ],
    )
    schema = _schema(
        [
            {"name": "a", "type": "string", "required": True},
            {"name": "b", "type": "string", "required": True},
        ]
    )

    result = GroundTruthCreator().create_ground_truth(df, task, schema)

    assert list(result["a"]) == ["a", "b", "c"]
    assert list(result["b"]) == ["100", "101", "102"]


def test_record_order_sorts_numeric_columns_numerically():
    """Every column is text, so "10" must not sort before "9"."""
    df = pd.DataFrame({"src_a": ["x", "y"], "seq": ["10", "9"]})
    task = _task(
        record_order={"fields": ["seq"]},
        mappings=[
            {"source_field": "src_a", "target_field": "a", "transformation": {"type": "copy"}}
        ],
    )
    schema = _schema([{"name": "a", "type": "string", "required": True}])

    assert list(GroundTruthCreator().create_ground_truth(df, task, schema)["a"]) == ["y", "x"]


def test_record_order_can_sort_descending():
    df = pd.DataFrame({"src_a": ["x", "y"], "seq": ["1", "2"]})
    task = _task(
        record_order={"fields": ["seq"], "direction": "descending"},
        mappings=[
            {"source_field": "src_a", "target_field": "a", "transformation": {"type": "copy"}}
        ],
    )
    schema = _schema([{"name": "a", "type": "string", "required": True}])

    assert list(GroundTruthCreator().create_ground_truth(df, task, schema)["a"]) == ["y", "x"]


def test_record_order_rejects_an_unknown_field():
    df = pd.DataFrame({"src_a": ["x"]})
    task = _task(record_order={"fields": ["nope"]})
    schema = _schema([{"name": "a", "type": "string", "required": True}])

    with pytest.raises(ValueError, match="nope"):
        GroundTruthCreator().create_ground_truth(df, task, schema)


def test_record_order_runs_after_filtering():
    """A filtered-out row must not consume a sequential number."""
    df = pd.DataFrame(
        {"src_a": ["c", "a", "b"], "seq": ["3", "1", "2"], "keep": ["y", "n", "y"]}
    )
    task = _task(
        filtering={"field": "keep", "operator": "==", "value": "y"},
        record_order={"fields": ["seq"]},
        mappings=[
            {"source_field": "src_a", "target_field": "a", "transformation": {"type": "copy"}},
            {"target_field": "b", "transformation": {"type": "sequential_number", "start": 1}},
        ],
    )
    schema = _schema(
        [
            {"name": "a", "type": "string", "required": True},
            {"name": "b", "type": "string", "required": True},
        ]
    )

    result = GroundTruthCreator().create_ground_truth(df, task, schema)

    assert list(result["a"]) == ["b", "c"]
    assert list(result["b"]) == ["1", "2"]


def _aggregating_task(**overrides):
    business_rules = {
        "filtering": {"rules": [{"field": "status", "operator": "!=", "value": "open"}]},
        "aggregation": {
            "key_fields": ["customer"],
            "aggregates": [
                {"name": "revenue", "source_field": "amount", "function": "sum"}
            ],
        },
        "record_order": {"fields": ["customer"]},
        "mappings": [
            {
                "source_field": "customer",
                "target_field": "a",
                "transformation": {"type": "copy"},
            },
            {
                "source_field": "revenue",
                "target_field": "b",
                "transformation": {"type": "copy"},
            },
        ],
    }
    business_rules.update(overrides)
    return Task(
        task_id="T1",
        objective="test",
        input={
            "source_dataset": "data/dataset.csv",
            "source_schema": "schemas/source_schema.yaml",
            "target_schema": "schemas/target_schema.yaml",
        },
        required_operations=["aggregation"],
        business_rules=business_rules,
        output={"format": "csv", "schema_reference": "schemas/target_schema.yaml"},
        constraints=[],
    )


def test_aggregation_collapses_the_records_of_a_group_into_one_row():
    df = pd.DataFrame(
        {
            "customer": ["B", "A", "B", "A"],
            "status": ["done", "done", "done", "done"],
            "amount": ["1", "7", "2", "3"],
        }
    )
    schema = _schema(
        [
            {"name": "a", "type": "string", "required": True},
            {"name": "b", "type": "string", "required": True},
        ]
    )
    result = GroundTruthCreator().create_ground_truth(df, _aggregating_task(), schema)
    # record_order sorts the collapsed records, not the source rows.
    assert list(result["a"]) == ["A", "B"]
    assert list(result["b"]) == ["10", "3"]


def test_filtered_records_do_not_contribute_to_an_aggregate():
    """Aggregation has to run after filtering: an order the migration does not
    carry over must not end up in the sum either."""
    df = pd.DataFrame(
        {
            "customer": ["A", "A"],
            "status": ["done", "open"],
            "amount": ["3", "1000"],
        }
    )
    schema = _schema(
        [
            {"name": "a", "type": "string", "required": True},
            {"name": "b", "type": "string", "required": True},
        ]
    )
    result = GroundTruthCreator().create_ground_truth(df, _aggregating_task(), schema)
    assert list(result["b"]) == ["3"]


def _expanding_task(**overrides):
    business_rules = {
        "filtering": {
            "rules": [{"field": "status", "operator": "==", "value": "active"}]
        },
        "expansion": {
            "reference": "cross_reference",
            "source_field": "cost center",
            "key_field": "legacy",
            "carry_fields": ["new"],
        },
        "record_order": {"fields": ["new"]},
        "mappings": [
            {
                "source_field": "new",
                "target_field": "a",
                "transformation": {"type": "copy"},
            },
            {
                "source_field": "description",
                "target_field": "b",
                "transformation": {"type": "copy"},
            },
        ],
    }
    business_rules.update(overrides)
    return Task(
        task_id="T1",
        objective="test",
        input={
            "source_dataset": "data/dataset.csv",
            "source_schema": "schemas/source_schema.yaml",
            "target_schema": "schemas/target_schema.yaml",
            "reference_data": [
                {
                    "name": "cross_reference",
                    "file": "data/cross_reference.csv",
                    "key_field": "legacy",
                    "value_field": "new",
                }
            ],
        },
        required_operations=["expansion"],
        business_rules=business_rules,
        output={"format": "csv", "schema_reference": "schemas/target_schema.yaml"},
        constraints=[],
    )


def _split_cost_centers():
    return pd.DataFrame(
        {
            "cost center": ["K1", "K2", "K3"],
            "description": ["one", "two", "three"],
            "status": ["active", "active", "blocked"],
        }
    )


def _split_reference():
    return {
        "cross_reference": pd.DataFrame(
            {
                "legacy": ["K1", "K2", "K1", "K3"],
                "new": ["N3", "N2", "N1", "N4"],
            }
        )
    }


def _two_column_schema():
    return Schema(
        table="t",
        description="d",
        attributes=[
            {"name": "a", "type": "string", "required": True},
            {"name": "b", "type": "string", "required": True},
        ],
    )


def test_expansion_turns_one_record_into_one_row_per_successor():
    result = GroundTruthCreator().create_ground_truth(
        _split_cost_centers(),
        _expanding_task(),
        _two_column_schema(),
        _split_reference(),
    )

    # K1 has two successors, K2 one, K3 is filtered out before expanding.
    # record_order then sorts the expanded rows, not the source rows.
    assert list(result["a"]) == ["N1", "N2", "N3"]
    assert list(result["b"]) == ["one", "two", "one"]


def test_a_filtered_record_acquires_no_successors():
    result = GroundTruthCreator().create_ground_truth(
        _split_cost_centers(),
        _expanding_task(),
        _two_column_schema(),
        _split_reference(),
    )

    assert "N4" not in set(result["a"])
