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
