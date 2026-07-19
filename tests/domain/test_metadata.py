import pytest
from pydantic import ValidationError

from agentdatabench.domain.metadata import Metadata

VALID = {
    "scenario_id": "001_TEST",
    "task_id": "001_TEST",
    "version": 1.0,
    "author": "Florian Berger",
    "created": "01/07/2026",
    "task_complexity": "basic",
    "specification_completeness": "explicit",
    "data_quality": "clean",
    "seed": None,
}


def test_metadata_coerces_float_version_to_str():
    metadata = Metadata(**VALID)
    assert metadata.version == "1.0"
    assert isinstance(metadata.version, str)


def test_metadata_seed_none_when_empty():
    metadata = Metadata(**VALID)
    assert metadata.seed is None


def test_metadata_invalid_task_complexity_raises():
    data = {**VALID, "task_complexity": "impossible"}
    with pytest.raises(ValidationError):
        Metadata(**data)


def test_metadata_invalid_specification_completeness_raises():
    data = {**VALID, "specification_completeness": "vague"}
    with pytest.raises(ValidationError):
        Metadata(**data)


def test_metadata_invalid_data_quality_raises():
    data = {**VALID, "data_quality": "dirty"}
    with pytest.raises(ValidationError):
        Metadata(**data)


def test_metadata_missing_required_field_raises():
    data = {k: v for k, v in VALID.items() if k != "author"}
    with pytest.raises(ValidationError):
        Metadata(**data)
