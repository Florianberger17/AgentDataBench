import pytest
from pydantic import ValidationError

from migratebench.domain.schema import Schema

VALID = {
    "table": "customer_master",
    "description": "Test schema.",
    "attributes": [
        {"name": "CustNo", "type": "string", "required": True, "unique": True},
        {"name": "CompanyName", "type": "string", "required": True},
    ],
}


def test_schema_happy_path_with_optional_fields_omitted():
    schema = Schema(**VALID)
    assert schema.entity is None
    assert schema.constraints is None
    assert schema.used_by is None
    assert schema.attributes[0].unique is True
    assert schema.attributes[1].unique is False


def test_schema_happy_path_with_all_optional_fields():
    data = {
        **VALID,
        "entity": "Customer",
        "constraints": [{"id": "CON-001", "description": "CustNo must be unique"}],
        "used_by": ["Evaluation Engine"],
    }
    schema = Schema(**data)
    assert schema.entity == "Customer"
    assert schema.constraints[0].id == "CON-001"
    assert schema.used_by == ["Evaluation Engine"]


def test_schema_missing_required_field_raises():
    data = {k: v for k, v in VALID.items() if k != "table"}
    with pytest.raises(ValidationError):
        Schema(**data)
