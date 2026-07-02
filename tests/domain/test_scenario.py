import pytest
from pydantic import ValidationError

from agentdatabench.domain.scenario import Scenario

VALID = {
    "scenario_id": "001_TEST",
    "name": "Test Scenario",
    "domain": "ERP Data Integration",
    "description": "A test scenario.",
    "business_objects": ["Customer"],
    "systems": {
        "source": {"name": "Legacy ERP", "description": "Old system."},
        "target": {"name": "Target ERP", "description": "New system."},
    },
}


def test_scenario_happy_path():
    scenario = Scenario(**VALID)
    assert scenario.scenario_id == "001_TEST"
    assert scenario.systems.source.name == "Legacy ERP"
    assert scenario.systems.target.name == "Target ERP"


def test_scenario_missing_required_field_raises():
    data = {k: v for k, v in VALID.items() if k != "description"}
    with pytest.raises(ValidationError):
        Scenario(**data)
