import pytest
from pydantic import ValidationError

from agentdatabench.domain.task import FilteringRules, MappingRule, TransformationSpec

INLINE_FILTER = {
    "description": "Only recent customers.",
    "field": "LastOrderDate",
    "operator": ">=",
    "value": "2023-01-01",
}

LIST_FILTER = {
    "description": "Exclude inactive parts.",
    "rules": [{"field": "part status", "operator": "<=", "value": 1}],
}


def test_filtering_rules_inline_shape_normalizes_to_rules_list():
    parsed = FilteringRules(**INLINE_FILTER)
    assert len(parsed.rules) == 1
    assert parsed.rules[0].field == "LastOrderDate"
    assert parsed.rules[0].operator == ">="
    assert parsed.rules[0].value == "2023-01-01"


def test_filtering_rules_list_shape_parses_directly():
    parsed = FilteringRules(**LIST_FILTER)
    assert len(parsed.rules) == 1
    assert parsed.rules[0].field == "part status"
    assert parsed.rules[0].value == 1


@pytest.mark.parametrize(
    "spec",
    [
        {"type": "sequential_number", "start": 30000000000, "increment": 1, "digits": 11},
        {"type": "concatenate", "separator": " "},
        {"type": "copy"},
        {"type": "value_mapping", "mapping": {"spare part": "erze"}},
        {"type": "date_format", "input_format": "DD.MM.YY", "output_format": "DD.MM.YYYY"},
    ],
)
def test_transformation_spec_passes_through_extra_fields(spec):
    parsed = TransformationSpec(**spec)
    assert parsed.type == spec["type"]
    for key, value in spec.items():
        if key == "type":
            continue
        assert getattr(parsed, key) == value


def test_mapping_rule_requires_exactly_one_source():
    base = {"target_field": "description", "transformation": {"type": "copy"}}

    with pytest.raises(ValidationError):
        MappingRule(**base)  # neither source_field nor source_fields

    with pytest.raises(ValidationError):
        MappingRule(**base, source_field="a", source_fields=["a", "b"])  # both

    ok = MappingRule(**base, source_field="a")
    assert ok.source_field == "a"

    ok2 = MappingRule(**base, source_fields=["a", "b"])
    assert ok2.source_fields == ["a", "b"]
