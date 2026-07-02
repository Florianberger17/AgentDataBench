import pandas as pd
import pytest

from migratebench.domain.task import MappingRule
from migratebench.generator.transformations import (
    ConcatenateHandler,
    CopyHandler,
    DateFormatHandler,
    SequentialNumberHandler,
    ValueMappingHandler,
)


def test_copy_handler():
    df = pd.DataFrame({"src": ["a", "b"]})
    mapping = MappingRule(
        source_field="src", target_field="dst", transformation={"type": "copy"}
    )
    result = CopyHandler().apply(df, mapping)
    assert list(result) == ["a", "b"]


def test_concatenate_handler_preserves_internal_spacing():
    df = pd.DataFrame({"a": ["1000-0001"], "b": ["slide  complete"]})
    mapping = MappingRule(
        source_fields=["a", "b"],
        target_field="description",
        transformation={"type": "concatenate", "separator": " "},
    )
    result = ConcatenateHandler().apply(df, mapping)
    assert list(result) == ["1000-0001 slide  complete"]


def test_value_mapping_handler_maps_known_values():
    df = pd.DataFrame({"country_code": ["DE", "AT", "CH"]})
    mapping = MappingRule(
        source_field="country_code",
        target_field="country",
        transformation={
            "type": "value_mapping",
            "mapping": {"DE": "DEU", "AT": "AUT", "CH": "CHE"},
        },
    )
    result = ValueMappingHandler().apply(df, mapping)
    assert list(result) == ["DEU", "AUT", "CHE"]


def test_value_mapping_handler_raises_on_unmapped_value():
    df = pd.DataFrame({"country_code": ["DE", "FR"]})
    mapping = MappingRule(
        source_field="country_code",
        target_field="country",
        transformation={"type": "value_mapping", "mapping": {"DE": "DEU"}},
    )
    with pytest.raises(ValueError, match="FR"):
        ValueMappingHandler().apply(df, mapping)


def test_date_format_handler_iso_to_ddmmyyyy():
    df = pd.DataFrame({"date": ["2025-01-30", "2026-03-19"]})
    mapping = MappingRule(
        source_field="date",
        target_field="out_date",
        transformation={
            "type": "date_format",
            "input_format": "YYYY-MM-DD",
            "output_format": "DDMMYYYY",
        },
    )
    result = DateFormatHandler().apply(df, mapping)
    assert list(result) == ["30012025", "19032026"]


def test_date_format_handler_dotted_two_digit_year():
    df = pd.DataFrame({"date": ["19.04.01"]})
    mapping = MappingRule(
        source_field="date",
        target_field="out_date",
        transformation={
            "type": "date_format",
            "input_format": "DD.MM.YY",
            "output_format": "DD.MM.YYYY",
        },
    )
    result = DateFormatHandler().apply(df, mapping)
    assert list(result) == ["19.04.2001"]


def test_sequential_number_handler_start_increment_digits():
    df = pd.DataFrame({"x": [0, 0, 0]})
    mapping = MappingRule(
        source_field="x",
        target_field="erp_no",
        transformation={
            "type": "sequential_number",
            "start": 30000000000,
            "increment": 1,
            "digits": 11,
        },
    )
    result = SequentialNumberHandler().apply(df, mapping)
    assert list(result) == ["30000000000", "30000000001", "30000000002"]
    assert len(set(result)) == len(result)
