"""Tests for apply_expansion: one source record becoming several."""

import pandas as pd
import pytest

from agentdatabench.domain.task import Expansion
from agentdatabench.generator.expansion import apply_expansion


def _expansion(**overrides):
    config = {
        "reference": "cross_reference",
        "source_field": "cost center",
        "key_field": "legacy",
        "carry_fields": ["new", "unit"],
    }
    config.update(overrides)
    return Expansion(**config)


def _cost_centers():
    return pd.DataFrame(
        {
            "cost center": ["0417-5000", "0417-2001", "0417-9999"],
            "description": ["General Management", "PMO", "Travel"],
            "category": ["1", "1", "1"],
        }
    )


def _cross_reference():
    # 0417-5000 was split across two units, 0417-2001 kept as one, and
    # 0417-9999 is absent from the list entirely.
    return {
        "cross_reference": pd.DataFrame(
            {
                "legacy": ["0417-5000", "0417-2001", "0417-5000"],
                "new": ["4170-6200", "4170-3210", "0454-6400"],
                "unit": ["Location Mgmt", "PMO", "Marketing"],
            }
        )
    }


def test_none_leaves_the_frame_untouched():
    df = _cost_centers()
    pd.testing.assert_frame_equal(apply_expansion(df, None), df)


def test_one_record_per_matching_reference_entry():
    result = apply_expansion(_cost_centers(), _expansion(), _cross_reference())

    assert list(result["cost center"]) == ["0417-5000", "0417-5000", "0417-2001"]
    assert list(result["new"]) == ["4170-6200", "0454-6400", "4170-3210"]


def test_records_keep_dataset_order_and_successors_reference_order():
    reference = _cross_reference()
    # The second successor of 0417-5000 sits last in the reference table, so a
    # result ordered by the reference alone would interleave the two cost
    # centers.
    result = apply_expansion(_cost_centers(), _expansion(), reference)

    assert list(result["unit"]) == ["Location Mgmt", "Marketing", "PMO"]


def test_carried_columns_become_source_columns():
    result = apply_expansion(_cost_centers(), _expansion(), _cross_reference())

    assert "new" in result.columns
    assert "unit" in result.columns
    assert list(result.columns) == [
        "cost center",
        "description",
        "category",
        "new",
        "unit",
    ]


def test_source_columns_are_repeated_onto_every_successor():
    result = apply_expansion(_cost_centers(), _expansion(), _cross_reference())

    first = result[result["new"] == "4170-6200"].iloc[0]
    second = result[result["new"] == "0454-6400"].iloc[0]
    assert first["description"] == second["description"] == "General Management"


def test_a_record_without_a_successor_is_dropped():
    result = apply_expansion(_cost_centers(), _expansion(), _cross_reference())

    assert "0417-9999" not in set(result["cost center"])
    assert len(result) == 3


def test_only_the_carried_columns_are_taken_from_the_reference():
    expansion = _expansion(carry_fields=["new"])
    result = apply_expansion(_cost_centers(), expansion, _cross_reference())

    assert "unit" not in result.columns
    assert "legacy" not in result.columns


def test_a_composite_key_is_matched_on_every_part():
    df = pd.DataFrame(
        {
            "cost center": ["0417-5000", "0417-5000"],
            "plant": ["0417", "0319"],
        }
    )
    reference = {
        "cross_reference": pd.DataFrame(
            {
                "legacy": ["0417-5000", "0417-5000"],
                "site": ["0417", "0319"],
                "new": ["4170-6200", "0454-6400"],
            }
        )
    }
    expansion = _expansion(
        source_field=["cost center", "plant"],
        key_field=["legacy", "site"],
        carry_fields=["new"],
    )

    result = apply_expansion(df, expansion, reference)

    assert list(result["new"]) == ["4170-6200", "0454-6400"]


def test_keys_are_matched_with_surrounding_whitespace_stripped():
    df = pd.DataFrame({"cost center": [" 0417-2001 "]})
    expansion = _expansion(carry_fields=["new"])

    result = apply_expansion(df, expansion, _cross_reference())

    assert list(result["new"]) == ["4170-3210"]


def test_a_reference_entry_without_a_key_is_ignored():
    reference = {
        "cross_reference": pd.DataFrame(
            {
                "legacy": ["0417-2001", None],
                "new": ["4170-3210", "4170-9999"],
                "unit": ["PMO", "Nowhere"],
            }
        )
    }

    result = apply_expansion(_cost_centers(), _expansion(), reference)

    assert list(result["new"]) == ["4170-3210"]


def test_an_empty_frame_expands_to_an_empty_frame_with_the_carried_columns():
    df = _cost_centers().iloc[:0]

    result = apply_expansion(df, _expansion(), _cross_reference())

    assert result.empty
    assert "new" in result.columns and "unit" in result.columns


def test_a_frame_matching_nothing_expands_to_nothing():
    df = pd.DataFrame({"cost center": ["0417-0001"], "description": ["Unknown"]})

    result = apply_expansion(df, _expansion(), _cross_reference())

    assert result.empty


def test_raises_for_a_reference_table_that_is_not_available():
    with pytest.raises(ValueError, match="not available"):
        apply_expansion(_cost_centers(), _expansion(), {})


def test_raises_for_an_unknown_source_field():
    expansion = _expansion(source_field="cost centre")

    with pytest.raises(ValueError, match="unknown source field"):
        apply_expansion(_cost_centers(), expansion, _cross_reference())


def test_raises_for_an_unknown_key_field():
    expansion = _expansion(key_field="old")

    with pytest.raises(ValueError, match="has no column 'old'"):
        apply_expansion(_cost_centers(), expansion, _cross_reference())


def test_raises_for_an_unknown_carry_field():
    expansion = _expansion(carry_fields=["new", "responsible"])

    with pytest.raises(ValueError, match="has no column 'responsible'"):
        apply_expansion(_cost_centers(), expansion, _cross_reference())


def test_raises_when_a_carried_column_would_replace_a_source_column():
    reference = _cross_reference()
    reference["cross_reference"]["description"] = ["a", "b", "c"]
    expansion = _expansion(carry_fields=["new", "description"])

    with pytest.raises(ValueError, match="would replace source data"):
        apply_expansion(_cost_centers(), expansion, reference)


def test_raises_when_source_and_key_fields_do_not_pair_up():
    expansion = _expansion(
        source_field=["cost center", "plant"],
        key_field="legacy",
        carry_fields=["new"],
    )

    with pytest.raises(ValueError, match="have to match up"):
        apply_expansion(_cost_centers(), expansion, _cross_reference())
