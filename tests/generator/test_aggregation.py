"""Tests for apply_aggregation: many source records collapsing into one."""

import pandas as pd
import pytest

from agentdatabench.domain.task import Aggregation
from agentdatabench.generator.aggregation import apply_aggregation


def _aggregation(**overrides):
    config = {
        "key_fields": ["customer"],
        "aggregates": [
            {
                "name": "revenue",
                "source_field": "amount",
                "function": "sum",
                "decimals": 2,
            }
        ],
    }
    config.update(overrides)
    return Aggregation(**config)


def _orders():
    return pd.DataFrame(
        {
            "order": ["1", "2", "3", "4"],
            "customer": ["B", "A", "B", "A"],
            "name": ["Beta", "Alpha", "Beta", "Alpha"],
            "amount": ["1.000,50", "7,00", "2,50", "3,00"],
        }
    )


def test_none_leaves_the_frame_untouched():
    df = _orders()
    pd.testing.assert_frame_equal(apply_aggregation(df, None), df)


def test_one_record_per_group_in_order_of_first_appearance():
    result = apply_aggregation(_orders(), _aggregation())
    assert list(result["customer"]) == ["B", "A"]
    assert len(result) == 2


def test_sum_keeps_the_decimal_convention_of_the_source_column():
    result = apply_aggregation(_orders(), _aggregation())
    assert list(result["revenue"]) == ["1003,00", "10,00"]


def test_columns_that_are_neither_key_nor_aggregate_take_the_first_record():
    result = apply_aggregation(_orders(), _aggregation())
    assert list(result["name"]) == ["Beta", "Alpha"]
    assert list(result["order"]) == ["1", "2"]


def test_count_counts_the_records_of_the_group():
    aggregation = _aggregation(
        aggregates=[{"name": "orders", "source_field": "amount", "function": "count"}]
    )
    result = apply_aggregation(_orders(), aggregation)
    assert list(result["orders"]) == ["2", "2"]


@pytest.mark.parametrize(
    ("function", "expected"),
    [("min", ["2,50", "3,00"]), ("max", ["1000,50", "7,00"])],
)
def test_min_and_max_read_the_column_as_numbers_not_as_text(function, expected):
    """"1.000,50" is the largest of B's amounts; compared as text it is the
    smallest."""
    aggregation = _aggregation(
        aggregates=[
            {
                "name": "extreme",
                "source_field": "amount",
                "function": function,
                "decimals": 2,
            }
        ]
    )
    result = apply_aggregation(_orders(), aggregation)
    assert list(result["extreme"]) == expected


def test_last_takes_the_value_of_the_group_s_last_record():
    aggregation = _aggregation(
        aggregates=[{"name": "latest", "source_field": "order", "function": "last"}]
    )
    result = apply_aggregation(_orders(), aggregation)
    assert list(result["latest"]) == ["3", "4"]


def test_composite_key_groups_on_the_combination():
    df = pd.DataFrame(
        {
            "customer": ["A", "A", "A"],
            "segment": ["EUR", "USD", "EUR"],
            "amount": ["1", "2", "4"],
        }
    )
    result = apply_aggregation(
        df, _aggregation(key_fields=["customer", "segment"])
    )
    assert list(zip(result["customer"], result["segment"], result["revenue"])) == [
        ("A", "EUR", "5.00"),
        ("A", "USD", "2.00"),
    ]


def test_unknown_key_field_raises():
    with pytest.raises(ValueError, match="key_fields"):
        apply_aggregation(_orders(), _aggregation(key_fields=["nope"]))


def test_unknown_aggregate_source_field_raises():
    aggregation = _aggregation(
        aggregates=[{"name": "revenue", "source_field": "nope", "function": "sum"}]
    )
    with pytest.raises(ValueError, match="unknown source field"):
        apply_aggregation(_orders(), aggregation)


def test_aggregate_named_after_a_key_field_raises():
    """It would overwrite the key it groups by, leaving the result without the
    column every later rule reads."""
    aggregation = _aggregation(
        aggregates=[{"name": "customer", "source_field": "amount", "function": "sum"}]
    )
    with pytest.raises(ValueError, match="overwrite the group key"):
        apply_aggregation(_orders(), aggregation)


def test_empty_frame_still_gains_the_aggregate_columns():
    empty = _orders().iloc[0:0]
    result = apply_aggregation(empty, _aggregation())
    assert result.empty
    assert "revenue" in result.columns


def _postings():
    """Two projects, each with a planned order value posted as a credit, cost
    postings, and a fixed-cost record repeating money already counted."""
    return pd.DataFrame(
        {
            "project": ["A", "A", "A", "A", "B", "B"],
            "value type": ["KAWV", "ISWV", "ISWV", "ISWF", "KAWV", "ISWF"],
            "amount": ["-1.000,00", "300,00", "250,00", "120,00", "-500,00", "90,00"],
        }
    )


def _where(value):
    return {"rules": [{"field": "value type", "operator": "==", "value": value}]}


def test_where_accumulates_only_the_records_it_admits():
    """The costs of a group rest on different records than its revenue, which
    is why filtering could not express this."""
    aggregation = _aggregation(
        key_fields=["project"],
        aggregates=[
            {
                "name": "costs",
                "source_field": "amount",
                "function": "sum",
                "decimals": 2,
                "where": _where("ISWV"),
            },
            {
                "name": "revenue",
                "source_field": "amount",
                "function": "sum",
                "decimals": 2,
                "where": _where("KAWV"),
            },
        ],
    )
    result = apply_aggregation(_postings(), aggregation)
    assert list(result["costs"]) == ["550,00", "0,00"]
    assert list(result["revenue"]) == ["-1000,00", "-500,00"]


def test_a_group_no_record_qualifies_in_sums_to_zero_not_to_a_blank():
    """Project B has no cost posting at all. Its costs are zero, not
    unknown - a blank cell would break every figure derived from them."""
    aggregation = _aggregation(
        key_fields=["project"],
        aggregates=[
            {
                "name": "costs",
                "source_field": "amount",
                "function": "sum",
                "decimals": 2,
                "where": _where("ISWV"),
            }
        ],
    )
    assert list(apply_aggregation(_postings(), aggregation)["costs"])[1] == "0,00"


def test_count_of_a_group_without_a_qualifying_record_is_zero():
    aggregation = _aggregation(
        key_fields=["project"],
        aggregates=[
            {
                "name": "postings",
                "source_field": "amount",
                "function": "count",
                "where": _where("ISWV"),
            }
        ],
    )
    assert list(apply_aggregation(_postings(), aggregation)["postings"]) == ["2", "0"]


def test_factor_flips_the_sign_a_figure_is_carried_with():
    aggregation = _aggregation(
        key_fields=["project"],
        aggregates=[
            {
                "name": "cap",
                "source_field": "amount",
                "function": "sum",
                "factor": -1,
                "decimals": 2,
                "where": _where("KAWV"),
            }
        ],
    )
    assert list(apply_aggregation(_postings(), aggregation)["cap"]) == [
        "1000,00",
        "500,00",
    ]


def test_where_may_test_a_reference_table():
    df = pd.DataFrame(
        {"project": ["A", "A"], "element": ["4711", "9999"], "amount": ["10", "90"]}
    )
    aggregation = _aggregation(
        key_fields=["project"],
        aggregates=[
            {
                "name": "relevant",
                "source_field": "amount",
                "function": "sum",
                "where": {
                    "rules": [
                        {
                            "field": "element",
                            "operator": "in",
                            "reference": "relevant_elements",
                            "key_field": "code",
                        }
                    ]
                },
            }
        ],
    )
    reference = {"relevant_elements": pd.DataFrame({"code": ["4711"]})}
    result = apply_aggregation(df, aggregation, reference)
    assert list(result["relevant"]) == ["10"]


def test_factor_on_a_textual_function_raises():
    aggregation = _aggregation(
        aggregates=[
            {
                "name": "name",
                "source_field": "name",
                "function": "first",
                "factor": -1,
            }
        ]
    )
    with pytest.raises(ValueError, match="not a number"):
        apply_aggregation(_orders(), aggregation)
