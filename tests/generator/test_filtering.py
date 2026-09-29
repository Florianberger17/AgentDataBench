import pandas as pd
import pytest

from agentdatabench.domain.task import FilteringRules
from agentdatabench.generator.filtering import apply_filtering


def test_apply_filtering_none_returns_df_unchanged():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert apply_filtering(df, None) is df


def test_apply_filtering_single_rule():
    df = pd.DataFrame({"amount": [1, 5, 10]})
    filtering = FilteringRules(rules=[{"field": "amount", "operator": ">=", "value": 5}])
    result = apply_filtering(df, filtering)
    assert list(result["amount"]) == [5, 10]


def test_apply_filtering_multiple_rules_are_anded():
    df = pd.DataFrame({"amount": [1, 5, 10, 20]})
    filtering = FilteringRules(
        rules=[
            {"field": "amount", "operator": ">=", "value": 5},
            {"field": "amount", "operator": "<=", "value": 10},
        ]
    )
    result = apply_filtering(df, filtering)
    assert list(result["amount"]) == [5, 10]


def test_apply_filtering_resets_index():
    df = pd.DataFrame({"amount": [1, 5, 10]})
    filtering = FilteringRules(rules=[{"field": "amount", "operator": ">=", "value": 5}])
    result = apply_filtering(df, filtering)
    assert list(result.index) == [0, 1]


def test_apply_filtering_with_non_iso_date_field_format():
    # naive string comparison against these values would give the opposite
    # result: "21.04.2022" > "2023-01-01" lexicographically, "01.07.2023" < it.
    df = pd.DataFrame({"last_activity": ["21.04.2022", "01.07.2023"]})
    filtering = FilteringRules(
        rules=[
            {
                "field": "last_activity",
                "operator": ">=",
                "value": "2023-01-01",
                "field_format": "DD.MM.YYYY",
            }
        ]
    )
    result = apply_filtering(df, filtering)
    assert list(result["last_activity"]) == ["01.07.2023"]


def test_numeric_filter_handles_a_comma_decimal_column():
    """A German-locale export writes "1234,56"; that is a number, not a parse
    error."""
    df = pd.DataFrame({"price": ["11629,42", "0,00", "5,28"]})
    rules = FilteringRules(rules=[{"field": "price", "operator": ">", "value": 0}])

    result = apply_filtering(df, rules)

    assert list(result["price"]) == ["11629,42", "5,28"]


def test_emptiness_operators_need_no_value():
    df = pd.DataFrame({"part no.": ["A", None, "  ", "B"]})

    kept = apply_filtering(df, FilteringRules(rules=[{"field": "part no.", "operator": "is not empty"}]))
    dropped = apply_filtering(df, FilteringRules(rules=[{"field": "part no.", "operator": "is empty"}]))

    assert list(kept["part no."]) == ["A", "B"]
    assert len(dropped) == 2


def test_in_operator_keeps_only_rows_present_in_a_reference_table():
    df = pd.DataFrame({"part no.": ["A", "B", "C"]})
    reference = {"materials": pd.DataFrame({"part no.": ["A", "C"]})}
    rules = FilteringRules(
        rules=[
            {
                "field": "part no.",
                "operator": "in",
                "reference": "materials",
                "key_field": "part no.",
            }
        ]
    )

    assert list(apply_filtering(df, rules, reference)["part no."]) == ["A", "C"]


def test_not_in_operator_is_the_inverse():
    df = pd.DataFrame({"part no.": ["A", "B"]})
    reference = {"materials": pd.DataFrame({"part no.": ["A"]})}
    rules = FilteringRules(
        rules=[
            {
                "field": "part no.",
                "operator": "not in",
                "reference": "materials",
                "key_field": "part no.",
            }
        ]
    )

    assert list(apply_filtering(df, rules, reference)["part no."]) == ["B"]


def test_in_operator_raises_when_reference_data_is_missing():
    df = pd.DataFrame({"part no.": ["A"]})
    rules = FilteringRules(
        rules=[
            {
                "field": "part no.",
                "operator": "in",
                "reference": "nope",
                "key_field": "part no.",
            }
        ]
    )

    with pytest.raises(ValueError, match="not available"):
        apply_filtering(df, rules)


def test_unknown_operator_is_rejected():
    df = pd.DataFrame({"a": ["1"]})
    rules = FilteringRules(rules=[{"field": "a", "operator": "~=", "value": 1}])

    with pytest.raises(ValueError, match="Unknown filter operator"):
        apply_filtering(df, rules)


def test_in_operator_supports_a_composite_key():
    """A customer is identified by customer number and address number
    together, so both have to be looked up as one key."""
    df = pd.DataFrame({"cust": ["1150011", "1150011", "9999999"], "addr": ["1", "2", "1"]})
    reference = {
        "customers": pd.DataFrame(
            {"customer no. legacy": ["1150011"], "address no.": ["1"]}
        )
    }
    rules = FilteringRules(
        rules=[
            {
                "field": ["cust", "addr"],
                "operator": "in",
                "reference": "customers",
                "key_field": ["customer no. legacy", "address no."],
            }
        ]
    )

    result = apply_filtering(df, rules, reference)

    assert result[["cust", "addr"]].values.tolist() == [["1150011", "1"]]


def test_composite_in_operator_rejects_mismatched_field_counts():
    df = pd.DataFrame({"a": ["1"], "b": ["2"]})
    reference = {"r": pd.DataFrame({"k": ["1"]})}
    rules = FilteringRules(
        rules=[{"field": ["a", "b"], "operator": "in", "reference": "r", "key_field": "k"}]
    )

    with pytest.raises(ValueError, match="have to match up"):
        apply_filtering(df, rules, reference)
