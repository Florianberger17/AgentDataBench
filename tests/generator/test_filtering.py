import pandas as pd

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
