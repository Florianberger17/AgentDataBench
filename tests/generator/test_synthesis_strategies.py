import random

import pandas as pd
import pytest
from faker import Faker

from migratebench.domain.synthesis_configuration import ColumnSynthesisConfig
from migratebench.generator.synthesis_strategies import (
    CategoricalResampleStrategy,
    DateDistributionStrategy,
    FakerStrategy,
    NumericDistributionStrategy,
    UniqueSequenceStrategy,
)

# Fabricated placeholder data, not real company data - real data lives only
# under artifacts/benchmark_package/*/source_data/ (gitignored).
REAL_LIKE = pd.DataFrame(
    {
        "company": ["Acme Corp", "Widget GmbH", "Fabrikat AG"],
        "lead_time_days": ["10", "40", "25"],
        "unit_price": ["10.50", "20.00", "15.75"],
        "created": ["01.01.2020", "15.06.2021", "30.12.2022"],
        "status": ["active", "active", "inactive"],
    }
)


def _faker():
    faker = Faker("de_DE")
    faker.seed_instance(0)
    return faker


def test_faker_strategy_same_seed_gives_identical_output():
    config = ColumnSynthesisConfig(column="company", strategy="faker", provider="company")

    faker_a = Faker("de_DE")
    faker_a.seed_instance(42)
    result_a = FakerStrategy().synthesize(REAL_LIKE["company"], config, random.Random(42), faker_a, 5)

    faker_b = Faker("de_DE")
    faker_b.seed_instance(42)
    result_b = FakerStrategy().synthesize(REAL_LIKE["company"], config, random.Random(42), faker_b, 5)

    assert list(result_a) == list(result_b)


def test_numeric_distribution_strategy_stays_within_observed_range_int():
    config = ColumnSynthesisConfig(column="lead_time_days", strategy="numeric_distribution")
    result = NumericDistributionStrategy().synthesize(
        REAL_LIKE["lead_time_days"], config, random.Random(0), _faker(), 50
    )
    values = [int(v) for v in result]
    assert all(10 <= v <= 40 for v in values)
    assert all("." not in v for v in result)


def test_numeric_distribution_strategy_preserves_float_precision():
    config = ColumnSynthesisConfig(column="unit_price", strategy="numeric_distribution")
    result = NumericDistributionStrategy().synthesize(
        REAL_LIKE["unit_price"], config, random.Random(0), _faker(), 50
    )
    for value in result:
        assert "." in value
        assert len(value.split(".")[1]) == 2
        assert 10.50 <= float(value) <= 20.00


def test_date_distribution_strategy_stays_within_observed_span():
    config = ColumnSynthesisConfig(
        column="created", strategy="date_distribution", field_format="DD.MM.YYYY"
    )
    result = DateDistributionStrategy().synthesize(
        REAL_LIKE["created"], config, random.Random(0), _faker(), 50
    )
    from datetime import datetime

    for value in result:
        d = datetime.strptime(value, "%d.%m.%Y")
        assert datetime(2020, 1, 1) <= d <= datetime(2022, 12, 30)


def test_unique_sequence_strategy_generates_disjoint_unique_values():
    config = ColumnSynthesisConfig(
        column="company", strategy="unique_sequence", format="{:07d}", start=1000000
    )
    result = UniqueSequenceStrategy().synthesize(
        REAL_LIKE["company"], config, random.Random(0), _faker(), 5
    )
    assert len(set(result)) == 5
    assert set(result).isdisjoint(set(REAL_LIKE["company"]))
    assert list(result) == ["1000000", "1000001", "1000002", "1000003", "1000004"]


def test_categorical_resample_strategy_only_emits_real_labels():
    config = ColumnSynthesisConfig(column="status", strategy="categorical_resample")
    result = CategoricalResampleStrategy().synthesize(
        REAL_LIKE["status"], config, random.Random(0), _faker(), 20
    )
    assert set(result).issubset({"active", "inactive"})


def test_categorical_resample_strategy_raises_over_max_cardinality():
    config = ColumnSynthesisConfig(
        column="company", strategy="categorical_resample", max_cardinality=2
    )
    with pytest.raises(ValueError, match="max_cardinality"):
        CategoricalResampleStrategy().synthesize(
            REAL_LIKE["company"], config, random.Random(0), _faker(), 5
        )
