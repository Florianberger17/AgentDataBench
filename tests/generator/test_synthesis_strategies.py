import random
import re

import pandas as pd
import pytest
from faker import Faker

from agentdatabench.domain.synthesis_configuration import ColumnSynthesisConfig
from agentdatabench.generator.synthesis_strategies import (
    CategoricalResampleStrategy,
    DateDistributionStrategy,
    FakerStrategy,
    IdentityStrategy,
    NumericDistributionStrategy,
    NumericMappingStrategy,
    PartNameReplacementStrategy,
    UniqueSequenceStrategy,
)

# Fabricated placeholder data, not real company data - real data lives only
# under artifacts/benchmark_package/*/source_data/ (gitignored).
REAL_LIKE = pd.DataFrame(
    {
        "company": ["Acme Corp", "Widget GmbH", "Fabrikat AG"],
        "lead_time_days": ["10", "40", "25"],
        "unit_price": ["10.50", "20.00", "15.75"],
        # German-locale export: decimal comma, mixed with plain integers.
        "open_quantity": ["4,25", "2", "42,25"],
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


def test_numeric_distribution_strategy_preserves_decimal_comma():
    config = ColumnSynthesisConfig(column="open_quantity", strategy="numeric_distribution")
    result = NumericDistributionStrategy().synthesize(
        REAL_LIKE["open_quantity"], config, random.Random(0), _faker(), 50
    )
    for value in result:
        assert "." not in value
        assert len(value.split(",")[1]) == 2
        assert 2.0 <= float(value.replace(",", ".")) <= 42.25


def test_numeric_distribution_strategy_rejects_mixed_decimal_separators():
    config = ColumnSynthesisConfig(column="mixed", strategy="numeric_distribution")
    series = pd.Series(["1,5", "2.5"])

    with pytest.raises(ValueError, match="mixes"):
        NumericDistributionStrategy().synthesize(
            series, config, random.Random(0), _faker(), 5
        )


def test_numeric_distribution_strategy_rejects_non_numeric_value():
    config = ColumnSynthesisConfig(column="priced", strategy="numeric_distribution")
    series = pd.Series(["1,50", "2,00 EUR"])

    with pytest.raises(ValueError, match="non-numeric"):
        NumericDistributionStrategy().synthesize(
            series, config, random.Random(0), _faker(), 5
        )


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


def test_identity_strategy_passes_values_through_unchanged():
    config = ColumnSynthesisConfig(column="status", strategy="identity")
    result = IdentityStrategy().synthesize(
        REAL_LIKE["status"], config, random.Random(0), _faker(), len(REAL_LIKE)
    )
    assert list(result) == list(REAL_LIKE["status"])


CRLF = chr(13) + chr(10)


def _write_library(tmp_path, names):
    library = tmp_path / "part_name_library.csv"
    # BOM + CRLF mirrors the Excel export shipped in artifacts/part_name_library.
    library.write_bytes(
        ("part name" + CRLF + CRLF.join(names) + CRLF).encode("utf-8-sig")
    )
    return library


def test_part_name_replacement_strategy_never_reuses_a_library_entry(tmp_path):
    names = [f"part {i}" for i in range(10)]
    library = _write_library(tmp_path, names)
    config = ColumnSynthesisConfig(
        column="company", strategy="part_name_replacement", library_path=str(library)
    )
    result = PartNameReplacementStrategy().synthesize(
        REAL_LIKE["company"], config, random.Random(0), _faker(), 10
    )
    assert len(result) == 10
    assert len(set(result)) == 10
    assert set(result) == set(names)
    assert set(result).isdisjoint(set(REAL_LIKE["company"]))


def test_part_name_replacement_strategy_same_seed_gives_identical_output(tmp_path):
    library = _write_library(tmp_path, [f"part {i}" for i in range(20)])
    config = ColumnSynthesisConfig(
        column="company", strategy="part_name_replacement", library_path=str(library)
    )
    result_a = PartNameReplacementStrategy().synthesize(
        REAL_LIKE["company"], config, random.Random(7), _faker(), 5
    )
    result_b = PartNameReplacementStrategy().synthesize(
        REAL_LIKE["company"], config, random.Random(7), _faker(), 5
    )
    assert list(result_a) == list(result_b)


def test_part_name_replacement_strategy_deduplicates_library_and_ignores_blanks(tmp_path):
    library = _write_library(tmp_path, ["bolt", "bolt", " nut ", "", "washer"])
    config = ColumnSynthesisConfig(
        column="company", strategy="part_name_replacement", library_path=str(library)
    )
    result = PartNameReplacementStrategy().synthesize(
        REAL_LIKE["company"], config, random.Random(0), _faker(), 3
    )
    assert sorted(result) == ["bolt", "nut", "washer"]


def test_part_name_replacement_strategy_raises_when_library_too_small(tmp_path):
    library = _write_library(tmp_path, ["bolt", "nut"])
    config = ColumnSynthesisConfig(
        column="company", strategy="part_name_replacement", library_path=str(library)
    )
    with pytest.raises(ValueError, match="only provides 2 distinct entries"):
        PartNameReplacementStrategy().synthesize(
            REAL_LIKE["company"], config, random.Random(0), _faker(), 3
        )


def test_part_name_replacement_strategy_raises_on_missing_library(tmp_path):
    config = ColumnSynthesisConfig(
        column="company",
        strategy="part_name_replacement",
        library_path=str(tmp_path / "does_not_exist.csv"),
    )
    with pytest.raises(FileNotFoundError, match="does_not_exist.csv"):
        PartNameReplacementStrategy().synthesize(
            REAL_LIKE["company"], config, random.Random(0), _faker(), 1
        )


def test_part_name_replacement_strategy_uses_shipped_library_by_default():
    config = ColumnSynthesisConfig(column="company", strategy="part_name_replacement")
    result = PartNameReplacementStrategy().synthesize(
        REAL_LIKE["company"], config, random.Random(0), _faker(), 50
    )
    assert len(result) == 50
    assert len(set(result)) == 50


def test_numeric_mapping_strategy_formats_within_range():
    series = pd.Series(["3001-6994", "3001-6994", "3001-7000"])
    config = ColumnSynthesisConfig(
        column="MaterialNo",
        strategy="numeric_mapping",
        format="00-{:06d}",
        min=0,
        max=999999,
    )
    result = NumericMappingStrategy().synthesize(
        series, config, random.Random(0), _faker(), 3
    )
    for value in result:
        assert re.fullmatch(r"00-\d{6}", value)
    assert result.iloc[0] == result.iloc[1]


def test_numeric_mapping_strategy_raises_when_range_too_small():
    series = pd.Series(["A", "B", "C"])
    config = ColumnSynthesisConfig(
        column="code", strategy="numeric_mapping", format="{:d}", min=0, max=1
    )
    with pytest.raises(ValueError, match="only provides 2 distinct numbers"):
        NumericMappingStrategy().synthesize(
            series, config, random.Random(0), _faker(), 3
        )


def test_numeric_mapping_strategy_raises_when_min_exceeds_max():
    series = pd.Series(["A"])
    config = ColumnSynthesisConfig(
        column="code", strategy="numeric_mapping", format="{:d}", min=5, max=1
    )
    with pytest.raises(ValueError, match="min .* greater than max"):
        NumericMappingStrategy().synthesize(
            series, config, random.Random(0), _faker(), 1
        )
