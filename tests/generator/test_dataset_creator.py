import random

import pandas as pd
import pytest
from faker import Faker

from agentdatabench.domain.synthesis_configuration import (
    ColumnSynthesisConfig,
    SynthesisConfiguration,
)
from agentdatabench.generator.dataset_creator import DatasetCreator
from agentdatabench.generator.synthesis_strategies import CategoricalResampleStrategy

# Fabricated placeholder data, not real company data.
SOURCE_DF = pd.DataFrame(
    {
        "id": ["1", "2", "3"],
        "company": ["Acme Corp", "Widget GmbH", "Fabrikat AG"],
    }
)

CONFIG = SynthesisConfiguration(
    seed=1,
    columns=[
        {"column": "id", "strategy": "unique_sequence", "format": "{:05d}", "start": 90000},
        {"column": "company", "strategy": "faker", "provider": "company"},
    ],
)


def test_create_clean_dataset_same_seed_gives_identical_output():
    result_a = DatasetCreator().create_clean_dataset(SOURCE_DF, CONFIG)
    result_b = DatasetCreator().create_clean_dataset(SOURCE_DF, CONFIG)
    pd.testing.assert_frame_equal(result_a, result_b)


def test_create_clean_dataset_preserves_column_names_and_order():
    result = DatasetCreator().create_clean_dataset(SOURCE_DF, CONFIG)
    assert list(result.columns) == ["id", "company"]
    assert len(result) == 3


def test_create_clean_dataset_raises_on_unconfigured_column():
    config = SynthesisConfiguration(
        seed=1, columns=[{"column": "id", "strategy": "unique_sequence", "format": "{:05d}"}]
    )
    with pytest.raises(ValueError, match="company"):
        DatasetCreator().create_clean_dataset(SOURCE_DF, config)


def test_create_clean_dataset_raises_on_unregistered_strategy():
    config = SynthesisConfiguration(
        seed=1,
        columns=[
            {"column": "id", "strategy": "does_not_exist"},
            {"column": "company", "strategy": "faker", "provider": "company"},
        ],
    )
    with pytest.raises(ValueError, match="does_not_exist"):
        DatasetCreator().create_clean_dataset(SOURCE_DF, config)


# Fabricated placeholder data, not real company data.
DATE_SOURCE_DF = pd.DataFrame(
    {
        "OrderDate": ["01.02.24", "05.03.24", "10.04.24", "20.05.24"],
        # Real lead times of 30, 20 and 40 days - plus one contradictory pair
        # (delivery 10 days *before* its order date) that must not be resampled.
        "DeliveryDate": ["02.03.24", "25.03.24", "20.05.24", "10.05.24"],
    }
)

DATE_CONFIG = SynthesisConfiguration(
    seed=7,
    columns=[
        {
            "column": "OrderDate",
            "strategy": "date_distribution",
            "field_format": "DD.MM.YY",
        },
        {
            "column": "DeliveryDate",
            "strategy": "date_offset",
            "reference_column": "OrderDate",
            "field_format": "DD.MM.YY",
            "min_offset_days": 0,
        },
    ],
)


def _as_dates(series):
    return pd.to_datetime(series, format="%d.%m.%y")


def test_date_offset_never_produces_a_date_before_its_reference():
    result = DatasetCreator().create_clean_dataset(DATE_SOURCE_DF, DATE_CONFIG)

    assert (_as_dates(result["DeliveryDate"]) >= _as_dates(result["OrderDate"])).all()


def test_date_offset_resamples_only_real_non_contradictory_offsets():
    result = DatasetCreator().create_clean_dataset(DATE_SOURCE_DF, DATE_CONFIG)

    offsets = (_as_dates(result["DeliveryDate"]) - _as_dates(result["OrderDate"])).dt.days
    assert set(offsets) <= {30, 20, 40}


def test_date_offset_is_reproducible_for_the_same_seed():
    first = DatasetCreator().create_clean_dataset(DATE_SOURCE_DF, DATE_CONFIG)
    second = DatasetCreator().create_clean_dataset(DATE_SOURCE_DF, DATE_CONFIG)

    assert first.equals(second)


def test_date_offset_honours_a_minimum_offset():
    config = SynthesisConfiguration(
        seed=7,
        columns=[
            {"column": "OrderDate", "strategy": "date_distribution", "field_format": "DD.MM.YY"},
            {
                "column": "DeliveryDate",
                "strategy": "date_offset",
                "reference_column": "OrderDate",
                "field_format": "DD.MM.YY",
                "min_offset_days": 25,
            },
        ],
    )
    result = DatasetCreator().create_clean_dataset(DATE_SOURCE_DF, config)

    offsets = (_as_dates(result["DeliveryDate"]) - _as_dates(result["OrderDate"])).dt.days
    assert set(offsets) <= {30, 40}


def test_date_offset_raises_when_reference_column_comes_later():
    """The reference has to be synthesized first, so configuring it after the
    dependent column is an authoring error, not a silent fallback."""
    config = SynthesisConfiguration(
        seed=7,
        columns=[
            {
                "column": "DeliveryDate",
                "strategy": "date_offset",
                "reference_column": "OrderDate",
                "field_format": "DD.MM.YY",
            },
            {"column": "OrderDate", "strategy": "date_distribution", "field_format": "DD.MM.YY"},
        ],
    )
    reversed_source = DATE_SOURCE_DF[["DeliveryDate", "OrderDate"]]

    with pytest.raises(ValueError, match="has not been synthesized yet"):
        DatasetCreator().create_clean_dataset(reversed_source, config)


def test_date_offset_raises_when_no_valid_real_offset_exists():
    contradictory = pd.DataFrame(
        {"OrderDate": ["10.05.24"], "DeliveryDate": ["01.05.24"]}
    )
    with pytest.raises(ValueError, match="No real offset"):
        DatasetCreator().create_clean_dataset(contradictory, DATE_CONFIG)


# Fabricated placeholder data, not real company data.
NAME_SOURCE_DF = pd.DataFrame({"Name": ["Anna Real"] * 5 + ["Bert Real"] * 3 + ["Cora Real"] * 2})

NAME_CONFIG = SynthesisConfiguration(
    seed=3,
    columns=[{"column": "Name", "strategy": "faker_categorical", "provider": "name"}],
)


def test_faker_categorical_lets_no_real_label_survive():
    result = DatasetCreator().create_clean_dataset(NAME_SOURCE_DF, NAME_CONFIG)

    assert not set(result["Name"]) & set(NAME_SOURCE_DF["Name"])


def test_faker_categorical_preserves_cardinality_exactly():
    result = DatasetCreator().create_clean_dataset(NAME_SOURCE_DF, NAME_CONFIG)

    assert result["Name"].nunique() == NAME_SOURCE_DF["Name"].nunique()


def test_faker_categorical_follows_the_real_frequency_distribution():
    """Resampling reproduces the real proportions in distribution, not
    row-for-row, so this needs a sample large enough to be stable."""
    source = pd.DataFrame({"Name": ["Anna Real"] * 600 + ["Bert Real"] * 300 + ["Cora Real"] * 100})
    result = DatasetCreator().create_clean_dataset(source, NAME_CONFIG)

    proportions = sorted(result["Name"].value_counts(normalize=True).tolist(), reverse=True)
    assert proportions == pytest.approx([0.6, 0.3, 0.1], abs=0.05)


def test_faker_categorical_is_reproducible_for_the_same_seed():
    first = DatasetCreator().create_clean_dataset(NAME_SOURCE_DF, NAME_CONFIG)
    second = DatasetCreator().create_clean_dataset(NAME_SOURCE_DF, NAME_CONFIG)

    assert first.equals(second)


def test_faker_categorical_rejects_high_cardinality_columns():
    source = pd.DataFrame({"Name": [f"Person {i}" for i in range(30)]})
    config = SynthesisConfiguration(
        seed=3,
        columns=[
            {
                "column": "Name",
                "strategy": "faker_categorical",
                "provider": "name",
                "max_cardinality": 5,
            }
        ],
    )
    with pytest.raises(ValueError, match="max_cardinality"):
        DatasetCreator().create_clean_dataset(source, config)


# Fabricated placeholder data, not real company data.
REPEATED_SOURCE_DF = pd.DataFrame(
    {
        "part": ["A-1", "A-1", "B-2", "A-1", "C-3", "B-2"],
        "description": ["Halter", "Halter", "Platte", "Halter", "Welle", "Platte"],
    }
)

MAPPING_CONFIG = SynthesisConfiguration(
    seed=11,
    columns=[
        {"column": "part", "strategy": "identity"},
        {"column": "description", "strategy": "part_name_mapping"},
    ],
)


def test_part_name_mapping_replaces_equal_values_with_equal_names():
    result = DatasetCreator().create_clean_dataset(REPEATED_SOURCE_DF, MAPPING_CONFIG)

    names = result["description"]
    assert names.iloc[0] == names.iloc[1] == names.iloc[3]  # the three "Halter" rows
    assert names.iloc[2] == names.iloc[5]  # the two "Platte" rows


def test_part_name_mapping_preserves_the_repetition_profile():
    result = DatasetCreator().create_clean_dataset(REPEATED_SOURCE_DF, MAPPING_CONFIG)

    real = REPEATED_SOURCE_DF["description"].value_counts().tolist()
    synthetic = result["description"].value_counts().tolist()
    assert synthetic == real


def test_part_name_mapping_keeps_the_pairing_with_another_column():
    """A part number must not end up with two different descriptions."""
    result = DatasetCreator().create_clean_dataset(REPEATED_SOURCE_DF, MAPPING_CONFIG)

    assert result.groupby("part")["description"].nunique().max() == 1


def test_part_name_mapping_lets_no_real_value_survive():
    result = DatasetCreator().create_clean_dataset(REPEATED_SOURCE_DF, MAPPING_CONFIG)

    assert not set(result["description"]) & set(REPEATED_SOURCE_DF["description"])


def test_part_name_mapping_never_collapses_two_values_onto_one_name():
    result = DatasetCreator().create_clean_dataset(REPEATED_SOURCE_DF, MAPPING_CONFIG)

    pairs = dict(zip(REPEATED_SOURCE_DF["description"], result["description"]))
    assert len(set(pairs.values())) == REPEATED_SOURCE_DF["description"].nunique()


def test_part_name_mapping_leaves_empty_values_empty():
    source = pd.DataFrame({"part": ["A-1", "B-2"], "description": ["Halter", None]})
    result = DatasetCreator().create_clean_dataset(source, MAPPING_CONFIG)

    assert pd.isna(result["description"].iloc[1])
    assert result["description"].iloc[0] != "Halter"


def test_part_name_mapping_is_reproducible_for_the_same_seed():
    first = DatasetCreator().create_clean_dataset(REPEATED_SOURCE_DF, MAPPING_CONFIG)
    second = DatasetCreator().create_clean_dataset(REPEATED_SOURCE_DF, MAPPING_CONFIG)

    assert first.equals(second)


def test_part_name_mapping_raises_when_the_library_is_too_small(tmp_path):
    library = tmp_path / "tiny_library.csv"
    library.write_text("part name\nonly-one-entry\n", encoding="utf-8")
    config = SynthesisConfiguration(
        seed=11,
        columns=[
            {"column": "part", "strategy": "identity"},
            {
                "column": "description",
                "strategy": "part_name_mapping",
                "library_path": str(library),
            },
        ],
    )
    with pytest.raises(ValueError, match="only provides 1 distinct entries"):
        DatasetCreator().create_clean_dataset(REPEATED_SOURCE_DF, config)


# Fabricated placeholder data, not real company data.
REPEATED_MATERIAL_SOURCE_DF = pd.DataFrame(
    {
        "MaterialNo": ["3001-6994", "3001-6994", "3001-7000", "3001-6994", "3001-7010"],
    }
)

NUMERIC_MAPPING_CONFIG = SynthesisConfiguration(
    seed=13,
    columns=[
        {
            "column": "MaterialNo",
            "strategy": "numeric_mapping",
            "format": "00-{:06d}",
            "min": 0,
            "max": 999999,
        },
    ],
)


def test_numeric_mapping_replaces_equal_values_with_equal_numbers():
    result = DatasetCreator().create_clean_dataset(
        REPEATED_MATERIAL_SOURCE_DF, NUMERIC_MAPPING_CONFIG
    )

    numbers = result["MaterialNo"]
    assert numbers.iloc[0] == numbers.iloc[1] == numbers.iloc[3]  # the three "3001-6994" rows
    assert numbers.iloc[2] != numbers.iloc[4]


def test_numeric_mapping_lets_no_real_value_survive():
    result = DatasetCreator().create_clean_dataset(
        REPEATED_MATERIAL_SOURCE_DF, NUMERIC_MAPPING_CONFIG
    )

    assert not set(result["MaterialNo"]) & set(REPEATED_MATERIAL_SOURCE_DF["MaterialNo"])


def test_numeric_mapping_never_collapses_two_values_onto_one_number():
    result = DatasetCreator().create_clean_dataset(
        REPEATED_MATERIAL_SOURCE_DF, NUMERIC_MAPPING_CONFIG
    )

    pairs = dict(zip(REPEATED_MATERIAL_SOURCE_DF["MaterialNo"], result["MaterialNo"]))
    assert len(set(pairs.values())) == REPEATED_MATERIAL_SOURCE_DF["MaterialNo"].nunique()


def test_numeric_mapping_matches_the_configured_format():
    result = DatasetCreator().create_clean_dataset(
        REPEATED_MATERIAL_SOURCE_DF, NUMERIC_MAPPING_CONFIG
    )

    assert result["MaterialNo"].str.match(r"^00-\d{6}$").all()


def test_numeric_mapping_leaves_empty_values_empty():
    source = pd.DataFrame({"MaterialNo": ["3001-6994", None]})
    result = DatasetCreator().create_clean_dataset(source, NUMERIC_MAPPING_CONFIG)

    assert pd.isna(result["MaterialNo"].iloc[1])
    assert result["MaterialNo"].iloc[0] != "3001-6994"


def test_numeric_mapping_is_reproducible_for_the_same_seed():
    first = DatasetCreator().create_clean_dataset(
        REPEATED_MATERIAL_SOURCE_DF, NUMERIC_MAPPING_CONFIG
    )
    second = DatasetCreator().create_clean_dataset(
        REPEATED_MATERIAL_SOURCE_DF, NUMERIC_MAPPING_CONFIG
    )

    assert first.equals(second)


def test_numeric_mapping_raises_when_the_range_is_too_small():
    config = SynthesisConfiguration(
        seed=13,
        columns=[
            {
                "column": "MaterialNo",
                "strategy": "numeric_mapping",
                "format": "00-{:06d}",
                "min": 0,
                "max": 1,
            },
        ],
    )
    with pytest.raises(ValueError, match="only provides 2 distinct numbers"):
        DatasetCreator().create_clean_dataset(REPEATED_MATERIAL_SOURCE_DF, config)


# Fabricated placeholder data, not real company data.
PAIRED_SOURCE_DF = pd.DataFrame(
    {
        "validity": ["valid from"] * 4 + ["generally valid"] * 4,
        "valid from date": ["37681", "37681", None, None, None, None, None, None],
    }
)

PAIRED_CONFIG = SynthesisConfiguration(
    seed=5,
    columns=[
        {"column": "validity", "strategy": "categorical_resample"},
        {
            "column": "valid from date",
            "strategy": "conditional_resample",
            "reference_column": "validity",
        },
    ],
)


def test_conditional_resample_never_invents_a_combination_absent_from_the_source():
    """No "generally valid" row carries a date in the source, so none may
    carry one in the synthetic data either."""
    result = DatasetCreator().create_clean_dataset(PAIRED_SOURCE_DF, PAIRED_CONFIG)

    generally_valid = result["validity"] == "generally valid"
    assert result.loc[generally_valid, "valid from date"].isna().all()


def test_conditional_resample_draws_within_the_reference_group():
    result = DatasetCreator().create_clean_dataset(PAIRED_SOURCE_DF, PAIRED_CONFIG)

    dated = result["valid from date"].notna()
    assert (result.loc[dated, "validity"] == "valid from").all()
    assert set(result.loc[dated, "valid from date"]) <= {"37681"}


def test_conditional_resample_is_reproducible_for_the_same_seed():
    first = DatasetCreator().create_clean_dataset(PAIRED_SOURCE_DF, PAIRED_CONFIG)
    second = DatasetCreator().create_clean_dataset(PAIRED_SOURCE_DF, PAIRED_CONFIG)

    assert first.equals(second)


def test_conditional_resample_raises_when_reference_column_comes_later():
    config = SynthesisConfiguration(
        seed=5,
        columns=[
            {
                "column": "valid from date",
                "strategy": "conditional_resample",
                "reference_column": "validity",
            },
            {"column": "validity", "strategy": "categorical_resample"},
        ],
    )
    reversed_source = PAIRED_SOURCE_DF[["valid from date", "validity"]]

    with pytest.raises(ValueError, match="has not been synthesized yet"):
        DatasetCreator().create_clean_dataset(reversed_source, config)


# Fabricated placeholder data, not real company data. "rare" occurs once in
# 20 rows - a weighted redraw loses it more often than not.
RARE_SOURCE_DF = pd.DataFrame({"status": ["common"] * 12 + ["other"] * 7 + ["rare"]})


def test_categorical_resample_ensure_all_values_keeps_every_real_label():
    config = SynthesisConfiguration(
        seed=2,
        columns=[
            {
                "column": "status",
                "strategy": "categorical_resample",
                "ensure_all_values": True,
            }
        ],
    )
    result = DatasetCreator().create_clean_dataset(RARE_SOURCE_DF, config)

    assert set(result["status"]) == set(RARE_SOURCE_DF["status"])
    assert len(result) == len(RARE_SOURCE_DF)


def test_categorical_resample_ensure_all_values_needs_enough_rows():
    config = ColumnSynthesisConfig(
        column="status", strategy="categorical_resample", ensure_all_values=True
    )
    with pytest.raises(ValueError, match="ensure_all_values cannot be satisfied"):
        CategoricalResampleStrategy().synthesize(
            pd.Series(["a", "b", "c"]), config, random.Random(2), Faker(), n=2
        )


def _library(tmp_path, short_count, long_names):
    """A library of `short_count` distinct short entries plus `long_names`.
    The short ones have to differ from each other - duplicates collapse and
    would make the library smaller than the test intends."""
    names = [f"short-name-{i:03d}" for i in range(short_count)] + list(long_names)
    path = tmp_path / "library.csv"
    path.write_text("part name\n" + "\n".join(names) + "\n", encoding="utf-8")
    return str(path)


def test_part_name_replacement_ensure_longer_than_draws_long_entries(tmp_path):
    """Without the guarantee a mostly-short library never reaches the target
    field's length limit, leaving a truncation rule dead."""
    library = _library(tmp_path, 20, ["x" * 45, "y" * 46])
    source = pd.DataFrame({"description": ["a"] * 10})
    config = SynthesisConfiguration(
        seed=4,
        columns=[
            {
                "column": "description",
                "strategy": "part_name_replacement",
                "library_path": library,
                "ensure_longer_than": 40,
                "ensure_count": 2,
            }
        ],
    )
    result = DatasetCreator().create_clean_dataset(source, config)

    assert int((result["description"].str.len() > 40).sum()) == 2
    assert result["description"].nunique() == len(result)


def test_part_name_replacement_raises_when_the_library_lacks_long_entries(tmp_path):
    library = _library(tmp_path, 20, [])
    source = pd.DataFrame({"description": ["a"] * 5})
    config = SynthesisConfiguration(
        seed=4,
        columns=[
            {
                "column": "description",
                "strategy": "part_name_replacement",
                "library_path": library,
                "ensure_longer_than": 40,
            }
        ],
    )
    with pytest.raises(ValueError, match="only 0 are"):
        DatasetCreator().create_clean_dataset(source, config)


def test_part_name_mapping_honours_the_same_length_guarantee(tmp_path):
    library = _library(tmp_path, 20, ["x" * 45])
    source = pd.DataFrame({"description": ["A", "A", "B", "C"]})
    config = SynthesisConfiguration(
        seed=4,
        columns=[
            {
                "column": "description",
                "strategy": "part_name_mapping",
                "library_path": library,
                "ensure_longer_than": 40,
            }
        ],
    )
    result = DatasetCreator().create_clean_dataset(source, config)

    assert int((result["description"].str.len() > 40).sum()) >= 1
    assert result["description"].iloc[0] == result["description"].iloc[1]


# Fabricated placeholder data, not real company data.
REPEATED_SUPPLIER_SOURCE_DF = pd.DataFrame(
    {
        "SupplierNo": ["2787004", "2787004", "2314005", "2787004", "2865002"],
        "SupplierName": ["Alpha AG", "Alpha AG", "Beta GmbH", "Alpha AG", "Gamma KG"],
    }
)

FAKER_MAPPING_CONFIG = SynthesisConfiguration(
    seed=17,
    columns=[
        {"column": "SupplierNo", "strategy": "identity"},
        {"column": "SupplierName", "strategy": "faker_mapping", "provider": "company"},
    ],
)


def test_faker_mapping_replaces_equal_values_with_equal_faker_values():
    result = DatasetCreator().create_clean_dataset(
        REPEATED_SUPPLIER_SOURCE_DF, FAKER_MAPPING_CONFIG
    )

    names = result["SupplierName"]
    assert names.iloc[0] == names.iloc[1] == names.iloc[3]  # the three "Alpha AG" rows
    assert names.iloc[2] != names.iloc[4]


def test_faker_mapping_lets_no_real_value_survive():
    result = DatasetCreator().create_clean_dataset(
        REPEATED_SUPPLIER_SOURCE_DF, FAKER_MAPPING_CONFIG
    )

    assert not set(result["SupplierName"]) & set(REPEATED_SUPPLIER_SOURCE_DF["SupplierName"])


def test_faker_mapping_never_collapses_two_values_onto_one():
    """Faker repeats itself when drawn naively - two distinct real values
    must still end up distinct."""
    result = DatasetCreator().create_clean_dataset(
        REPEATED_SUPPLIER_SOURCE_DF, FAKER_MAPPING_CONFIG
    )

    pairs = dict(zip(REPEATED_SUPPLIER_SOURCE_DF["SupplierName"], result["SupplierName"]))
    assert len(set(pairs.values())) == REPEATED_SUPPLIER_SOURCE_DF["SupplierName"].nunique()


def test_faker_mapping_preserves_the_repetition_profile():
    result = DatasetCreator().create_clean_dataset(
        REPEATED_SUPPLIER_SOURCE_DF, FAKER_MAPPING_CONFIG
    )

    real = REPEATED_SUPPLIER_SOURCE_DF["SupplierName"].value_counts().tolist()
    assert result["SupplierName"].value_counts().tolist() == real


def test_faker_mapping_keeps_the_pairing_with_another_column():
    """A supplier number must not end up with two different names."""
    result = DatasetCreator().create_clean_dataset(
        REPEATED_SUPPLIER_SOURCE_DF, FAKER_MAPPING_CONFIG
    )

    assert result.groupby("SupplierNo")["SupplierName"].nunique().max() == 1


def test_faker_mapping_leaves_empty_values_empty():
    source = pd.DataFrame({"SupplierNo": ["1", "2"], "SupplierName": ["Alpha AG", None]})
    result = DatasetCreator().create_clean_dataset(source, FAKER_MAPPING_CONFIG)

    assert pd.isna(result["SupplierName"].iloc[1])
    assert result["SupplierName"].iloc[0] != "Alpha AG"


def test_faker_mapping_is_reproducible_for_the_same_seed():
    first = DatasetCreator().create_clean_dataset(
        REPEATED_SUPPLIER_SOURCE_DF, FAKER_MAPPING_CONFIG
    )
    second = DatasetCreator().create_clean_dataset(
        REPEATED_SUPPLIER_SOURCE_DF, FAKER_MAPPING_CONFIG
    )

    assert first.equals(second)


def test_faker_mapping_raises_on_a_provider_too_low_cardinality():
    """`boolean` yields only True/False - it cannot cover three distinct
    real values, and the strategy says so instead of looping forever."""
    config = SynthesisConfiguration(
        seed=17,
        columns=[
            {"column": "SupplierNo", "strategy": "identity"},
            {"column": "SupplierName", "strategy": "faker_mapping", "provider": "boolean"},
        ],
    )
    with pytest.raises(ValueError, match="too low-cardinality"):
        DatasetCreator().create_clean_dataset(REPEATED_SUPPLIER_SOURCE_DF, config)


def test_constant_strategy_standardises_a_code():
    """`identity` would carry the real plant through; a benchmark package may
    want the same code in every row instead."""
    source = pd.DataFrame({"plant": ["100", "200", "100"]})
    config = SynthesisConfiguration(
        seed=1, columns=[{"column": "plant", "strategy": "constant", "value": "4444"}]
    )
    result = DatasetCreator().create_clean_dataset(source, config)

    assert list(result["plant"]) == ["4444", "4444", "4444"]


def test_constant_strategy_renders_an_unquoted_yaml_integer_as_text():
    """`value: 2026` must not turn the column numeric - every column is text."""
    source = pd.DataFrame({"fiscal year": ["2022", "2022"]})
    config = SynthesisConfiguration(
        seed=1, columns=[{"column": "fiscal year", "strategy": "constant", "value": 2026}]
    )
    result = DatasetCreator().create_clean_dataset(source, config)

    assert list(result["fiscal year"]) == ["2026", "2026"]
