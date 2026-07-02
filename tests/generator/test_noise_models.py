import random
from collections import Counter

import pandas as pd

from migratebench.domain.noise_configuration import NoiseTypeConfig
from migratebench.generator.noise_models import (
    DuplicateNoiseModel,
    MissingValueNoiseModel,
    TypoNoiseModel,
)


def test_typo_model_preserves_length_and_character_multiset():
    df = pd.DataFrame({"a": ["hello", "world"]}, dtype=str)
    config = NoiseTypeConfig(type="typo", probability=1.0)
    result = TypoNoiseModel().apply(df, config, random.Random(0), excluded_columns=set())

    for original, mutated in zip(df["a"], result["a"]):
        assert len(mutated) == len(original)
        assert Counter(mutated) == Counter(original)


def test_typo_model_skips_excluded_columns():
    df = pd.DataFrame({"a": ["hello"], "b": ["world"]}, dtype=str)
    config = NoiseTypeConfig(type="typo", probability=1.0)
    result = TypoNoiseModel().apply(df, config, random.Random(0), excluded_columns={"a"})
    assert result["a"].iloc[0] == "hello"


def test_typo_model_skips_null_and_short_cells():
    df = pd.DataFrame({"a": [pd.NA, "x"]}, dtype=str)
    config = NoiseTypeConfig(type="typo", probability=1.0)
    result = TypoNoiseModel().apply(df, config, random.Random(0), excluded_columns=set())
    assert pd.isna(result["a"].iloc[0])
    assert result["a"].iloc[1] == "x"


def test_missing_value_model_probability_one_sets_all_na():
    df = pd.DataFrame({"a": ["x", "y"], "b": ["1", "2"]}, dtype=str)
    config = NoiseTypeConfig(type="missing_value", probability=1.0)
    result = MissingValueNoiseModel().apply(df, config, random.Random(0), excluded_columns=set())
    assert result["a"].isna().all()
    assert result["b"].isna().all()


def test_missing_value_model_probability_zero_changes_nothing():
    df = pd.DataFrame({"a": ["x", "y"]}, dtype=str)
    config = NoiseTypeConfig(type="missing_value", probability=0.0)
    result = MissingValueNoiseModel().apply(df, config, random.Random(0), excluded_columns=set())
    assert list(result["a"]) == ["x", "y"]


def test_missing_value_model_skips_excluded_columns():
    df = pd.DataFrame({"a": ["x"], "b": ["y"]}, dtype=str)
    config = NoiseTypeConfig(type="missing_value", probability=1.0)
    result = MissingValueNoiseModel().apply(df, config, random.Random(0), excluded_columns={"a"})
    assert result["a"].iloc[0] == "x"
    assert pd.isna(result["b"].iloc[0])


def test_duplicate_model_probability_one_doubles_rows():
    df = pd.DataFrame({"a": ["x", "y"]}, dtype=str)
    config = NoiseTypeConfig(type="duplicate", probability=1.0)
    result = DuplicateNoiseModel().apply(df, config, random.Random(0), excluded_columns=set())
    assert len(result) == 4
    assert list(result["a"]) == ["x", "x", "y", "y"]


def test_duplicate_model_probability_zero_leaves_row_count_unchanged():
    df = pd.DataFrame({"a": ["x", "y"]}, dtype=str)
    config = NoiseTypeConfig(type="duplicate", probability=0.0)
    result = DuplicateNoiseModel().apply(df, config, random.Random(0), excluded_columns=set())
    assert len(result) == 2
    assert list(result["a"]) == ["x", "y"]
