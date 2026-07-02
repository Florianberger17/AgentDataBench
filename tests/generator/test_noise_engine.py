import pandas as pd
import pytest

from agentdatabench.domain.noise_configuration import NoiseConfiguration
from agentdatabench.generator.noise_engine import NoiseEngine


def test_same_seed_gives_identical_output():
    df = pd.DataFrame({"a": ["hello", "world", "panda"]}, dtype=str)
    config = NoiseConfiguration(
        seed=7,
        noise_types=[
            {"type": "typo", "probability": 0.5},
            {"type": "missing_value", "probability": 0.2},
            {"type": "duplicate", "probability": 0.3},
        ],
    )
    result_a = NoiseEngine().apply_noise(df, config)
    result_b = NoiseEngine().apply_noise(df, config)
    pd.testing.assert_frame_equal(result_a, result_b)


def test_excluded_columns_survive_full_pipeline():
    df = pd.DataFrame({"key": ["k1", "k2", "k3"], "value": ["hello", "world", "panda"]}, dtype=str)
    config = NoiseConfiguration(
        seed=1,
        excluded_columns=["key"],
        noise_types=[
            {"type": "typo", "probability": 1.0},
            {"type": "missing_value", "probability": 1.0},
            {"type": "duplicate", "probability": 1.0},
        ],
    )
    result = NoiseEngine().apply_noise(df, config)
    assert set(result["key"]) == {"k1", "k2", "k3"}


def test_unregistered_noise_type_raises():
    df = pd.DataFrame({"a": ["x"]}, dtype=str)
    config = NoiseConfiguration(seed=1, noise_types=[{"type": "does_not_exist", "probability": 1.0}])
    with pytest.raises(ValueError, match="does_not_exist"):
        NoiseEngine().apply_noise(df, config)


def test_order_of_noise_types_affects_result():
    df = pd.DataFrame({"a": ["hello", "world", "panda"]}, dtype=str)

    typo_then_duplicate = NoiseConfiguration(
        seed=7,
        noise_types=[
            {"type": "typo", "probability": 1.0},
            {"type": "duplicate", "probability": 1.0},
        ],
    )
    duplicate_then_typo = NoiseConfiguration(
        seed=7,
        noise_types=[
            {"type": "duplicate", "probability": 1.0},
            {"type": "typo", "probability": 1.0},
        ],
    )

    result_a = NoiseEngine().apply_noise(df, typo_then_duplicate)
    result_b = NoiseEngine().apply_noise(df, duplicate_then_typo)

    assert not result_a.equals(result_b)
    # typo-before-duplicate: every duplicated pair is byte-identical (the
    # typo was already baked in before the row got copied).
    assert list(result_a["a"][0::2]) == list(result_a["a"][1::2])
    # duplicate-before-typo: typo runs independently on each copy afterward,
    # so the two copies of a pair generally diverge.
    assert list(result_b["a"][0::2]) != list(result_b["a"][1::2])
