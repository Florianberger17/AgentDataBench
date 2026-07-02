import pandas as pd
import pytest

from agentdatabench.domain.synthesis_configuration import SynthesisConfiguration
from agentdatabench.generator.dataset_creator import DatasetCreator

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
