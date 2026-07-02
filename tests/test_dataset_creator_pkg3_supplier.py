"""Privacy-focused end-to-end test for package 003 (supplier migration).

Runs the real (gitignored, local-only) source_data.csv through DatasetCreator
and asserts no real value survives into the synthetic output. Skips if the
raw file isn't present locally (it's deliberately never committed).
"""

import pytest

from agentdatabench.domain.common import load_yaml
from agentdatabench.domain.dataset import Dataset
from agentdatabench.domain.synthesis_configuration import SynthesisConfiguration
from agentdatabench.generator.dataset_creator import DatasetCreator


def test_dataset_creator_does_not_leak_real_values_on_pkg3(pkg3_root):
    source_path = pkg3_root / "source_data" / "source_data.csv"
    if not source_path.is_file():
        pytest.skip("real source_data.csv not present locally (gitignored)")

    config = SynthesisConfiguration(**load_yaml(pkg3_root / "synthesis_configuration.yaml"))
    source_df = Dataset(source_path).df

    result = DatasetCreator().create_clean_dataset(source_df, config)

    for column in ["name 1", "name 2", "street"]:
        assert set(result[column]).isdisjoint(set(source_df[column])), column

    assert set(result["supplier no"]).isdisjoint(set(source_df["supplier no"]))
    assert result["supplier no"].nunique() == len(result)

    from datetime import datetime

    real_dates = [datetime.strptime(v, "%d.%m.%Y") for v in source_df["last_activity"]]
    low, high = min(real_dates), max(real_dates)
    for value in result["last_activity"]:
        d = datetime.strptime(value, "%d.%m.%Y")
        assert low <= d <= high
