"""NoiseEngine: deterministically injects errors into a CleanDataset to
produce the BenchmarkDataset an agent receives."""

from __future__ import annotations

import random

import pandas as pd

from agentdatabench.domain.noise_configuration import NoiseConfiguration
from agentdatabench.generator.noise_models import DEFAULT_NOISE_MODELS, NoiseModel


class NoiseEngine:
    def __init__(self, models: dict[str, NoiseModel] | None = None) -> None:
        self._models = models or DEFAULT_NOISE_MODELS

    def apply_noise(self, clean_df: pd.DataFrame, config: NoiseConfiguration) -> pd.DataFrame:
        rng = random.Random(config.seed)
        excluded_columns = set(config.excluded_columns)

        df = clean_df
        for noise_type_config in config.noise_types:
            model = self._models.get(noise_type_config.type)
            if model is None:
                raise ValueError(f"No noise model registered for type '{noise_type_config.type}'")
            df = model.apply(df, noise_type_config, rng, excluded_columns)

        return df
