"""DatasetCreator: turns real/raw company data into a synthetic, publishable
CleanDataset with the same column structure but no original values.

Columns are synthesized in source-column order, and each strategy is handed a
SynthesisContext carrying the real frame plus the columns produced so far. A
strategy that derives its column from another one (`date_offset`) therefore
has to be configured *after* the column it references; it raises rather than
silently reading a missing column.
"""

from __future__ import annotations

import random

import pandas as pd
from faker import Faker

from agentdatabench.domain.synthesis_configuration import SynthesisConfiguration
from agentdatabench.generator.synthesis_strategies import (
    DEFAULT_SYNTHESIS_STRATEGIES,
    SynthesisContext,
    SynthesisStrategy,
)


class DatasetCreator:
    def __init__(self, strategies: dict[str, SynthesisStrategy] | None = None) -> None:
        self._strategies = strategies or DEFAULT_SYNTHESIS_STRATEGIES

    def create_clean_dataset(
        self, source_df: pd.DataFrame, config: SynthesisConfiguration
    ) -> pd.DataFrame:
        faker = Faker(config.locale)
        faker.seed_instance(config.seed)
        rng = random.Random(config.seed)
        n = len(source_df)

        configured = {c.column: c for c in config.columns}

        columns: dict[str, pd.Series] = {}
        for column_name in source_df.columns:
            column_config = configured.get(column_name)
            if column_config is None:
                raise ValueError(
                    f"No synthesis strategy configured for source column '{column_name}'"
                )

            strategy = self._strategies.get(column_config.strategy)
            if strategy is None:
                raise ValueError(
                    f"No synthesis strategy registered for '{column_config.strategy}'"
                )

            columns[column_name] = strategy.synthesize(
                source_df[column_name],
                column_config,
                rng,
                faker,
                n,
                SynthesisContext(source_df=source_df, synthesized=columns),
            )

        return pd.DataFrame(columns)
