"""SynthesisConfiguration domain object: describes how DatasetCreator turns
real/raw company data into a synthetic, publishable CleanDataset."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from migratebench.domain.common import StrictBaseModel


class ColumnSynthesisConfig(BaseModel):
    """Generic per-column synthesis spec. Only `column`/`strategy` are
    fixed; other keys are strategy-specific extras interpreted by a
    SynthesisStrategy registry.
    """

    model_config = ConfigDict(extra="allow")

    column: str
    strategy: str


class SynthesisConfiguration(StrictBaseModel):
    seed: int
    locale: str = "de_DE"
    columns: list[ColumnSynthesisConfig]
