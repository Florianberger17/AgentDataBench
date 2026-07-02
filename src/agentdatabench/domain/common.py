"""Shared building blocks for the agentdatabench domain model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict


class StrictBaseModel(BaseModel):
    """Base model for all domain objects parsed from benchmark package YAML files.

    Defaults to ``extra="forbid"`` so that unexpected fields in a YAML file are
    caught immediately by validation instead of silently being dropped.
    """

    model_config = ConfigDict(extra="forbid")


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a plain dict.

    An empty file (or a file containing only ``null``) yields an empty dict,
    since several fixture files contain trailing blank keys (e.g. ``seed:``).
    """
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}
