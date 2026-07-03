"""AgentRunResult domain object: the outcome of running one AgentAdapter over
one BenchmarkPackage. Deliberately holds a path to the produced CSV rather
than a loaded Dataset, mirroring how BenchmarkPackage itself only loads CSVs
lazily - keeps the result plain/serializable for later use in an evaluation
report.
"""

from __future__ import annotations

from pathlib import Path

from agentdatabench.domain.common import StrictBaseModel


class AgentRunResult(StrictBaseModel):
    agent_name: str
    task_id: str
    success: bool
    duration_seconds: float
    workspace: Path
    output_dataset_path: Path | None = None
    error: str | None = None
