"""End-to-end tests for EvaluationRunner: BenchmarkPackage + AgentAdapter ->
EvaluationResult. Uses small fake adapters instead of a real agent SDK, same
approach as test_agent_adapter.py.
"""

import asyncio
import shutil
from pathlib import Path

import pandas as pd
import pytest

from agentdatabench.domain.benchmark_package import BenchmarkPackage
from agentdatabench.evaluation.agent_adapter import OUTPUT_FILENAME, AgentAdapter
from agentdatabench.evaluation.runner import EvaluationRunner


class _PerfectFakeAdapter(AgentAdapter):
    """"Solves" the task by copying the real ground_truth.csv - proves the
    runner scores a correct solution as passed, not that any real agent
    reasoning happened."""

    def __init__(self, ground_truth_path: Path) -> None:
        super().__init__(name="perfect-fake")
        self._ground_truth_path = ground_truth_path

    async def _invoke(self, prompt: str, workspace: Path) -> None:
        shutil.copy(self._ground_truth_path, workspace / OUTPUT_FILENAME)


class _GarbageFakeAdapter(AgentAdapter):
    async def _invoke(self, prompt: str, workspace: Path) -> None:
        pd.DataFrame({"nonsense": ["a", "b"]}).to_csv(workspace / OUTPUT_FILENAME, index=False)


class _FailingFakeAdapter(AgentAdapter):
    async def _invoke(self, prompt: str, workspace: Path) -> None:
        raise RuntimeError("boom")


@pytest.mark.parametrize(
    "package_dir_fixture",
    ["pkg1_root", "pkg2_root", "pkg3_root", "pkg4_root", "pkg5_root"],
)
def test_runner_passes_when_agent_reproduces_ground_truth(package_dir_fixture, request):
    root = request.getfixturevalue(package_dir_fixture)
    if not (root / "data" / "dataset.csv").is_file():
        pytest.skip(f"{root} has no built dataset.csv yet")

    package = BenchmarkPackage.load(root)
    adapter = _PerfectFakeAdapter(ground_truth_path=package.ground_truth.path)

    result = asyncio.run(EvaluationRunner().run(package, adapter))

    assert result.passed, [m.model_dump() for m in result.metrics]
    assert result.error is None
    scores = {m.name: m.score for m in result.metrics}
    assert scores == {name: 1.0 for name in scores}, scores
    assert set(scores) == {
        "schema_accuracy",
        "row_accuracy",
        "filtering_accuracy",
        "field_mapping_accuracy",
        "transformation_accuracy",
        "record_accuracy",
        "error_correction_accuracy",
    }


def test_runner_fails_when_agent_produces_garbage(pkg1_root):
    package = BenchmarkPackage.load(pkg1_root)
    adapter = _GarbageFakeAdapter(name="garbage")

    result = asyncio.run(EvaluationRunner().run(package, adapter))

    assert not result.passed
    assert result.error is None
    scores = {m.name: m.score for m in result.metrics}
    assert scores["schema_accuracy"] == 0.0
    assert scores["row_accuracy"] == 0.0


def test_runner_records_agent_failure_without_computing_metrics(pkg1_root):
    package = BenchmarkPackage.load(pkg1_root)
    adapter = _FailingFakeAdapter(name="failing")

    result = asyncio.run(EvaluationRunner().run(package, adapter))

    assert not result.passed
    assert result.metrics == []
    assert "boom" in result.error


def test_runner_passes_workspace_root_through_to_adapter(pkg1_root, tmp_path):
    package = BenchmarkPackage.load(pkg1_root)
    adapter = _PerfectFakeAdapter(ground_truth_path=package.ground_truth.path)
    workspace_root = tmp_path / "manual_runs"

    asyncio.run(EvaluationRunner().run(package, adapter, workspace_root=workspace_root))

    children = list(workspace_root.iterdir())
    assert len(children) == 1
    assert children[0].name.startswith("perfect-fake_")
