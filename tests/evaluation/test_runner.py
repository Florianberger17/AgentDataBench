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
from agentdatabench.evaluation.agent_adapter import (
    OUTPUT_FILENAME,
    SOLUTION_SCRIPT_FILENAME,
    AgentAdapter,
)
from agentdatabench.evaluation.runner import EvaluationRunner


class _PerfectFakeAdapter(AgentAdapter):
    """"Solves" the task by copying the real ground_truth.csv - proves the
    runner scores a correct solution as passed, not that any real agent
    reasoning happened. Also writes a solution.py that deterministically
    reproduces that same output, so the runner's reproducibility check has a
    genuine success case to score, not just the "script missing" case the
    other fakes below exercise."""

    def __init__(self, ground_truth_path: Path) -> None:
        super().__init__(name="perfect-fake")
        self._ground_truth_path = ground_truth_path

    async def _invoke(self, prompt: str, workspace: Path) -> None:
        output_path = workspace / OUTPUT_FILENAME
        shutil.copy(self._ground_truth_path, output_path)
        script = (
            "import shutil\n"
            f'shutil.copy(r"{self._ground_truth_path}", r"{output_path}")\n'
        )
        (workspace / SOLUTION_SCRIPT_FILENAME).write_text(script)


class _GarbageFakeAdapter(AgentAdapter):
    async def _invoke(self, prompt: str, workspace: Path) -> None:
        pd.DataFrame({"nonsense": ["a", "b"]}).to_csv(workspace / OUTPUT_FILENAME, index=False)


class _FailingFakeAdapter(AgentAdapter):
    async def _invoke(self, prompt: str, workspace: Path) -> None:
        raise RuntimeError("boom")


class _MetadataFakeAdapter(AgentAdapter):
    """Reports framework-specific run metadata (steps/tokens) alongside its
    output, exercising the metadata pass-through EvaluationRunner ->
    EvaluationResult without needing a real DI/AG2 run."""

    def __init__(self, ground_truth_path: Path) -> None:
        super().__init__(name="metadata-fake")
        self._ground_truth_path = ground_truth_path

    async def _invoke(self, prompt: str, workspace: Path) -> dict:
        shutil.copy(self._ground_truth_path, workspace / OUTPUT_FILENAME)
        return {"steps": 4, "prompt_tokens": 250}


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
    assert result.reproducibility is not None
    assert result.reproducibility.score == 1.0


def test_runner_fails_when_agent_produces_garbage(pkg1_root):
    package = BenchmarkPackage.load(pkg1_root)
    adapter = _GarbageFakeAdapter(name="garbage")

    result = asyncio.run(EvaluationRunner().run(package, adapter))

    assert not result.passed
    assert result.error is None
    scores = {m.name: m.score for m in result.metrics}
    assert scores["schema_accuracy"] == 0.0
    assert scores["row_accuracy"] == 0.0
    # No solution.py was written - a low-effort agent still gets scored, just
    # with a 0.0 reproducibility result rather than a crash.
    assert result.reproducibility is not None
    assert result.reproducibility.score == 0.0


def test_runner_records_agent_failure_without_computing_metrics(pkg1_root):
    package = BenchmarkPackage.load(pkg1_root)
    adapter = _FailingFakeAdapter(name="failing")

    result = asyncio.run(EvaluationRunner().run(package, adapter))

    assert not result.passed
    assert result.metrics == []
    assert result.reproducibility is None
    assert "boom" in result.error
    # Duration is still meaningful even for a failed run (how long it took
    # to fail); metadata is empty since _invoke never returned normally.
    assert result.duration_seconds >= 0.0
    assert result.metadata == {}


def test_runner_copies_duration_and_metadata_from_agent_result(pkg1_root):
    package = BenchmarkPackage.load(pkg1_root)
    adapter = _MetadataFakeAdapter(ground_truth_path=package.ground_truth.path)

    result = asyncio.run(EvaluationRunner().run(package, adapter))

    assert result.duration_seconds >= 0.0
    assert result.metadata == {"steps": 4, "prompt_tokens": 250}


def test_runner_passes_workspace_root_through_to_adapter(pkg1_root, tmp_path):
    package = BenchmarkPackage.load(pkg1_root)
    adapter = _PerfectFakeAdapter(ground_truth_path=package.ground_truth.path)
    workspace_root = tmp_path / "manual_runs"

    asyncio.run(EvaluationRunner().run(package, adapter, workspace_root=workspace_root))

    children = list(workspace_root.iterdir())
    assert len(children) == 1
    assert children[0].name.startswith("perfect-fake_")
