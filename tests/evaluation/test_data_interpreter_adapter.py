"""Tests for DataInterpreterAdapter. Injects a fake role factory so these run
without metagpt installed (it isn't compatible with this project's Python
version - see data_interpreter_adapter.py) while still covering the
adapter's own prompt/invocation wiring.
"""

import asyncio
from pathlib import Path

import pandas as pd

from agentdatabench.domain.benchmark_package import BenchmarkPackage
from agentdatabench.evaluation.agent_adapter import OUTPUT_FILENAME
from agentdatabench.evaluation.data_interpreter_adapter import DataInterpreterAdapter


class _FakeDataInterpreterRole:
    def __init__(self):
        self.received_requirement: str | None = None

    async def run(self, requirement: str) -> None:
        self.received_requirement = requirement
        dataset_line = next(
            line for line in requirement.splitlines() if line.startswith("Input dataset:")
        )
        workspace = Path(dataset_line.split(": ", 1)[1]).parent
        df = pd.read_csv(workspace / "dataset.csv", dtype=str)
        df.to_csv(workspace / OUTPUT_FILENAME, index=False)


def test_invoke_passes_prompt_to_role_and_collects_output(pkg1_root):
    package = BenchmarkPackage.load(pkg1_root)
    fake_role = _FakeDataInterpreterRole()
    adapter = DataInterpreterAdapter(role_factory=lambda **_: fake_role)

    result = asyncio.run(adapter.run(package))

    assert result.success, result.error
    assert result.agent_name == "data-interpreter"
    assert fake_role.received_requirement is not None
    assert package.task.objective.strip() in fake_role.received_requirement
    assert str(OUTPUT_FILENAME) in fake_role.received_requirement


def test_role_factory_receives_role_kwargs(pkg1_root):
    package = BenchmarkPackage.load(pkg1_root)
    received_kwargs = {}

    def factory(**kwargs):
        received_kwargs.update(kwargs)
        return _FakeDataInterpreterRole()

    adapter = DataInterpreterAdapter(role_factory=factory, use_reflection=True)
    asyncio.run(adapter.run(package))

    assert received_kwargs == {"use_reflection": True}


def test_default_role_factory_without_metagpt_installed_fails_cleanly(pkg1_root):
    package = BenchmarkPackage.load(pkg1_root)
    adapter = DataInterpreterAdapter()

    result = asyncio.run(adapter.run(package))

    assert not result.success
    assert "metagpt" in result.error.lower() or "modulenotfounderror" in result.error.lower()
