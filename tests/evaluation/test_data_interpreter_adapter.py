"""Tests for DataInterpreterAdapter. Injects a fake subprocess launcher so
these run without a real venv-di + metagpt installation (metagpt isn't
compatible with this project's own Python version - see
data_interpreter_adapter.py) while still covering the adapter's real
subprocess-invocation wiring: prompt file, argv, env, log capture, exit-code
handling and cancellation-on-timeout.
"""

import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest

from agentdatabench.domain.benchmark_package import BenchmarkPackage
from agentdatabench.evaluation.agent_adapter import OUTPUT_FILENAME
from agentdatabench.evaluation.data_interpreter_adapter import DataInterpreterAdapter


class _FakeProcess:
    def __init__(self, returncode: int = 0, stdout: bytes = b"", on_communicate=None):
        self.returncode = returncode
        self._stdout = stdout
        self._on_communicate = on_communicate
        self.killed = False
        self.waited = False

    async def communicate(self):
        if self._on_communicate:
            await self._on_communicate()
        return self._stdout, b""

    def kill(self):
        self.killed = True

    async def wait(self):
        self.waited = True
        return self.returncode


def _launcher_recording_calls(calls, process: _FakeProcess):
    async def launch(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return process

    return launch


def test_invoke_writes_prompt_file_and_launches_subprocess_with_venv_python(pkg1_root, tmp_path):
    package = BenchmarkPackage.load(pkg1_root)
    calls = []
    process = _FakeProcess(returncode=0, stdout=b"some log output")

    async def fake_invoke_success(*args, **kwargs):
        # Simulate the runner script's effect: the agent writes solution.csv.
        workspace = Path(args[2])
        df = pd.read_csv(workspace / "dataset.csv", dtype=str)
        df.to_csv(workspace / OUTPUT_FILENAME, index=False)

    async def launch(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        await fake_invoke_success(*args, **kwargs)
        return process

    adapter = DataInterpreterAdapter(
        venv_python=Path("/fake/venv-di/bin/python"),
        default_workspace_root=tmp_path,
        subprocess_launcher=launch,
    )

    result = asyncio.run(adapter.run(package))

    assert result.success, result.error
    assert len(calls) == 1
    args = calls[0]["args"]
    assert args[0] == str(Path("/fake/venv-di/bin/python").absolute())
    assert args[1].endswith("_data_interpreter_runner.py")
    workspace = Path(args[2])
    assert (workspace / "prompt.txt").is_file()
    assert package.task.objective.strip() in (workspace / "prompt.txt").read_text()
    assert (workspace / "run.log").read_bytes() == b"some log output"


def test_default_workspace_root_is_manual_runs_under_metagpt_project_root(tmp_path):
    adapter = DataInterpreterAdapter(metagpt_project_root=tmp_path / "metagpt-runtime")

    assert adapter._default_workspace_root == tmp_path / "metagpt-runtime" / "manual_runs"


def test_invoke_passes_role_kwargs_as_json(tmp_path):
    calls = []
    process = _FakeProcess(returncode=0)
    adapter = DataInterpreterAdapter(
        subprocess_launcher=_launcher_recording_calls(calls, process),
        use_reflection=True,
    )

    asyncio.run(adapter._invoke("a prompt", tmp_path))

    assert len(calls) == 1
    assert json.loads(calls[0]["args"][3]) == {"use_reflection": True}


def test_invoke_sets_metagpt_project_root_env_var(tmp_path):
    calls = []
    process = _FakeProcess(returncode=0)
    adapter = DataInterpreterAdapter(
        metagpt_project_root=Path("/fake/metagpt-runtime"),
        subprocess_launcher=_launcher_recording_calls(calls, process),
    )

    asyncio.run(adapter._invoke("a prompt", tmp_path))

    env = calls[0]["kwargs"]["env"]
    assert env["METAGPT_PROJECT_ROOT"] == str(Path("/fake/metagpt-runtime").resolve())


def test_run_fails_when_subprocess_exits_nonzero(pkg1_root, tmp_path):
    package = BenchmarkPackage.load(pkg1_root)
    process = _FakeProcess(returncode=1, stdout=b"traceback here")
    adapter = DataInterpreterAdapter(
        default_workspace_root=tmp_path,
        subprocess_launcher=_launcher_recording_calls([], process),
    )

    result = asyncio.run(adapter.run(package))

    assert not result.success
    assert "exited with code 1" in result.error
    assert (result.workspace / "run.log").read_bytes() == b"traceback here"


def test_run_kills_subprocess_on_timeout(pkg1_root, tmp_path):
    package = BenchmarkPackage.load(pkg1_root)

    async def hang_forever():
        await asyncio.sleep(10)

    process = _FakeProcess(on_communicate=hang_forever)
    adapter = DataInterpreterAdapter(
        default_workspace_root=tmp_path,
        subprocess_launcher=_launcher_recording_calls([], process),
    )

    result = asyncio.run(adapter.run(package, timeout=0.05))

    assert not result.success
    assert "did not finish" in result.error
    assert process.killed
    assert process.waited


def test_launching_a_nonexistent_venv_python_fails_cleanly(pkg1_root, tmp_path):
    package = BenchmarkPackage.load(pkg1_root)
    adapter = DataInterpreterAdapter(
        venv_python=Path("/nonexistent/venv-di/bin/python"),
        default_workspace_root=tmp_path,
    )

    result = asyncio.run(adapter.run(package))

    assert not result.success
    assert result.error is not None
