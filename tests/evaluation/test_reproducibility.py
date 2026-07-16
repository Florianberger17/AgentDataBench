"""Tests for ReproducibilityCheck: re-executes an agent's own solution.py
and compares the result to the agent's own solution.csv. Real subprocess
execution throughout (sys.executable running tiny, deterministic scripts) -
same approach as test_direct_llm_adapter.py; nothing here needs a real
agent/LLM.
"""

import asyncio
import sys
from pathlib import Path

import pandas as pd

from agentdatabench.domain.agent_run_result import AgentRunResult
from agentdatabench.evaluation.agent_adapter import OUTPUT_FILENAME, SOLUTION_SCRIPT_FILENAME
from agentdatabench.evaluation.reproducibility import ReproducibilityCheck


def _agent_result(workspace: Path) -> AgentRunResult:
    return AgentRunResult(
        agent_name="fake-agent",
        task_id="task-1",
        success=True,
        duration_seconds=0.0,
        workspace=workspace,
        output_dataset_path=workspace / OUTPUT_FILENAME,
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_check_scores_zero_when_script_missing(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _write_csv(workspace / OUTPUT_FILENAME, [{"a": "1"}])

    result = asyncio.run(
        ReproducibilityCheck().check(_agent_result(workspace), Path(sys.executable))
    )

    assert result.name == "reproducibility"
    assert result.score == 0.0
    assert SOLUTION_SCRIPT_FILENAME in result.details["reason"]
    assert (workspace / OUTPUT_FILENAME).is_file()


def test_check_scores_one_when_script_reproduces_identical_result(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    output_path = workspace / OUTPUT_FILENAME
    _write_csv(output_path, [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}])
    original_content = output_path.read_text()

    script = (
        "import pandas as pd\n"
        f'pd.DataFrame([{{"a": "1", "b": "x"}}, {{"a": "2", "b": "y"}}]).to_csv(r"{output_path}", index=False)\n'
    )
    (workspace / SOLUTION_SCRIPT_FILENAME).write_text(script)

    result = asyncio.run(
        ReproducibilityCheck().check(_agent_result(workspace), Path(sys.executable))
    )

    assert result.score == 1.0
    assert result.details == {"matched_rows": 2, "total_rows": 2}
    assert output_path.read_text() == original_content
    assert (workspace / "solution_reproduced.csv").is_file()


def test_check_scores_partial_when_reproduction_differs(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    output_path = workspace / OUTPUT_FILENAME
    _write_csv(output_path, [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}])

    script = (
        "import pandas as pd\n"
        f'pd.DataFrame([{{"a": "1", "b": "x"}}, {{"a": "2", "b": "DIFFERENT"}}]).to_csv(r"{output_path}", index=False)\n'
    )
    (workspace / SOLUTION_SCRIPT_FILENAME).write_text(script)

    result = asyncio.run(
        ReproducibilityCheck().check(_agent_result(workspace), Path(sys.executable))
    )

    assert result.score == 0.5
    assert result.details["matched_rows"] == 1
    assert result.details["total_rows"] == 2


def test_check_scores_zero_and_restores_output_when_script_fails(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    output_path = workspace / OUTPUT_FILENAME
    _write_csv(output_path, [{"a": "1"}])
    original_content = output_path.read_text()
    (workspace / SOLUTION_SCRIPT_FILENAME).write_text("raise RuntimeError('boom')\n")

    result = asyncio.run(
        ReproducibilityCheck().check(_agent_result(workspace), Path(sys.executable))
    )

    assert result.score == 0.0
    assert "exited with code" in result.details["reason"]
    assert "boom" in result.details["log"]
    assert output_path.read_text() == original_content


def test_check_scores_zero_when_script_does_not_produce_output(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    output_path = workspace / OUTPUT_FILENAME
    _write_csv(output_path, [{"a": "1"}])
    original_content = output_path.read_text()
    (workspace / SOLUTION_SCRIPT_FILENAME).write_text("pass\n")

    result = asyncio.run(
        ReproducibilityCheck().check(_agent_result(workspace), Path(sys.executable))
    )

    assert result.score == 0.0
    assert OUTPUT_FILENAME in result.details["reason"]
    assert output_path.read_text() == original_content


def test_check_scores_zero_on_timeout(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    output_path = workspace / OUTPUT_FILENAME
    _write_csv(output_path, [{"a": "1"}])
    original_content = output_path.read_text()
    (workspace / SOLUTION_SCRIPT_FILENAME).write_text("import time\ntime.sleep(5)\n")

    result = asyncio.run(
        ReproducibilityCheck().check(
            _agent_result(workspace), Path(sys.executable), timeout=0.2
        )
    )

    assert result.score == 0.0
    assert "did not finish within" in result.details["reason"]
    assert output_path.read_text() == original_content


def test_check_scores_zero_when_execution_python_is_invalid(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    output_path = workspace / OUTPUT_FILENAME
    _write_csv(output_path, [{"a": "1"}])
    original_content = output_path.read_text()
    (workspace / SOLUTION_SCRIPT_FILENAME).write_text("pass\n")

    result = asyncio.run(
        ReproducibilityCheck().check(
            _agent_result(workspace), Path("/nonexistent/python-interpreter")
        )
    )

    assert result.score == 0.0
    assert output_path.read_text() == original_content
