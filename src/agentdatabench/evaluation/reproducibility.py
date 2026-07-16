"""ReproducibilityCheck: verifies that an agent's SOLUTION_SCRIPT_FILENAME,
executed fresh against dataset.csv, reproduces the exact same OUTPUT_FILENAME
the agent itself produced. A mismatch is evidence that the agent's own
documentation of "how I got this result" doesn't actually reproduce it - a
signal of hallucination or context loss, not a measure of task correctness
(that's what the 7 accuracy metrics in metrics.py are for, scored against
ground_truth.csv).

Kept as its own component rather than a Metric: it needs the agent's own
prior output as its reference (not ground_truth.csv) and re-executes code in
the agent's workspace - neither fits Metric.compute's pure
(output_df, ground_truth_df, package) signature, which assumes no filesystem/
execution side effects.

The agent's generated script hard-codes the exact absolute output path it
was given in the prompt (workspace / OUTPUT_FILENAME), so it cannot simply be
re-run into a different location - it always overwrites that same path.
`check()` therefore moves the agent's original output aside before
re-running the script, and restores it afterwards (success or failure) so
the workspace always ends up with the agent's original OUTPUT_FILENAME
intact, plus a `solution_reproduced.csv` left alongside it for inspection.
"""

from __future__ import annotations

import asyncio
import shutil
from collections import Counter
from pathlib import Path

import pandas as pd

from agentdatabench.domain.agent_run_result import AgentRunResult
from agentdatabench.domain.dataset import Dataset
from agentdatabench.domain.evaluation_result import MetricResult
from agentdatabench.evaluation.agent_adapter import OUTPUT_FILENAME, SOLUTION_SCRIPT_FILENAME

_REPRODUCED_FILENAME = "solution_reproduced.csv"
_BACKUP_FILENAME = "solution_original.csv"
_LOG_FILENAME = "reproducibility_run.log"


def _row_multiset_score(
    reference_df: pd.DataFrame, reproduced_df: pd.DataFrame
) -> tuple[float, dict]:
    """Same multiset-of-rows comparison as RowAccuracyMetric, just against
    the agent's own prior output instead of ground_truth.csv."""
    if list(reference_df.columns) != list(reproduced_df.columns):
        return 0.0, {
            "reason": "column mismatch between original and reproduced output",
            "original_columns": list(reference_df.columns),
            "reproduced_columns": list(reproduced_df.columns),
        }

    reference_rows = Counter(reference_df.astype(str).itertuples(index=False, name=None))
    reproduced_rows = Counter(reproduced_df.astype(str).itertuples(index=False, name=None))

    total = sum(reference_rows.values())
    matched = sum((reference_rows & reproduced_rows).values())
    score = matched / total if total else 1.0

    return score, {"matched_rows": matched, "total_rows": total}


class ReproducibilityCheck:
    name = "reproducibility"

    async def check(
        self,
        agent_result: AgentRunResult,
        execution_python: Path,
        *,
        timeout: float | None = None,
    ) -> MetricResult:
        workspace = agent_result.workspace
        script_path = workspace / SOLUTION_SCRIPT_FILENAME
        if not script_path.is_file():
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"reason": f"agent did not produce {SOLUTION_SCRIPT_FILENAME}"},
            )

        original_output = agent_result.output_dataset_path
        backup_path = workspace / _BACKUP_FILENAME
        shutil.move(original_output, backup_path)

        try:
            returncode, log_text = await self._execute(
                execution_python, script_path, workspace, timeout
            )
            reproduced_path = workspace / OUTPUT_FILENAME

            if returncode != 0:
                return MetricResult(
                    name=self.name,
                    score=0.0,
                    details={
                        "reason": f"{SOLUTION_SCRIPT_FILENAME} exited with code {returncode}",
                        "log": log_text[-2000:],
                    },
                )
            if not reproduced_path.is_file():
                return MetricResult(
                    name=self.name,
                    score=0.0,
                    details={
                        "reason": (
                            f"{SOLUTION_SCRIPT_FILENAME} ran but did not produce "
                            f"{OUTPUT_FILENAME}"
                        )
                    },
                )

            reproduced_final_path = workspace / _REPRODUCED_FILENAME
            shutil.move(reproduced_path, reproduced_final_path)

            reference_df = Dataset(backup_path).df
            reproduced_df = Dataset(reproduced_final_path).df
            score, details = _row_multiset_score(reference_df, reproduced_df)
            return MetricResult(name=self.name, score=score, details=details)
        except TimeoutError:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={
                    "reason": f"{SOLUTION_SCRIPT_FILENAME} did not finish within {timeout}s"
                },
            )
        except Exception as exc:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"reason": f"{type(exc).__name__}: {exc}"},
            )
        finally:
            shutil.move(backup_path, original_output)

    async def _execute(
        self,
        execution_python: Path,
        script_path: Path,
        workspace: Path,
        timeout: float | None,
    ) -> tuple[int, str]:
        process = await asyncio.create_subprocess_exec(
            str(execution_python),
            str(script_path),
            cwd=str(workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError from None

        log_text = stdout.decode(errors="replace")
        (workspace / _LOG_FILENAME).write_text(log_text)
        return process.returncode, log_text
