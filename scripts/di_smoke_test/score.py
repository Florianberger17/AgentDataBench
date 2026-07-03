"""Step 3/3 of the manual Data Interpreter smoke test - run with the MAIN
venv (.venv/bin/python, agentdatabench installed).

Scores workspace/solution.csv (written by run.py) against the package's
ground_truth.csv using the real Metric implementations, prints the result,
and writes it as evaluation_result.json next to solution.csv.

Usage:
    .venv/bin/python scripts/di_smoke_test/score.py 003_supplier_migration <workspace_path>
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

from agentdatabench.domain.benchmark_package import BenchmarkPackage
from agentdatabench.domain.dataset import Dataset
from agentdatabench.domain.evaluation_result import EvaluationResult
from agentdatabench.evaluation.metrics import DEFAULT_METRICS

if len(sys.argv) != 3:
    raise SystemExit("Usage: score.py <package_dir_name> <workspace_path (printed by prepare.py)>")

package_dir, workspace = sys.argv[1], Path(sys.argv[2])
solution_path = workspace / "solution.csv"
if not solution_path.is_file():
    raise SystemExit(f"{solution_path} does not exist - did run.py finish successfully?")

package = BenchmarkPackage.load(Path("artifacts/benchmark_package") / package_dir)
output_df = Dataset(solution_path).df
ground_truth_df = package.ground_truth.df

metric_results = [
    metric.compute(output_df, ground_truth_df, package) for metric in DEFAULT_METRICS
]
passed = all(m.score == 1.0 for m in metric_results)

result = EvaluationResult(
    task_id=package.task.task_id,
    agent_name="data-interpreter",
    metrics=metric_results,
    passed=passed,
    timestamp=datetime.now(timezone.utc),
)

for metric_result in metric_results:
    print(f"{metric_result.name}: {metric_result.score:.3f}  {metric_result.details}")
print("passed:", passed)

result_path = workspace / "evaluation_result.json"
result_path.write_text(result.model_dump_json(indent=2))
print("wrote", result_path)
