"""Optional step 4/4 of the manual Data Interpreter smoke test - run with
the MAIN venv (.venv/bin/python, agentdatabench installed).

Aggregates every evaluation_result.json written so far (by score.py, across
all past runs) into one EvaluationReport and prints/saves a summary. Useful
after accumulating several runs across different packages to see the
overall pass rate and average metric scores at a glance.

Usage:
    .venv/bin/python scripts/di_smoke_test/report.py
"""

from pathlib import Path

from agentdatabench.evaluation.report_generator import (
    ReportGenerator,
    load_evaluation_results,
    render_markdown,
)

RUNS_DIR = Path("venv-di/metagpt-runtime/manual_runs")

results = load_evaluation_results(RUNS_DIR)
if not results:
    raise SystemExit(f"No evaluation_result.json files found under {RUNS_DIR}")

report = ReportGenerator().generate(results)

report_json_path = RUNS_DIR / "report.json"
report_md_path = RUNS_DIR / "report.md"
report_json_path.write_text(report.model_dump_json(indent=2))
report_md_path.write_text(render_markdown(report))

print(render_markdown(report))
print(f"Wrote {report_json_path} and {report_md_path}")
