"""Step 1/3 of the manual Data Interpreter smoke test - run with the MAIN
venv (.venv/bin/python, agentdatabench installed).

Prepares a workspace + prompt for one benchmark package exactly like
AgentAdapter.run() would, then writes the prompt to workspace/prompt.txt so
run.py (executed separately in venv-di, see README.md in this folder) can
pick it up.

Usage:
    .venv/bin/python scripts/di_smoke_test/prepare.py 003_supplier_migration
"""

import sys
from pathlib import Path
from tempfile import mkdtemp

from agentdatabench.domain.benchmark_package import BenchmarkPackage
from agentdatabench.evaluation.data_interpreter_adapter import DataInterpreterAdapter

if len(sys.argv) != 2:
    raise SystemExit("Usage: prepare.py <package_dir_name, e.g. 003_supplier_migration>")

package_dir = sys.argv[1]
PACKAGE_ROOT = Path("artifacts/benchmark_package") / package_dir
RUNS_DIR = Path("venv-di/metagpt-runtime/manual_runs")

package = BenchmarkPackage.load(PACKAGE_ROOT)
adapter = DataInterpreterAdapter()

RUNS_DIR.mkdir(parents=True, exist_ok=True)
workspace = Path(mkdtemp(prefix="di_smoke_", dir=RUNS_DIR))

adapter._prepare_workspace(package, workspace)
prompt = adapter._build_prompt(package, workspace)
(workspace / "prompt.txt").write_text(prompt)

# Written so the shell commands in README.md can pick the path back up with
# WORKSPACE=$(cat venv-di/metagpt-runtime/manual_runs/.last_workspace)
# instead of you having to copy-paste it by hand.
RUNS_DIR.joinpath(".last_workspace").write_text(str(workspace))

print("WORKSPACE:", workspace)
print()
print("--- prompt.txt ---")
print(prompt)
