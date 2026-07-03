# Manual Data Interpreter smoke test

Runs one real BenchmarkPackage through MetaGPT's Data Interpreter and scores
the result. Manual/two-venv because `metagpt` requires Python `<3.12` while
`agentdatabench` requires `>=3.12`, so they cannot be installed into the same
virtualenv. `DataInterpreterAdapter` itself isn't used end-to-end here for
that reason - these scripts replicate what it does, split across the two
venvs.

## One-time setup (already done on this machine)

- `venv-di/` - a Python 3.11 venv with `metagpt==0.8.2` installed
  (`uv venv venv-di --python 3.11` + `uv pip install --python venv-di/bin/python
  --prerelease=allow metagpt==0.8.2` - the `--prerelease=allow` is required,
  metagpt pins a `.dev0` release of `semantic-kernel`).
- `~/.metagpt/config2.yaml` - your OpenAI API key + model. Note: not every
  model works - metagpt 0.8.2 sends the legacy `max_tokens` parameter, which
  newer models (e.g. `gpt-5.4-mini`) reject. `gpt-4o-mini` is confirmed working.

## Running a package

From the project root, in three steps. `WORKSPACE` is captured automatically
from `.last_workspace` (written by prepare.py) - no copy-pasting of paths
needed:

```bash
cd /home/florian/Projects/agentdatabench

# 1. Prepare workspace + prompt (main venv)
.venv/bin/python scripts/di_smoke_test/prepare.py 003_supplier_migration
WORKSPACE=$(cat venv-di/metagpt-runtime/manual_runs/.last_workspace)

# 2. Run Data Interpreter for real (venv-di) - this costs real API money
export METAGPT_PROJECT_ROOT="$(pwd)/venv-di/metagpt-runtime"
venv-di/bin/python scripts/di_smoke_test/run.py "$WORKSPACE"

# 3. Score the result against ground_truth.csv (main venv)
.venv/bin/python scripts/di_smoke_test/score.py 003_supplier_migration "$WORKSPACE"
```

Run all four lines (1, `WORKSPACE=...`, 2, 3) in the *same* terminal session
- `WORKSPACE` only stays set for as long as that shell is open. To test a
different package, just repeat all four lines with the new package name in
both step 1 and step 3.

Step 2 writes the full run log (plan, generated code, LLM cost) to
`run.log`, and step 3 writes `evaluation_result.json` - both next to
`solution.csv` inside `$WORKSPACE` (i.e.
`venv-di/metagpt-runtime/manual_runs/di_smoke_.../`). After a full run, that
folder has: `dataset.csv`, `prompt.txt`, `run.log`, `solution.csv`,
`evaluation_result.json`, `source_schema.yaml`, `target_schema.yaml`.

`<package_dir_name>` is any folder name under `artifacts/benchmark_package/`
that has a built `data/dataset.csv` (i.e. `PackageBuilder` has already run
for it) - e.g. `001_customer_migration`, `002_material_master_migration`,
`003_supplier_migration`, `004_supplier_migration`, `005_material_master_migration`.

`METAGPT_PROJECT_ROOT` in step 2 keeps metagpt's own runtime files (tool
schemas, logs) inside `venv-di/` instead of writing them into the project
root - without it, metagpt falls back to the current working directory.

## Summarizing all runs so far (optional step 4)

After accumulating a few runs (across one or several packages), aggregate
all their `evaluation_result.json` files into one report:

```bash
.venv/bin/python scripts/di_smoke_test/report.py
```

Prints a per-agent summary (pass rate, average metric scores across all
runs found) and writes `report.json` + `report.md` into
`venv-di/metagpt-runtime/manual_runs/`. Runs that failed before producing an
output (empty `metrics`, `error` set) count toward the pass rate but are
excluded from the metric averages, so one crashed run doesn't silently drag
the average score down.
