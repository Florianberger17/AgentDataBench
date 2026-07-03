cd /home/florian/Projects/agentdatabench

.venv/bin/python scripts/di_smoke_test/prepare.py 004_supplier_migration
WORKSPACE=$(cat venv-di/metagpt-runtime/manual_runs/.last_workspace)

export METAGPT_PROJECT_ROOT="$(pwd)/venv-di/metagpt-runtime"
venv-di/bin/python scripts/di_smoke_test/run.py "$WORKSPACE"

.venv/bin/python scripts/di_smoke_test/score.py 004_supplier_migration "$WORKSPACE"
