"""Step 2/3 of the manual Data Interpreter smoke test - run with venv-di
(Python 3.11, metagpt installed). No agentdatabench import here on purpose:
metagpt requires Python <3.12, agentdatabench requires >=3.12, so the two
can't share a venv (see scripts/di_smoke_test/README.md).

Reads workspace/prompt.txt (written by prepare.py) and runs it through
MetaGPT's DataInterpreter, which writes workspace/solution.csv itself. The
full run log (plan, generated code, LLM cost) is written to
workspace/run.log - the same folder solution.csv ends up in - regardless of
how this script is invoked, so a plain `python run.py <workspace>` (no
piping/redirection needed) is enough to get a log you can inspect afterward.

Usage (from the project root, with METAGPT_PROJECT_ROOT exported - see README):
    venv-di/bin/python scripts/di_smoke_test/run.py <workspace_path>
"""

import asyncio
import sys
from pathlib import Path

from loguru import logger

if len(sys.argv) != 2:
    raise SystemExit("Usage: run.py <workspace_path (printed by prepare.py)>")

workspace = Path(sys.argv[1]).resolve()
prompt = (workspace / "prompt.txt").read_text()
logger.add(workspace / "run.log")


async def main() -> None:
    from metagpt.roles.di.data_interpreter import DataInterpreter

    di = DataInterpreter()
    await di.run(prompt)


if __name__ == "__main__":
    asyncio.run(main())
