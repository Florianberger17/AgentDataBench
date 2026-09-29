"""Runs one AgentAdapter over a set of BenchmarkPackages and writes a report.

The single entry point for a benchmark run: loads every package under
`--packages-root` (or the ones named with `--package`), hands each to the
chosen adapter through EvaluationRunner, and writes both the raw
EvaluationResults (JSON, one object per run) and a rendered Markdown summary.

`--repeat` runs every package N times. Runs are kept as separate results
rather than averaged here, so the variance stays visible in the raw JSON and
can be analysed afterwards; the Markdown summary averages across all of them.

Two adapters need no API access and exist to check the harness itself
against real packages before spending money on a real agent run:
`stub` produces nothing (exercises the agent-failure path), and `perfect`
writes ground_truth.csv verbatim, which must score 1.0 on every metric -
if it does not, the package or a metric is at fault, not the agent.

Usage (from the project root):
    python scripts/run_benchmark.py --agent perfect
    python scripts/run_benchmark.py --agent data-interpreter --repeat 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from pathlib import Path

from agentdatabench.domain.benchmark_package import BenchmarkPackage
from agentdatabench.evaluation.agent_adapter import AgentAdapter
from agentdatabench.evaluation.report_generator import ReportGenerator, render_markdown
from agentdatabench.evaluation.runner import EvaluationRunner

DEFAULT_PACKAGES_ROOT = Path("artifacts/benchmark_package")
DEFAULT_OUT_DIR = Path("artifacts/evaluation_runs")
# Pre-rename leftovers that are not part of the benchmark (see the 00X_/09X_
# directories). Excluded unless --include-legacy is passed, so a plain run
# always covers exactly the numbered benchmark packages.
LEGACY_PREFIXES = ("09", "00X")


class PerfectAdapter(AgentAdapter):
    """Writes ground_truth.csv as the solution. Not an agent - a self-test of
    the harness: every metric has to score 1.0 for every package, otherwise
    the package or the metric is broken."""

    def __init__(self) -> None:
        super().__init__(name="perfect")
        self._package: BenchmarkPackage | None = None

    def _prepare_workspace(self, package: BenchmarkPackage, workspace: Path) -> None:
        super()._prepare_workspace(package, workspace)
        self._package = package

    async def _invoke(self, prompt: str, workspace: Path) -> dict | None:
        shutil.copy(self._package.ground_truth.path, workspace / "solution.csv")
        (workspace / "solution.py").write_text(
            "# harness self-test: the expected result, copied verbatim\n",
            encoding="utf-8",
        )
        return {"steps": 0}


class StubAdapter(AgentAdapter):
    """Produces no solution at all, so every run reports the agent-level
    failure path. Useful to verify a package is loadable and the prompt
    renders, without any API access."""

    def __init__(self) -> None:
        super().__init__(name="stub")

    async def _invoke(self, prompt: str, workspace: Path) -> dict | None:
        (workspace / "prompt.txt").write_text(prompt, encoding="utf-8")
        return {"steps": 0}


def build_adapter(name: str) -> AgentAdapter:
    if name == "stub":
        return StubAdapter()
    if name == "perfect":
        return PerfectAdapter()
    if name == "data-interpreter":
        from agentdatabench.evaluation.data_interpreter_adapter import DataInterpreterAdapter

        return DataInterpreterAdapter()
    if name == "ag2":
        from agentdatabench.evaluation.ag2_adapter import AG2Adapter

        return AG2Adapter()
    raise SystemExit(f"Unknown agent {name!r} (known: stub, perfect, data-interpreter, ag2)")


def discover_packages(root: Path, names: list[str] | None, include_legacy: bool) -> list[Path]:
    """The numbered benchmark packages, in order."""
    if names:
        return [root / name for name in names]
    return sorted(
        d
        for d in root.iterdir()
        if d.is_dir()
        and d.name[:3].isdigit()
        and (d / "task.yaml").is_file()
        and (include_legacy or not d.name.startswith(LEGACY_PREFIXES))
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="stub", help="stub | perfect | data-interpreter | ag2")
    parser.add_argument("--package", action="append", help="package directory name (repeatable)")
    parser.add_argument("--packages-root", type=Path, default=DEFAULT_PACKAGES_ROOT)
    parser.add_argument("--repeat", type=int, default=1, help="runs per package")
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds per run")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--include-legacy", action="store_true", help=f"also run {LEGACY_PREFIXES} packages"
    )
    args = parser.parse_args()

    adapter = build_adapter(args.agent)
    package_dirs = discover_packages(args.packages_root, args.package, args.include_legacy)
    if not package_dirs:
        raise SystemExit(f"No benchmark packages found under {args.packages_root}")

    print(f"agent: {adapter.name} | packages: {len(package_dirs)} | repeat: {args.repeat}")
    runner = EvaluationRunner()
    results = []
    for repetition in range(args.repeat):
        for package_dir in package_dirs:
            package = BenchmarkPackage.load(package_dir)
            result = await runner.run(package, adapter, timeout=args.timeout)
            results.append(result)
            scores = " ".join(f"{m.name.split('_')[0]}={m.score:.2f}" for m in result.metrics)
            status = "PASS" if result.passed else (result.error or "fail")
            print(
                f"  [{repetition + 1}/{args.repeat}] {package_dir.name[:46]:46s} "
                f"{status[:40]:40s} {scores}"
            )

    report = ReportGenerator().generate(results)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = report.generated_at.strftime("%Y%m%d_%H%M%S")
    json_path = args.out_dir / f"{adapter.name}_{stamp}.json"
    markdown_path = args.out_dir / f"{adapter.name}_{stamp}.md"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    print()
    print(render_markdown(report))
    print(f"raw results: {json_path}")
    print(f"summary    : {markdown_path}")


if __name__ == "__main__":
    asyncio.run(main())
