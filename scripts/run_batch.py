from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from worldsimflow.core.batch_runner import BatchRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a batch of WorldSimFlow Scenario JSON files.")
    parser.add_argument("--scenario-dir", default=str(ROOT / "data" / "generated" / "phase2"))
    parser.add_argument("--pattern", default="*.json")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--html-dir", default=str(ROOT / "outputs" / "batch"))
    parser.add_argument("--report", default=str(ROOT / "outputs" / "batch" / "batch_report.json"))
    args = parser.parse_args()

    scenario_dir = resolve(args.scenario_dir)
    paths = sorted(scenario_dir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No scenarios matched {args.pattern!r} under {scenario_dir}")

    runner = BatchRunner()
    results = runner.run_paths(paths, steps=args.steps, html_dir=resolve(args.html_dir))
    report_path = runner.write_report(results, resolve(args.report))

    print("batch_run=ok")
    print(f"scenarios={len(results)}")
    print(f"report={report_path}")
    for item in results:
        print(f"{item.scenario_id} steps={item.steps} done={item.done} events={item.events} hash={item.trace_hash[:12]}")


def resolve(path: str) -> Path:
    item = Path(path)
    return item if item.is_absolute() else (ROOT / item).resolve()


if __name__ == "__main__":
    main()