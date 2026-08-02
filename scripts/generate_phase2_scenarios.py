from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from worldsimflow import DeterministicFlowController, LaneKeepPolicy, ReplayBackend, ScenarioLoader
from worldsimflow.core.metrics import summarize_trace
from worldsimflow.core.scenario_generation import InterventionSpec, ScenarioInterventionEngine, write_scenario_json
from worldsimflow.visualizer import render_trace_html

DEFAULT_KINDS = ["hard_brake", "cut_in", "close_follow", "pedestrian_crossing"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 2 intervention scenarios and validate replay runs.")
    parser.add_argument("--scenario", default=str(ROOT / "data" / "sample_scenario.json"))
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "generated" / "phase2"))
    parser.add_argument("--html-dir", default=str(ROOT / "outputs" / "phase2"))
    parser.add_argument("--report", default=str(ROOT / "outputs" / "phase2" / "phase2_report.json"))
    parser.add_argument("--kinds", default=",".join(DEFAULT_KINDS))
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--no-html", action="store_true")
    args = parser.parse_args()

    scenario_path = resolve(args.scenario)
    output_dir = resolve(args.output_dir)
    html_dir = resolve(args.html_dir)
    report_path = resolve(args.report)

    base = ScenarioLoader().load(scenario_path)
    engine = ScenarioInterventionEngine()
    specs = [
        InterventionSpec(kind=kind, seed=args.seed + index + 1)
        for index, kind in enumerate(parse_kinds(args.kinds))
    ]
    variants = [engine.apply(base, spec) for spec in specs]

    output_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for scenario in variants:
        json_path = write_scenario_json(scenario, output_dir / f"{scenario.scenario_id}.json")
        trace, final_hash, metrics = run_replay(scenario, args.steps)
        html_path = None
        if not args.no_html:
            html_path = render_trace_html(scenario, trace, html_dir / f"{scenario.scenario_id}.html")
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "json": str(json_path),
                "html": str(html_path) if html_path else None,
                "mutation": scenario.metadata.get("mutation", {}),
                "steps": metrics.steps,
                "done": metrics.done,
                "events": metrics.event_counts,
                "total_reward": metrics.total_reward,
                "avg_speed": metrics.avg_speed,
                "min_front_gap": metrics.min_front_gap,
                "trace_hash": final_hash,
                "actor_count": len(scenario.actors),
            }
        )

    report = {
        "base_scenario": base.scenario_id,
        "base_path": str(scenario_path),
        "seed": args.seed,
        "count": len(rows),
        "variants": rows,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("phase2_generation=ok")
    print(f"base={base.scenario_id}")
    print(f"variants={len(rows)}")
    print(f"output_dir={output_dir}")
    print(f"html_dir={html_dir if not args.no_html else 'disabled'}")
    print(f"report={report_path}")
    for row in rows:
        print(
            "variant="
            f"{row['scenario_id']} steps={row['steps']} done={row['done']} "
            f"events={row['events']} hash={row['trace_hash'][:12]}"
        )


def parse_kinds(raw: str) -> list[str]:
    kinds = [item.strip() for item in raw.split(",") if item.strip()]
    return kinds or list(DEFAULT_KINDS)


def resolve(path: str) -> Path:
    item = Path(path)
    if item.is_absolute():
        return item
    return (ROOT / item).resolve()


def run_replay(scenario, steps: int | None):
    flow = DeterministicFlowController(scenario, LaneKeepPolicy(), backend=ReplayBackend(scenario))
    try:
        trace = flow.run(steps)
        metrics = summarize_trace(trace)
        final_hash = flow.final_trace_hash()
        return trace, final_hash, metrics
    finally:
        flow.close()


if __name__ == "__main__":
    main()