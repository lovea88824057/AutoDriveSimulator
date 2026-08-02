from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from worldsimflow import DeterministicFlowController, LaneKeepPolicy, ReplayBackend
from worldsimflow.core.metrics import summarize_trace
from worldsimflow.core.scenario import ScenarioLoader
from worldsimflow.core.scenario_generation import write_scenario_json
from worldsimflow.core.traffic_manager import TrafficManagerConfig, TrafficManagerLite
from worldsimflow.visualizer import render_trace_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and run a TrafficManagerLite scenario.")
    parser.add_argument("--scenario", default="data/converted/openlog_waymo_2a1e44d405a6833f.json")
    parser.add_argument("--mode", choices=["replay", "lane_aware_idm", "hybrid"], default="hybrid")
    parser.add_argument("--actor-ids", default="", help="Comma-separated actor ids controlled by lane-aware IDM.")
    parser.add_argument("--max-reactive-actors", type=int, default=4)
    parser.add_argument("--desired-speed", type=float, default=8.0)
    parser.add_argument("--min-gap", type=float, default=4.0)
    parser.add_argument("--time-headway", type=float, default=1.3)
    parser.add_argument("--lane-snap-ratio", type=float, default=0.35)
    parser.add_argument("--output", default=None)
    parser.add_argument("--html", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--steps", type=int, default=80)
    args = parser.parse_args()

    base_path = resolve(args.scenario)
    base = ScenarioLoader().load(base_path)
    actor_ids = parse_actor_ids(args.actor_ids)
    config = TrafficManagerConfig(
        mode=args.mode,
        desired_speed=args.desired_speed,
        min_gap=args.min_gap,
        time_headway=args.time_headway,
        lane_snap_ratio=args.lane_snap_ratio,
        max_reactive_actors=args.max_reactive_actors,
        reactive_actor_ids=actor_ids,
    )
    scenario = TrafficManagerLite().build_scenario(base, config)

    output = resolve(args.output) if args.output else ROOT / "data" / "generated" / "traffic_manager" / f"{scenario.scenario_id}.json"
    html_path = resolve(args.html) if args.html else ROOT / "outputs" / "all_results" / "traffic_manager" / f"{scenario.scenario_id}.html"
    report_path = resolve(args.report) if args.report else html_path.with_suffix(".run.json")
    write_scenario_json(scenario, output)

    flow = DeterministicFlowController(scenario, LaneKeepPolicy(), backend=ReplayBackend(scenario))
    try:
        trace = flow.run(args.steps)
        metrics = summarize_trace(trace)
        render_trace_html(scenario, trace, html_path)
        report = {
            "scenario_id": scenario.scenario_id,
            "base_scenario_id": base.scenario_id,
            "scenario_path": str(output),
            "html": str(html_path),
            "traffic_manager_mode": scenario.metadata.get("traffic_manager_mode"),
            "traffic_manager": scenario.metadata.get("traffic_manager"),
            "traffic_mode": scenario.metadata.get("traffic_mode"),
            "actor_count": len(scenario.actors),
            "steps": metrics.steps,
            "done": metrics.done,
            "events": metrics.event_counts,
            "total_reward": metrics.total_reward,
            "avg_speed": metrics.avg_speed,
            "min_front_gap": metrics.min_front_gap,
            "trace_hash": flow.final_trace_hash(),
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("traffic_manager_lite=ok")
        print(f"scenario_id={scenario.scenario_id}")
        print(f"mode={scenario.metadata.get('traffic_manager_mode')}")
        print(f"output={output}")
        print(f"html={html_path}")
        print(f"report={report_path}")
        print(f"steps={metrics.steps}")
        print(f"done={metrics.done}")
        print(f"events={metrics.event_counts}")
        print(f"trace_hash={flow.final_trace_hash()}")
    finally:
        flow.close()


def parse_actor_ids(value: str) -> tuple[str, ...] | None:
    ids = tuple(item.strip() for item in value.split(",") if item.strip())
    return ids or None


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()