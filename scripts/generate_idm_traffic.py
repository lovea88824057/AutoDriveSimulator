from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from worldsimflow import DeterministicFlowController, LaneKeepPolicy, ReplayBackend, ScenarioLoader
from worldsimflow.core.metrics import summarize_trace
from worldsimflow.core.scenario_generation import write_scenario_json
from worldsimflow.core.traffic_policy import IDMConfig, TrafficPolicyRunner
from worldsimflow.visualizer import render_trace_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an IDM-lite reactive traffic scenario.")
    parser.add_argument("--scenario", default="data/sample_scenario.json")
    parser.add_argument("--output", default="data/generated/idm_lite/sample_idm_lite.json")
    parser.add_argument("--html", default="outputs/idm_lite/sample_idm_lite.html")
    parser.add_argument("--actor-id", default="idm_follower")
    parser.add_argument("--leader-id", default="ego")
    parser.add_argument("--desired-speed", type=float, default=10.0)
    parser.add_argument("--min-gap", type=float, default=4.0)
    parser.add_argument("--time-headway", type=float, default=1.2)
    parser.add_argument("--initial-gap", type=float, default=16.0)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    scenario_path = resolve(args.scenario)
    output_path = resolve(args.output)
    html_path = resolve(args.html) if args.html else None

    base = ScenarioLoader().load(scenario_path)
    config = IDMConfig(
        desired_speed=args.desired_speed,
        min_gap=args.min_gap,
        time_headway=args.time_headway,
        initial_gap=args.initial_gap,
    )
    scenario = TrafficPolicyRunner().add_idm_follower(base, actor_id=args.actor_id, leader_id=args.leader_id, config=config)
    write_scenario_json(scenario, output_path)

    flow = DeterministicFlowController(scenario, LaneKeepPolicy(), backend=ReplayBackend(scenario))
    try:
        trace = flow.run(args.steps)
        metrics = summarize_trace(trace)
        print(f"scenario_id={scenario.scenario_id}")
        print(f"output={output_path}")
        print(f"steps={metrics.steps}")
        print(f"done={metrics.done}")
        print(f"events={metrics.event_counts}")
        print(f"min_front_gap={metrics.min_front_gap}")
        print(f"trace_hash={flow.final_trace_hash()}")
        if html_path:
            render_trace_html(scenario, trace, html_path)
            print(f"visualization={html_path}")
    finally:
        flow.close()


def resolve(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


if __name__ == "__main__":
    main()
