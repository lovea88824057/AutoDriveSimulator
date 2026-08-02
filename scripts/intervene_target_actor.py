from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from worldsimflow import DeterministicFlowController, LaneKeepPolicy, ReplayBackend
from worldsimflow.core.metrics import summarize_trace
from worldsimflow.core.scenario import ScenarioLoader
from worldsimflow.core.scenario_diff import ScenarioDiff
from worldsimflow.core.scenario_generation import write_scenario_json
from worldsimflow.core.target_intervention import TargetActorInterventionEngine, TargetInterventionSpec
from worldsimflow.visualizer import render_trace_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a deterministic intervention to a specified actor_id.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--actor-id", required=True)
    parser.add_argument("--kind", choices=["hard_brake", "cut_in", "speed_change", "lateral_shift"], required=True)
    parser.add_argument("--start-step", type=int, default=20)
    parser.add_argument("--duration", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--deceleration", type=float, default=None)
    parser.add_argument("--target-lateral", type=float, default=None)
    parser.add_argument("--shift", type=float, default=None)
    parser.add_argument("--target-speed", type=float, default=None)
    parser.add_argument("--speed-delta", type=float, default=None)
    parser.add_argument("--speed-scale", type=float, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--render-html", default=None)
    parser.add_argument("--diff-output", default=None)
    parser.add_argument("--run-output", default=None)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    base_path = resolve(args.scenario)
    base = ScenarioLoader().load(base_path)
    params = build_params(args)
    spec = TargetInterventionSpec(
        kind=args.kind,
        actor_id=args.actor_id,
        start_step=args.start_step,
        duration=args.duration,
        seed=args.seed,
        params=params,
    )
    variant = TargetActorInterventionEngine().apply(base, spec)
    output = resolve(args.output) if args.output else ROOT / "data" / "generated" / "targeted" / f"{variant.scenario_id}.json"
    write_scenario_json(variant, output)

    diff = ScenarioDiff().compare(base, variant).to_dict()
    diff_output = resolve(args.diff_output) if args.diff_output else output.with_suffix(".diff.json")
    diff_output.parent.mkdir(parents=True, exist_ok=True)
    diff_output.write_text(json.dumps(diff, ensure_ascii=False, indent=2), encoding="utf-8")

    flow = DeterministicFlowController(variant, LaneKeepPolicy(), backend=ReplayBackend(variant))
    try:
        trace = flow.run(args.steps)
        metrics = summarize_trace(trace)
        html_path = None
        if args.render_html:
            html_path = render_trace_html(variant, trace, resolve(args.render_html))
        run_output = resolve(args.run_output) if args.run_output else default_run_output(output, html_path)
        run_payload = {
            "scenario_id": variant.scenario_id,
            "base_scenario_id": base.scenario_id,
            "actor_id": args.actor_id,
            "kind": args.kind,
            "steps": metrics.steps,
            "done": metrics.done,
            "events": metrics.event_counts,
            "total_reward": metrics.total_reward,
            "avg_speed": metrics.avg_speed,
            "min_front_gap": metrics.min_front_gap,
            "trace_hash": flow.final_trace_hash(),
            "html": str(html_path) if html_path else None,
            "scenario_path": str(output),
            "diff": str(diff_output),
        }
        run_output.parent.mkdir(parents=True, exist_ok=True)
        run_output.write_text(json.dumps(run_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        print("target_intervention=ok")
        print(f"base={base_path}")
        print(f"scenario_id={variant.scenario_id}")
        print(f"actor_id={args.actor_id}")
        print(f"kind={args.kind}")
        print(f"output={output}")
        print(f"diff={diff_output}")
        print(f"run_report={run_output}")
        print(f"steps={metrics.steps}")
        print(f"done={metrics.done}")
        print(f"events={metrics.event_counts}")
        print(f"trace_hash={flow.final_trace_hash()}")
        if html_path:
            print(f"visualization={html_path}")
    finally:
        flow.close()


def build_params(args: argparse.Namespace) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if args.deceleration is not None:
        params["deceleration"] = args.deceleration
    if args.target_lateral is not None:
        params["target_lateral"] = args.target_lateral
    if args.shift is not None:
        params["shift"] = args.shift
    if args.target_speed is not None:
        params["target_speed"] = args.target_speed
    if args.speed_delta is not None:
        params["speed_delta"] = args.speed_delta
    if args.speed_scale is not None:
        params["speed_scale"] = args.speed_scale
    return params


def default_run_output(output: Path, html_path: Path | None) -> Path:
    if html_path is not None:
        return Path(html_path).with_suffix(".run.json")
    return output.with_suffix(".run.json")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
