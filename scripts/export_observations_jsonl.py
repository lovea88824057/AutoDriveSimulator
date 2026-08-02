from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from worldsimflow.core.scenario import ScenarioLoader
from worldsimflow.core.transition_dataset import TransitionDatasetExporter, TransitionExportConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Export WorldSimFlow RL/world-model transitions as observations.jsonl.")
    parser.add_argument("--scenario", default="data/sample_scenario.json")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--policy", choices=["rule", "random"], default="rule")
    parser.add_argument("--compact", action="store_true", help="Only write schema_version/feature_names/state_vector for obs and next_obs.")
    parser.add_argument("--include-bev", action="store_true", help="Include BEV raster and BEV history in obs and next_obs.")
    parser.add_argument("--bev-history-length", type=int, default=4)
    parser.add_argument("--bev-width", type=int, default=32)
    parser.add_argument("--bev-height", type=int, default=32)
    parser.add_argument("--bev-meters-per-pixel", type=float, default=1.0)
    parser.add_argument("--output", default="outputs/all_results/rl_eval/observations.jsonl")
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    scenario = ScenarioLoader().load(resolve(args.scenario))
    rng = random.Random(args.seed)
    policy = random_policy(rng) if args.policy == "random" else rule_policy
    exporter = TransitionDatasetExporter(
        scenario,
        TransitionExportConfig(
            episodes=args.episodes,
            max_steps=args.steps,
            seed=args.seed,
            include_full_observation=not args.compact,
            include_bev=args.include_bev,
            bev_history_length=args.bev_history_length,
            bev_width=args.bev_width,
            bev_height=args.bev_height,
            bev_meters_per_pixel=args.bev_meters_per_pixel,
        ),
    )
    output = resolve(args.output)
    report = exporter.export(output, policy)
    report.update(
        {
            "policy": args.policy,
            "compact": bool(args.compact),
            "include_bev": bool(args.include_bev),
            "bev_history_length": args.bev_history_length if args.include_bev else 0,
            "usage": "Each JSONL line is one transition: obs_t, action_t, reward_t, obs_t+1, terminated/truncated, event_codes, optional BEV history.",
        }
    )
    report_path = resolve(args.report) if args.report else output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("observations_jsonl=ok")
    print(f"scenario_id={scenario.scenario_id}")
    print(f"episodes={args.episodes}")
    print(f"steps_per_episode={args.steps}")
    print(f"policy={args.policy}")
    print(f"transition_count={report['transition_count']}")
    print(f"observation_schema_version={report['observation_schema_version']}")
    print(f"feature_count={report['feature_count']}")
    print(f"include_bev={report['include_bev']}")
    print(f"bev_history_length={report['bev_history_length']}")
    print(f"output={output}")
    print(f"report={report_path}")


def rule_policy(obs: dict[str, Any]) -> dict[str, float]:
    speed = float(obs.get("ego_speed") or 0.0)
    front_gap = obs.get("front_gap")
    lane_l = float(obs.get("ego_lane_l") or obs.get("lane_center_offset") or 0.0)

    target_speed = 8.0
    acceleration = max(-2.5, min(2.0, (target_speed - speed) * 0.45))
    if front_gap is not None and float(front_gap) < 10.0:
        acceleration = min(acceleration, -1.5)
    steering = max(-0.35, min(0.35, -lane_l * 0.16))
    return {"acceleration": acceleration, "steering": steering}


def random_policy(rng: random.Random):
    actions = [
        {"acceleration": -2.0, "steering": 0.0},
        {"acceleration": 0.0, "steering": 0.0},
        {"acceleration": 1.5, "steering": 0.0},
        {"acceleration": 0.0, "steering": 0.25},
        {"acceleration": 0.0, "steering": -0.25},
    ]

    def choose(_obs: dict[str, Any]) -> dict[str, float]:
        return dict(rng.choice(actions))

    return choose


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
