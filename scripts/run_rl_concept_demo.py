from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from worldsimflow.core.rl_env import WorldSimFlowEnv
from worldsimflow.core.scenario import ScenarioLoader


ACTIONS: list[dict[str, Any]] = [
    {"name": "brake", "action": {"acceleration": -2.0, "steering": 0.0}},
    {"name": "keep", "action": {"acceleration": 0.0, "steering": 0.0}},
    {"name": "accelerate", "action": {"acceleration": 1.5, "steering": 0.0}},
    {"name": "steer_left", "action": {"acceleration": 0.0, "steering": 0.25}},
    {"name": "steer_right", "action": {"acceleration": 0.0, "steering": -0.25}},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny Q-learning concept demo on WorldSimFlowEnv.")
    parser.add_argument("--scenario", default="data/sample_scenario.json")
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--gamma", type=float, default=0.92)
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--output", default="outputs/all_results/rl_eval/rl_concept_demo_report.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    scenario = ScenarioLoader().load(resolve(args.scenario))
    q_table: dict[str, list[float]] = defaultdict(lambda: [0.0 for _ in ACTIONS])
    episodes = []

    for episode_idx in range(args.episodes):
        env = WorldSimFlowEnv(scenario, max_steps=args.steps)
        obs, reset_info = env.reset(seed=args.seed + episode_idx)
        total_reward = 0.0
        action_counts = {item["name"]: 0 for item in ACTIONS}
        first_steps = []
        terminated = False
        truncated = False
        try:
            for step_idx in range(args.steps):
                state_key = discretize_observation(obs)
                action_index = choose_action(q_table[state_key], rng, args.epsilon)
                action_record = ACTIONS[action_index]
                next_obs, reward, terminated, truncated, info = env.step(action_record["action"])
                next_key = discretize_observation(next_obs)
                target = float(reward) + args.gamma * max(q_table[next_key]) * (0.0 if terminated else 1.0)
                old_value = q_table[state_key][action_index]
                q_table[state_key][action_index] = old_value + args.alpha * (target - old_value)
                total_reward += float(reward)
                action_counts[action_record["name"]] += 1
                if step_idx < 6:
                    first_steps.append(
                        {
                            "step": step_idx,
                            "state": state_key,
                            "action": action_record["name"],
                            "reward": round(float(reward), 6),
                            "next_state": next_key,
                            "front_gap": next_obs.get("front_gap"),
                            "ego_speed": next_obs.get("ego_speed"),
                        }
                    )
                obs = next_obs
                if terminated or truncated:
                    break
            episodes.append(
                {
                    "episode": episode_idx,
                    "steps": step_idx + 1,
                    "total_reward": round(total_reward, 6),
                    "terminated": terminated,
                    "truncated": truncated,
                    "action_counts": action_counts,
                    "first_steps": first_steps,
                    "final_trace_hash": env.final_trace_hash(),
                }
            )
        finally:
            env.close()

    report = {
        "demo": "q_learning_concept",
        "scenario_id": scenario.scenario_id,
        "episodes": args.episodes,
        "max_steps": args.steps,
        "seed": args.seed,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epsilon": args.epsilon,
        "observation_schema_version": reset_info.get("observation_schema_version"),
        "feature_count": len(reset_info.get("feature_names", [])),
        "actions": [item["name"] for item in ACTIONS],
        "state_definition": {
            "speed_bucket": "slow < 6, target 6-9, fast > 9",
            "front_gap_bucket": "danger < 8, close 8-15, open >= 15 or none",
            "lane_bucket": "left / center / right by ego_lane_l",
        },
        "worldsimflow_design_note": {
            "BaseEnv": "reset/step returns observation, reward, terminated, truncated, info",
            "StateObservation": "model-facing state should be separated from simulator internals",
            "RLlib examples": "heavy training frameworks can be added later; this demo keeps the concept dependency-free",
        },
        "episodes_detail": episodes,
        "reward_curve": [item["total_reward"] for item in episodes],
        "learned_policy": summarize_policy(q_table),
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    rewards = report["reward_curve"]
    print("rl_concept_demo=ok")
    print(f"scenario_id={scenario.scenario_id}")
    print(f"episodes={args.episodes}")
    print(f"steps_per_episode={args.steps}")
    print(f"observation_schema_version={report['observation_schema_version']}")
    print(f"feature_count={report['feature_count']}")
    print(f"first_reward={rewards[0] if rewards else 0.0}")
    print(f"last_reward={rewards[-1] if rewards else 0.0}")
    print(f"q_states={len(q_table)}")
    print(f"output={output}")


def discretize_observation(obs: dict[str, Any]) -> str:
    speed = float(obs.get("ego_speed") or 0.0)
    front_gap = obs.get("front_gap")
    lane_l = obs.get("ego_lane_l")

    if speed < 6.0:
        speed_bucket = "slow"
    elif speed <= 9.0:
        speed_bucket = "target"
    else:
        speed_bucket = "fast"

    if front_gap is None:
        gap_bucket = "open"
    else:
        gap = float(front_gap)
        if gap < 8.0:
            gap_bucket = "danger"
        elif gap < 15.0:
            gap_bucket = "close"
        else:
            gap_bucket = "open"

    if lane_l is None:
        lane_bucket = "unknown"
    else:
        l = float(lane_l)
        if l > 0.7:
            lane_bucket = "left"
        elif l < -0.7:
            lane_bucket = "right"
        else:
            lane_bucket = "center"

    return f"speed={speed_bucket}|gap={gap_bucket}|lane={lane_bucket}"


def choose_action(values: list[float], rng: random.Random, epsilon: float) -> int:
    if rng.random() < epsilon:
        return rng.randrange(len(values))
    best_value = max(values)
    best_indices = [idx for idx, value in enumerate(values) if value == best_value]
    return rng.choice(best_indices)


def summarize_policy(q_table: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
    summary = {}
    for state, values in sorted(q_table.items()):
        best_index = max(range(len(values)), key=lambda idx: values[idx])
        summary[state] = {
            "best_action": ACTIONS[best_index]["name"],
            "q_values": {ACTIONS[idx]["name"]: round(value, 6) for idx, value in enumerate(values)},
        }
    return summary


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()

