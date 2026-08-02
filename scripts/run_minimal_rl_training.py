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


BASE_ACTIONS = [
    {"name": "brake", "action": {"acceleration": -2.0, "steering": 0.0}},
    {"name": "keep", "action": {"acceleration": 0.0, "steering": 0.0}},
    {"name": "accelerate", "action": {"acceleration": 1.5, "steering": 0.0}},
    {"name": "left", "action": {"acceleration": 0.0, "steering": 0.25}},
    {"name": "right", "action": {"acceleration": 0.0, "steering": -0.25}},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a minimal dependency-free RL training loop with normalized_vector + SpaceSpec.")
    parser.add_argument("--scenario", default="data/sample_scenario.json")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--epsilon", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--gamma", type=float, default=0.90)
    parser.add_argument("--output", default="outputs/all_results/rl_eval/minimal_rl_training_report.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    scenario = ScenarioLoader().load(resolve(args.scenario))
    q_table: dict[str, list[float]] = defaultdict(lambda: [0.0 for _ in BASE_ACTIONS])
    episode_reports = []
    reset_info: dict[str, Any] = {}
    all_actions_valid = True
    all_normalized_valid = True

    for episode in range(args.episodes):
        env = WorldSimFlowEnv(scenario, max_steps=args.steps)
        obs, reset_info = env.reset(seed=args.seed + episode)
        total_reward = 0.0
        first_steps = []
        try:
            for step in range(args.steps):
                all_normalized_valid = all_normalized_valid and normalized_vector_is_valid(obs)
                state_key = discretize_normalized_obs(obs, env.feature_names)
                action_index = choose_action(q_table[state_key], rng, args.epsilon)
                action_record = BASE_ACTIONS[action_index]
                action = env.clip_action(action_record["action"])
                all_actions_valid = all_actions_valid and env.contains_action(action)
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_key = discretize_normalized_obs(next_obs, env.feature_names)
                target = float(reward) + args.gamma * max(q_table[next_key]) * (0.0 if terminated else 1.0)
                q_table[state_key][action_index] += args.alpha * (target - q_table[state_key][action_index])
                total_reward += float(reward)
                if step < 5:
                    first_steps.append(
                        {
                            "step": step,
                            "state_key": state_key,
                            "action_name": action_record["name"],
                            "action": action,
                            "reward": round(float(reward), 6),
                            "next_key": next_key,
                            "action_within_space": info.get("action_within_space"),
                            "normalized_head": [round(float(v), 4) for v in next_obs.get("normalized_vector", [])[:5]],
                        }
                    )
                obs = next_obs
                if terminated or truncated:
                    break
            episode_reports.append(
                {
                    "episode": episode,
                    "steps": step + 1,
                    "total_reward": round(total_reward, 6),
                    "terminated": terminated,
                    "truncated": truncated,
                    "final_trace_hash": env.final_trace_hash(),
                    "first_steps": first_steps,
                }
            )
        finally:
            env.close()

    report = {
        "demo": "minimal_normalized_q_learning",
        "purpose": "Verify the lightest RL loop: normalized obs -> action space -> env.step -> reward -> Q update.",
        "scenario_id": scenario.scenario_id,
        "episodes": args.episodes,
        "steps_per_episode": args.steps,
        "seed": args.seed,
        "observation_schema_version": reset_info.get("observation_schema_version"),
        "feature_names": reset_info.get("feature_names", []),
        "observation_space_spec": reset_info.get("observation_space_spec"),
        "action_space_spec": reset_info.get("action_space_spec"),
        "all_actions_valid": all_actions_valid,
        "all_normalized_values_in_range": all_normalized_valid,
        "episode_rewards": [item["total_reward"] for item in episode_reports],
        "q_state_count": len(q_table),
        "learned_policy": summarize_policy(q_table),
        "episodes_detail": episode_reports,
        "worldsimflow_design_note": {
            "similarity": "WorldSimFlowEnv exposes reset/step, observation space, action space, reward and done/info for lightweight RL loops.",
            "difference": "This demo keeps the first RL loop dependency-free and 2D/LogSim-oriented; richer physics, sensors, and large-scale trainers can be added behind the same interface later.",
        },
    }

    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("minimal_rl_training=ok")
    print(f"scenario_id={scenario.scenario_id}")
    print(f"episodes={args.episodes}")
    print(f"steps_per_episode={args.steps}")
    print(f"feature_count={len(report['feature_names'])}")
    print(f"observation_space_shape={report['observation_space_spec']['shape']}")
    print(f"action_space_low={report['action_space_spec']['low']}")
    print(f"action_space_high={report['action_space_spec']['high']}")
    print(f"all_actions_valid={all_actions_valid}")
    print(f"all_normalized_values_in_range={all_normalized_valid}")
    print(f"q_state_count={len(q_table)}")
    print(f"output={output}")


def discretize_normalized_obs(obs: dict[str, Any], feature_names: list[str]) -> str:
    values = obs.get("normalized_vector") or []
    lookup = {name: float(values[idx]) for idx, name in enumerate(feature_names) if idx < len(values)}
    speed = bucket(lookup.get("ego_speed", 0.0), [-0.70, -0.40, 0.0], ["very_slow", "slow", "target", "fast"])
    lane = bucket(lookup.get("ego_lane_l", 0.0), [-0.18, 0.18], ["right", "center", "left"])
    gap = bucket(lookup.get("front_gap", 1.0), [-0.96, -0.90, -0.70], ["danger", "close", "medium", "open"])
    return f"speed={speed}|lane={lane}|gap={gap}"


def bucket(value: float, thresholds: list[float], labels: list[str]) -> str:
    for threshold, label in zip(thresholds, labels):
        if value < threshold:
            return label
    return labels[-1]


def choose_action(values: list[float], rng: random.Random, epsilon: float) -> int:
    if rng.random() < epsilon:
        return rng.randrange(len(values))
    best = max(values)
    choices = [idx for idx, value in enumerate(values) if value == best]
    return rng.choice(choices)


def normalized_vector_is_valid(obs: dict[str, Any]) -> bool:
    values = obs.get("normalized_vector") or []
    return bool(values) and all(-1.000001 <= float(value) <= 1.000001 for value in values)


def summarize_policy(q_table: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
    out = {}
    for key, values in sorted(q_table.items()):
        best_index = max(range(len(values)), key=lambda idx: values[idx])
        out[key] = {
            "best_action": BASE_ACTIONS[best_index]["name"],
            "q_values": {BASE_ACTIONS[idx]["name"]: round(value, 6) for idx, value in enumerate(values)},
        }
    return out


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()

