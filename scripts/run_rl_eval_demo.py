from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from worldsimflow.core.rl_env import WorldSimFlowEnv
from worldsimflow.core.scenario import ScenarioLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small RL-style evaluation loop on WorldSimFlowEnv.")
    parser.add_argument("--scenario", default="data/sample_scenario.json")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--output", default="outputs/all_results/rl_eval/rl_eval_report.json")
    args = parser.parse_args()

    scenario = ScenarioLoader().load(resolve(args.scenario))
    env = WorldSimFlowEnv(scenario, max_steps=args.steps)
    obs, info = env.reset()
    total_reward = 0.0
    transitions = []
    try:
        for _ in range(args.steps):
            action = policy_action(obs)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            total_reward += reward
            transitions.append(
                {
                    "step": next_obs["step"],
                    "action": action,
                    "reward": round(reward, 6),
                    "terminated": terminated,
                    "truncated": truncated,
                    "event_codes": step_info["event_codes"],
                    "ego_speed": next_obs["ego"]["speed"] if next_obs["ego"] else None,
                    "ego_lane_l": next_obs.get("ego_lane_l"),
                    "front_gap": next_obs["front_gap"],
                    "state_vector_head": next_obs.get("state_vector", [])[:5],
                }
            )
            obs = next_obs
            if terminated or truncated:
                break
        report = {
            "scenario_id": scenario.scenario_id,
            "steps": len(transitions),
            "total_reward": round(total_reward, 6),
            "terminated": transitions[-1]["terminated"] if transitions else False,
            "truncated": transitions[-1]["truncated"] if transitions else False,
            "final_trace_hash": env.final_trace_hash(),
            "observation_schema_version": env.observation_schema_version,
            "feature_names": env.feature_names,
            "feature_count": len(env.feature_names),
            "reset_info": info,
            "transitions": transitions,
        }
        output = resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("rl_eval=ok")
        print(f"scenario_id={scenario.scenario_id}")
        print(f"steps={report['steps']}")
        print(f"total_reward={report['total_reward']}")
        print(f"terminated={report['terminated']}")
        print(f"truncated={report['truncated']}")
        print(f"trace_hash={report['final_trace_hash']}")
        print(f"observation_schema_version={report['observation_schema_version']}")
        print(f"feature_count={report['feature_count']}")
        print(f"output={output}")
    finally:
        env.close()


def policy_action(obs: dict) -> dict[str, float]:
    ego = obs.get("ego") or {}
    speed = float(ego.get("speed", 0.0))
    target_speed = 8.0
    acceleration = max(-2.0, min(2.0, (target_speed - speed) * 0.5))
    lane_offset = float(obs.get("lane_center_offset") or 0.0)
    steering = max(-0.4, min(0.4, -lane_offset * 0.1))
    return {"acceleration": acceleration, "steering": steering}


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
