from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from worldsimflow.core.policy_evaluator import PolicyEvaluationConfig, PolicyEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multiple WorldSimFlow policies on the same scenario set.")
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["data/sample_scenario.json", "data/worldsimflow_mini_straight.json", "data/generated/phase2"],
    )
    parser.add_argument("--policies", nargs="+", default=["rule", "random", "minimal-q"], choices=["rule", "random", "minimal-q"])
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--max-scenarios", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/all_results/policy_eval")
    args = parser.parse_args()

    report = PolicyEvaluator(
        PolicyEvaluationConfig(
            roots=[resolve(item) for item in args.roots],
            output_dir=resolve(args.output_dir),
            policies=list(args.policies),
            episodes=args.episodes,
            max_steps=args.steps,
            seed=args.seed,
            max_scenarios=args.max_scenarios or None,
        )
    ).evaluate()
    ranking = report["summary"]["ranking"]

    print("policy_evaluation=ok")
    print(f"scenario_count={report['scenario_count']}")
    print(f"policy_count={report['policy_count']}")
    print(f"episode_count={report['episode_count']}")
    print(f"steps_per_episode={report['steps_per_episode']}")
    print("ranking=" + ",".join(f"{item['policy_name']}:{item['mean_reward_per_scenario']}" for item in ranking))
    print(f"report={report['report']}")
    print(f"html={report['html']}")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
