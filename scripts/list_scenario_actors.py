from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from worldsimflow.core.scenario import ScenarioLoader


def main() -> None:
    parser = argparse.ArgumentParser(description="List actors in a WorldSimFlow scenario.")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()

    scenario = ScenarioLoader().load(resolve(args.scenario))
    step = max(0, min(args.step, scenario.max_steps - 1))
    print(f"scenario_id={scenario.scenario_id}")
    print(f"actors={len(scenario.actors)} step={step}")
    print("actor_id\tobject_type\tx\ty\tyaw_deg\tspeed\tlength\twidth")
    for actor in scenario.actors[: args.limit]:
        state = actor.states[min(step, len(actor.states) - 1)]
        print(
            f"{actor.actor_id}\t{state.object_type}\t{state.x:.2f}\t{state.y:.2f}\t"
            f"{math.degrees(state.yaw):.1f}\t{state.speed:.2f}\t{state.length:.2f}\t{state.width:.2f}"
        )


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


if __name__ == "__main__":
    main()
