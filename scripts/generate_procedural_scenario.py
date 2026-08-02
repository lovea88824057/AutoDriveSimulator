from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from worldsimflow.core.procedural_scenario_generator import ProceduralScenarioConfig, ProceduralScenarioGenerator
from worldsimflow.core.scenario_generation import write_scenario_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a lightweight WorldSimFlow procedural Scenario JSON.")
    parser.add_argument("--map", default="S")
    parser.add_argument("--traffic-density", type=float, default=0.35)
    parser.add_argument("--vehicle-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = ProceduralScenarioConfig(
        map_name=args.map,
        traffic_density=args.traffic_density,
        vehicle_count=args.vehicle_count,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    scenario = ProceduralScenarioGenerator().generate(config)
    output = resolve(args.output) if args.output else ROOT / "data" / "generated" / f"{scenario.scenario_id}.json"
    write_scenario_json(scenario, output)
    print("procedural_generation=ok")
    print(f"scenario_id={scenario.scenario_id}")
    print(f"actors={len(scenario.actors)}")
    print(f"output={output}")
    print(json.dumps(scenario.metadata["procedural_generator"], ensure_ascii=False))


def resolve(path: str) -> Path:
    item = Path(path)
    return item if item.is_absolute() else (ROOT / item).resolve()


if __name__ == "__main__":
    main()
