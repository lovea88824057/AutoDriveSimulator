from __future__ import annotations

import argparse
from collections import Counter
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from worldsimflow.core.scenario import ScenarioLoader
from worldsimflow.core.sim import MiniDrivingSimulator
from worldsimflow.core.types import Action


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit WorldSimFlow scenario inputs and runtime coverage.")
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    scenario_path = Path(args.scenario)
    if not scenario_path.is_absolute():
        scenario_path = (ROOT / scenario_path).resolve()

    raw = json.loads(scenario_path.read_text(encoding="utf-8-sig"))
    scenario = ScenarioLoader().load(scenario_path)
    sim = MiniDrivingSimulator(scenario)
    reset_obs = sim.reset()
    step_obs, reward, done, events = sim.step(Action(acceleration=0.0, steering=0.0))

    metadata = raw.get("metadata", {})
    map_features = metadata.get("map_features", [])
    ego_replay_states = metadata.get("ego_replay_states", [])
    raw_actor_metadata_count = sum(1 for actor in raw.get("actors", []) if actor.get("metadata"))
    actor_state_lengths = [len(actor.get("states", [])) for actor in raw.get("actors", [])]
    actor_type_counts = Counter(
        actor.get("states", [{}])[0].get("object_type", "UNKNOWN")
        for actor in raw.get("actors", [])
        if actor.get("states")
    )
    truck_like_count = sum(
        1
        for actor in raw.get("actors", [])
        if actor.get("states")
        and actor["states"][0].get("object_type") == "VEHICLE"
        and (actor["states"][0].get("length", 0.0) >= 7.0 or actor["states"][0].get("width", 0.0) >= 2.6)
    )

    print(f"scenario={scenario.scenario_id}")
    print(f"path={scenario_path}")
    print("\n[loaded_by_scenario_loader]")
    print(f"dt={scenario.dt}")
    print(f"max_steps={scenario.max_steps}")
    print(f"road={scenario.road}")
    print(f"ego_initial={scenario.ego}")
    print(f"actors={len(scenario.actors)}")
    print(f"actor_state_len_minmax={min(actor_state_lengths) if actor_state_lengths else 0}/{max(actor_state_lengths) if actor_state_lengths else 0}")
    print(f"actor_type_counts={dict(sorted(actor_type_counts.items()))}")
    print(f"truck_like_vehicle_count={truck_like_count}")
    print(f"metadata_keys={sorted(scenario.metadata.keys())}")

    print("\n[used_by_replay_simulator]")
    print("ego_initial_or_metadata.ego_replay_states=yes")
    print(f"ego_replay_states={len(ego_replay_states)}")
    print("actors.states=yes")
    print("road.length/lane_width/lane_count=yes")
    print("dt/max_steps=yes")
    print(f"observation_keys={sorted(step_obs.keys())}")
    print(f"sample_front_gap={step_obs.get('front_gap')}")
    print(f"sample_closest_actor_distance={step_obs.get('closest_actor_distance')}")
    print(f"sample_nearby_actor_count={step_obs.get('nearby_actor_count')}")
    print(f"sample_reward={reward}")
    print(f"sample_done={done}")
    print(f"sample_events={[event.code for event in events]}")

    print("\n[used_by_html_viewer]")
    print("trace frames: ego/actors/reward/events/action/trace_hash=yes")
    print(f"metadata.map_features={len(map_features)}")
    print(f"metadata.ego_mode={metadata.get('ego_mode')}")
    print(f"metadata.monitor_mode={metadata.get('monitor_mode')}")

    print("\n[not_fully_modeled_yet]")
    print(f"actor.metadata_loaded_into_runtime=no (raw actors with metadata={raw_actor_metadata_count})")
    print("metadata.map_features_used_for_physics=no, viewer_only=yes")
    print("dynamic_map_states/traffic_lights=not_in_worldsimflow_json_yet")
    print("raw sensor/camera/rgb=not_in_worldsimflow_json_yet")
    print("true drivable_area_offroad_check=not_yet, current HealthMonitor is lightweight")


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    main()