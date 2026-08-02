from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

from .types import DrivableArea, MapFeature, ReplayActor, Road, Scenario, VehicleState


class ScenarioLoader:
    """Load compact log-like scenario files with explicit or generated tracks."""

    def load(self, path: str | Path) -> Scenario:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        road = Road(**data["road"])
        ego = VehicleState(actor_id="ego", **data["ego"])
        dt = float(data["dt"])
        max_steps = int(data["max_steps"])
        metadata = data.get("metadata", {})
        actors = sorted(
            [self._load_actor(item, dt, max_steps) for item in data.get("actors", [])],
            key=lambda actor: actor.actor_id,
        )
        map_features = self._load_map_features(data.get("map_features", metadata.get("map_features", [])))
        drivable_area = self._load_drivable_area(data.get("drivable_area", metadata.get("drivable_area")))
        if drivable_area is None:
            drivable_area = DrivableArea.from_map_features(map_features)
        return Scenario(
            scenario_id=data["scenario_id"],
            seed=int(data["seed"]),
            dt=dt,
            road=road,
            ego=ego,
            actors=actors,
            max_steps=max_steps,
            map_features=map_features,
            drivable_area=drivable_area,
            metadata=metadata,
        )

    def _load_actor(self, item: dict[str, Any], dt: float, max_steps: int) -> ReplayActor:
        actor_id = item["actor_id"]
        if "states" in item:
            states = [VehicleState(actor_id=actor_id, **state) for state in item["states"]]
            states = self._pad_states(states, max_steps)
        elif "trajectory" in item:
            states = self._build_trajectory(actor_id, item["trajectory"], dt, max_steps)
        else:
            raise ValueError(f"Actor {actor_id} must define states or trajectory")
        return ReplayActor(actor_id, states)

    def _load_map_features(self, raw_features: list[dict[str, Any]]) -> list[MapFeature]:
        features: list[MapFeature] = []
        for index, item in enumerate(raw_features or []):
            raw_polyline = item.get("polyline") or item.get("points") or []
            polyline = []
            for point in raw_polyline:
                if isinstance(point, dict):
                    polyline.append((float(point["x"]), float(point["y"])))
                else:
                    polyline.append((float(point[0]), float(point[1])))
            feature_id = str(item.get("feature_id", item.get("id", f"map_feature_{index}")))
            feature_type = str(item.get("feature_type", item.get("type", "UNKNOWN")))
            features.append(MapFeature(feature_id=feature_id, feature_type=feature_type, polyline=polyline))
        return sorted(features, key=lambda feature: feature.feature_id)

    def _load_drivable_area(self, raw: dict[str, Any] | None) -> DrivableArea | None:
        if not raw:
            return None
        polygons = []
        for polygon in raw.get("polygons", []) or []:
            polygons.append([(float(point[0]), float(point[1])) for point in polygon])
        return DrivableArea(
            min_x=float(raw["min_x"]),
            max_x=float(raw["max_x"]),
            min_y=float(raw["min_y"]),
            max_y=float(raw["max_y"]),
            polygons=polygons,
        )

    def _pad_states(self, states: list[VehicleState], max_steps: int) -> list[VehicleState]:
        if not states:
            raise ValueError("Replay actor states cannot be empty")
        if len(states) >= max_steps:
            return states[:max_steps]
        return states + [states[-1]] * (max_steps - len(states))

    def _build_trajectory(
        self,
        actor_id: str,
        trajectory: dict[str, Any],
        dt: float,
        max_steps: int,
    ) -> list[VehicleState]:
        start = trajectory["start"]
        x = float(start["x"])
        y = float(start["y"])
        yaw = float(start.get("yaw", 0.0))
        speed = float(start["speed"])
        target_y = float(trajectory.get("target_y", y))
        lane_change_steps = max(1, int(trajectory.get("lane_change_steps", max_steps)))
        acceleration = float(trajectory.get("acceleration", 0.0))
        hard_brake_at = trajectory.get("hard_brake_at")
        brake_acc = float(trajectory.get("brake_acc", -3.0))
        length = float(trajectory.get("length", 4.6))
        width = float(trajectory.get("width", 1.9))
        object_type = str(trajectory.get("object_type", trajectory.get("type", "VEHICLE")))
        states: list[VehicleState] = []

        for step in range(max_steps):
            if hard_brake_at is not None and step >= int(hard_brake_at):
                speed = max(0.0, speed + brake_acc * dt)
            else:
                speed = max(0.0, speed + acceleration * dt)
            progress = min(1.0, step / lane_change_steps)
            current_y = y + (target_y - y) * progress
            states.append(
                VehicleState(
                    actor_id=actor_id,
                    x=x,
                    y=current_y,
                    yaw=yaw,
                    speed=speed,
                    length=length,
                    width=width,
                    object_type=object_type,
                )
            )
            x += speed * dt
        return states


class ScenarioMutator:
    """Generate reproducible long-tail variants from a base scenario."""

    def mutate_close_cut_in(
        self,
        scenario: Scenario,
        seed: int,
        lateral_shift_range: tuple[float, float] = (-0.8, 0.8),
        speed_scale_range: tuple[float, float] = (0.75, 1.15),
    ) -> Scenario:
        rng = random.Random(seed)
        actors = copy.deepcopy(scenario.actors)
        for actor in actors:
            shifted = []
            lateral_shift = rng.uniform(*lateral_shift_range)
            speed_scale = rng.uniform(*speed_scale_range)
            for state in actor.states:
                shifted.append(
                    VehicleState(
                        actor_id=state.actor_id,
                        x=state.x,
                        y=state.y + lateral_shift,
                        yaw=state.yaw,
                        speed=state.speed * speed_scale,
                        length=state.length,
                        width=state.width,
                        object_type=state.object_type,
                    )
                )
            actor.states[:] = shifted

        metadata = {
            **scenario.metadata,
            "mutation": {
                "type": "close_cut_in",
                "seed": seed,
                "lateral_shift_range": lateral_shift_range,
                "speed_scale_range": speed_scale_range,
            },
        }
        return Scenario(
            scenario_id=f"{scenario.scenario_id}_mut_{seed}",
            seed=seed,
            dt=scenario.dt,
            road=scenario.road,
            ego=scenario.ego,
            actors=sorted(actors, key=lambda actor: actor.actor_id),
            max_steps=scenario.max_steps,
            map_features=scenario.map_features,
            drivable_area=scenario.drivable_area,
            metadata=metadata,
        )
