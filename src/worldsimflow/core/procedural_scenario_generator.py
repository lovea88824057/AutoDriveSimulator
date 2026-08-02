from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

from .scenario_generation import scenario_to_dict
from .types import DrivableArea, MapFeature, ReplayActor, Road, Scenario, VehicleState


@dataclass(frozen=True)
class ProceduralScenarioConfig:
    map_name: str = "S"
    traffic_density: float = 0.2
    seed: int = 0
    vehicle_count: int = 8
    max_steps: int = 120
    dt: float = 0.1
    metadata: dict[str, Any] = field(default_factory=dict)


class ProceduralScenarioGenerator:
    """Create lightweight synthetic scenarios for WorldSimFlow smoke tests and RL loops."""

    def generate(self, config: ProceduralScenarioConfig) -> Scenario:
        rng = random.Random(config.seed)
        road = Road(length=300.0, lane_width=3.6, lane_count=3)
        ego = VehicleState(actor_id="ego", x=0.0, y=0.0, yaw=0.0, speed=7.0, object_type="EGO")
        map_features = self._build_map_features(config.map_name, road)
        drivable_area = DrivableArea.from_map_features(map_features, margin=road.lane_width * 0.6)
        actors: list[ReplayActor] = []
        spacing = max(10.0, 35.0 - config.traffic_density * 20.0)
        for index in range(config.vehicle_count):
            lane = rng.choice([-1, 0, 1])
            start_x = 28.0 + index * spacing + rng.uniform(-3.0, 3.0)
            speed = rng.uniform(4.5, 9.5)
            states = []
            for step in range(config.max_steps):
                x = start_x + speed * config.dt * step
                y = lane * road.lane_width
                yaw = 0.0
                if "C" in config.map_name.upper():
                    curve = 0.0025 * x
                    y += math.sin(curve) * road.lane_width * 0.6
                    yaw = math.atan2(math.cos(curve) * 0.0025 * road.lane_width * 0.6, 1.0)
                states.append(
                    VehicleState(
                        actor_id=f"flow_actor_{index:02d}",
                        x=x,
                        y=y,
                        yaw=yaw,
                        speed=speed,
                        length=4.6,
                        width=1.9,
                        object_type="VEHICLE",
                    )
                )
            actors.append(ReplayActor(f"flow_actor_{index:02d}", states))
        metadata = {
            "source": "worldsimflow_procedural",
            "procedural_generator": {
                "map": config.map_name,
                "traffic_density": config.traffic_density,
                "vehicle_count": config.vehicle_count,
            },
            "tags": ["synthetic", "procedural", "map_aware"],
            **config.metadata,
        }
        return Scenario(
            scenario_id=f"procedural_{config.map_name}_{config.seed}",
            seed=config.seed,
            dt=config.dt,
            road=road,
            ego=ego,
            actors=actors,
            max_steps=config.max_steps,
            map_features=map_features,
            drivable_area=drivable_area,
            metadata=metadata,
        )

    def generate_dict(self, config: ProceduralScenarioConfig) -> dict[str, Any]:
        return scenario_to_dict(self.generate(config))

    def _build_map_features(self, map_name: str, road: Road) -> list[MapFeature]:
        xs = [i * road.length / 60.0 for i in range(61)]
        half_width = road.half_width
        lane_offsets = [(-half_width + road.lane_width * i) for i in range(road.lane_count + 1)]
        features: list[MapFeature] = []
        for index, offset in enumerate(lane_offsets):
            points = [(x, self._lane_y(x, offset, map_name, road)) for x in xs]
            feature_type = "ROAD_EDGE_BOUNDARY" if index in {0, len(lane_offsets) - 1} else "ROAD_LINE_BROKEN_WHITE"
            features.append(MapFeature(f"lane_line_{index}", feature_type, points))
        center_points = [(x, self._lane_y(x, 0.0, map_name, road)) for x in xs]
        features.append(MapFeature("lane_center_0", "LANE_CENTER", center_points))
        return features

    def _lane_y(self, x: float, offset: float, map_name: str, road: Road) -> float:
        if "C" in map_name.upper():
            return offset + math.sin(0.0025 * x) * road.lane_width * 0.6
        return offset

