from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VehicleState:
    actor_id: str
    x: float
    y: float
    yaw: float
    speed: float
    length: float = 4.6
    width: float = 1.9
    object_type: str = "VEHICLE"


@dataclass(frozen=True)
class Action:
    acceleration: float
    steering: float


@dataclass(frozen=True)
class Road:
    length: float
    lane_width: float
    lane_count: int

    @property
    def half_width(self) -> float:
        return self.lane_width * self.lane_count / 2.0


@dataclass(frozen=True)
class MapFeature:
    feature_id: str
    feature_type: str
    polyline: list[tuple[float, float]]


@dataclass(frozen=True)
class DrivableArea:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    polygons: list[list[tuple[float, float]]] = field(default_factory=list)

    def contains(self, x: float, y: float) -> bool:
        if x < self.min_x or x > self.max_x or y < self.min_y or y > self.max_y:
            return False
        if not self.polygons:
            return True
        return any(self._point_in_polygon(x, y, polygon) for polygon in self.polygons)

    @classmethod
    def from_map_features(cls, features: list[MapFeature], margin: float = 6.0) -> "DrivableArea | None":
        points = [point for feature in features for point in feature.polyline]
        if not points:
            return None
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return cls(
            min_x=min(xs) - margin,
            max_x=max(xs) + margin,
            min_y=min(ys) - margin,
            max_y=max(ys) + margin,
        )

    def _point_in_polygon(self, x: float, y: float, polygon: list[tuple[float, float]]) -> bool:
        inside = False
        if len(polygon) < 3:
            return False
        j = len(polygon) - 1
        for i, current in enumerate(polygon):
            xi, yi = current
            xj, yj = polygon[j]
            crosses = (yi > y) != (yj > y)
            if crosses:
                denom = yj - yi
                if abs(denom) < 1e-12:
                    j = i
                    continue
                x_at_y = (xj - xi) * (y - yi) / denom + xi
                if x < x_at_y:
                    inside = not inside
            j = i
        return inside


@dataclass(frozen=True)
class ReplayActor:
    actor_id: str
    states: list[VehicleState]


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    seed: int
    dt: float
    road: Road
    ego: VehicleState
    actors: list[ReplayActor]
    max_steps: int
    map_features: list[MapFeature] = field(default_factory=list)
    drivable_area: DrivableArea | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HealthEvent:
    step: int
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class StepResult:
    step: int
    observation: dict[str, Any]
    reward: float
    done: bool
    events: list[HealthEvent]
    trace_hash: str
    action: Action | None = None
