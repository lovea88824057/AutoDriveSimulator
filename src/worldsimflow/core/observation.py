from __future__ import annotations

import math
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Iterable

from .lane_graph import LaneGraph
from .spaces import BoxSpaceSpec
from .types import Action, HealthEvent, Scenario, VehicleState


@dataclass(frozen=True)
class ObservationConfig:
    """Configuration for the lightweight state observation schema."""

    schema_version: str = "state_v1"
    nearby_radius: float = 50.0
    front_max_lateral: float | None = None
    no_front_gap_value: float = 1000.0
    no_nearest_actor_value: float = 1000.0
    include_state_vector: bool = True
    include_normalized_vector: bool = True


class ObservationBuilder:
    """Build stable model-facing observations from raw simulator observations.

    The simulator backend is free to expose rich Python objects. This class turns those
    objects into a versioned dict/vector contract for RL, VLA and future world-model
    data export.
    """

    feature_names = [
        "ego_speed",
        "ego_accel",
        "ego_yaw_rate",
        "ego_lane_s",
        "ego_lane_l",
        "ego_heading_error",
        "route_progress",
        "front_gap",
        "front_relative_speed",
        "nearby_actor_count",
        "nearest_actor_distance",
        "collision_flag",
        "offroad_flag",
        "stale_replay_flag",
        "last_acceleration",
        "last_steering",
        "reactive_actor_count",
    ]

    def __init__(self, scenario: Scenario, config: ObservationConfig | None = None):
        self.scenario = scenario
        self.config = config or ObservationConfig()
        self.lane_graph = LaneGraph.from_scenario(scenario)


    def observation_space_spec(self) -> BoxSpaceSpec:
        road_length = max(float(self.scenario.road.length), 1.0)
        front_high = max(float(self.config.no_front_gap_value), road_length)
        nearest_high = max(float(self.config.no_nearest_actor_value), road_length)
        return BoxSpaceSpec(
            name=f"{self.config.schema_version}_observation",
            shape=(len(self.feature_names),),
            labels=list(self.feature_names),
            low=[
                0.0,      # ego_speed
                -10.0,    # ego_accel
                -3.0,     # ego_yaw_rate
                0.0,      # ego_lane_s
                -10.0,    # ego_lane_l
                -math.pi, # ego_heading_error
                0.0,      # route_progress
                0.0,      # front_gap
                -50.0,    # front_relative_speed
                0.0,      # nearby_actor_count
                0.0,      # nearest_actor_distance
                0.0,      # collision_flag
                0.0,      # offroad_flag
                0.0,      # stale_replay_flag
                -6.0,     # last_acceleration
                -1.0,     # last_steering
                0.0,      # reactive_actor_count
            ],
            high=[
                50.0,
                10.0,
                3.0,
                road_length,
                10.0,
                math.pi,
                1.0,
                front_high,
                50.0,
                64.0,
                nearest_high,
                1.0,
                1.0,
                1.0,
                3.0,
                1.0,
                128.0,
            ],
        )
    def build(
        self,
        raw_observation: dict[str, Any],
        *,
        step: int | None = None,
        last_action: Action | dict[str, float] | list[float] | tuple[float, float] | None = None,
        previous_observation: dict[str, Any] | None = None,
        events: Iterable[HealthEvent | dict[str, Any]] = (),
    ) -> dict[str, Any]:
        ego = self._state(raw_observation.get("ego"))
        previous_ego = self._state(previous_observation.get("ego")) if previous_observation else None
        actors = [state for state in (self._state(item) for item in raw_observation.get("actors", [])) if state]
        step_id = int(raw_observation.get("step", 0) if step is None else step)
        action = self._action(last_action)
        event_codes = self._event_codes(events)

        lane = self._lane_features(ego)
        front = self._front_features(ego, actors)
        surrounding = self._surrounding_features(ego, actors)
        dynamics = self._dynamics(ego, previous_ego)
        traffic = self._traffic_features()
        safety = {
            "collision_flag": 1.0 if "collision" in event_codes else 0.0,
            "offroad_flag": 1.0 if "offroad" in event_codes else 0.0,
            "stale_replay_flag": 1.0 if "stale_replay" in event_codes else 0.0,
        }

        obs: dict[str, Any] = {
            "schema_version": self.config.schema_version,
            "scenario_id": self.scenario.scenario_id,
            "step": step_id,
            "dt": self.scenario.dt,
            "ego": self._state_dict(ego),
            "ego_speed": ego.speed if ego else 0.0,
            "ego_accel": dynamics["ego_accel"],
            "ego_yaw_rate": dynamics["ego_yaw_rate"],
            "ego_lane_id": lane["ego_lane_id"],
            "ego_lane_s": lane["ego_lane_s"],
            "ego_lane_l": lane["ego_lane_l"],
            "ego_heading_error": lane["ego_heading_error"],
            "route_progress": lane["route_progress"],
            "front_actor_id": front["front_actor_id"],
            "front_gap": front["front_gap"],
            "front_relative_speed": front["front_relative_speed"],
            "nearby_actor_count": surrounding["nearby_actor_count"],
            "nearest_actor_id": surrounding["nearest_actor_id"],
            "nearest_actor_distance": surrounding["nearest_actor_distance"],
            "collision_flag": safety["collision_flag"],
            "offroad_flag": safety["offroad_flag"],
            "stale_replay_flag": safety["stale_replay_flag"],
            "last_acceleration": action.acceleration if action else 0.0,
            "last_steering": action.steering if action else 0.0,
            "traffic_manager_mode": traffic["traffic_manager_mode"],
            "reactive_actor_count": traffic["reactive_actor_count"],
            "actor_count": len(actors),
            "map_feature_count": int(raw_observation.get("map_feature_count", len(self.scenario.map_features))),
            "ego_mode": raw_observation.get("ego_mode", self.scenario.metadata.get("ego_mode", "closed_loop")),
            "event_codes": event_codes,
            # Backward-compatible aliases used by existing demos/policies.
            "lane_center_offset": lane["ego_lane_l"] if lane["ego_lane_l"] is not None else raw_observation.get("lane_center_offset"),
            "closest_actor_distance": surrounding["nearest_actor_distance"],
            "raw_front_gap": raw_observation.get("front_gap"),
        }
        if self.config.include_state_vector:
            obs["feature_names"] = list(self.feature_names)
            obs["state_vector"] = self.build_vector(obs)
            if self.config.include_normalized_vector:
                obs["normalized_vector"] = self.observation_space_spec().normalize(obs["state_vector"])
                obs["normalized_range"] = [-1.0, 1.0]
        return obs

    def build_vector(self, observation: dict[str, Any]) -> list[float]:
        return [self._vector_value(name, observation.get(name)) for name in self.feature_names]

    def _lane_features(self, ego: VehicleState | None) -> dict[str, Any]:
        if ego is None:
            return {"ego_lane_id": None, "ego_lane_s": None, "ego_lane_l": None, "ego_heading_error": None, "route_progress": 0.0}
        projection = self.lane_graph.project_state(ego)
        if projection is None:
            return {"ego_lane_id": None, "ego_lane_s": None, "ego_lane_l": None, "ego_heading_error": None, "route_progress": 0.0}
        lane_length = max(self.lane_graph.lane_length(projection.lane_id), 1e-6)
        heading_error = self.lane_graph.heading_mismatch(ego, projection)
        return {
            "ego_lane_id": projection.lane_id,
            "ego_lane_s": projection.s,
            "ego_lane_l": projection.l,
            "ego_heading_error": heading_error,
            "route_progress": max(0.0, min(1.0, projection.s / lane_length)),
        }

    def _front_features(self, ego: VehicleState | None, actors: list[VehicleState]) -> dict[str, Any]:
        if ego is None:
            return {"front_actor_id": None, "front_gap": None, "front_relative_speed": None}
        front, _ = self.lane_graph.find_front_back_actors(ego, actors, max_lateral=self.config.front_max_lateral)
        if front is None:
            return {"front_actor_id": None, "front_gap": None, "front_relative_speed": None}
        actor, _projection, center_gap = front
        bbox_gap = max(0.0, center_gap - (ego.length + actor.length) / 2.0)
        return {
            "front_actor_id": actor.actor_id,
            "front_gap": bbox_gap,
            "front_relative_speed": actor.speed - ego.speed,
        }

    def _surrounding_features(self, ego: VehicleState | None, actors: list[VehicleState]) -> dict[str, Any]:
        if ego is None or not actors:
            return {"nearby_actor_count": 0, "nearest_actor_id": None, "nearest_actor_distance": None}
        nearby = 0
        nearest_actor_id = None
        nearest_distance = None
        for actor in actors:
            distance = math.hypot(actor.x - ego.x, actor.y - ego.y)
            if distance <= self.config.nearby_radius:
                nearby += 1
            if nearest_distance is None or distance < nearest_distance:
                nearest_actor_id = actor.actor_id
                nearest_distance = distance
        return {"nearby_actor_count": nearby, "nearest_actor_id": nearest_actor_id, "nearest_actor_distance": nearest_distance}

    def _dynamics(self, ego: VehicleState | None, previous_ego: VehicleState | None) -> dict[str, float]:
        if ego is None or previous_ego is None:
            return {"ego_accel": 0.0, "ego_yaw_rate": 0.0}
        dt = max(self.scenario.dt, 1e-9)
        return {
            "ego_accel": (ego.speed - previous_ego.speed) / dt,
            "ego_yaw_rate": self._angle_diff(ego.yaw, previous_ego.yaw) / dt,
        }

    def _traffic_features(self) -> dict[str, Any]:
        manager = self.scenario.metadata.get("traffic_manager") or {}
        mode = self.scenario.metadata.get("traffic_manager_mode") or self.scenario.metadata.get("traffic_mode") or "replay"
        return {
            "traffic_manager_mode": mode,
            "reactive_actor_count": int(manager.get("reactive_actor_count", 0) or 0),
        }

    def _state(self, value: Any) -> VehicleState | None:
        if value is None:
            return None
        if isinstance(value, VehicleState):
            return value
        if is_dataclass(value):
            value = asdict(value)
        if isinstance(value, dict):
            return VehicleState(
                actor_id=str(value.get("actor_id", "")),
                x=float(value.get("x", 0.0)),
                y=float(value.get("y", 0.0)),
                yaw=float(value.get("yaw", 0.0)),
                speed=float(value.get("speed", 0.0)),
                length=float(value.get("length", 4.6)),
                width=float(value.get("width", 1.9)),
                object_type=str(value.get("object_type", "VEHICLE")),
            )
        return None

    def _state_dict(self, value: VehicleState | None) -> dict[str, Any] | None:
        return asdict(value) if value is not None else None

    def _action(self, value: Action | dict[str, float] | list[float] | tuple[float, float] | None) -> Action | None:
        if value is None:
            return None
        if isinstance(value, Action):
            return value
        if isinstance(value, dict):
            return Action(acceleration=float(value.get("acceleration", 0.0)), steering=float(value.get("steering", 0.0)))
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return Action(acceleration=float(value[0]), steering=float(value[1]))
        raise TypeError("last_action must be Action, dict, [acceleration, steering], or None")

    def _event_codes(self, events: Iterable[HealthEvent | dict[str, Any]]) -> list[str]:
        codes = []
        for event in events:
            if isinstance(event, HealthEvent):
                codes.append(event.code)
            elif isinstance(event, dict) and event.get("code") is not None:
                codes.append(str(event["code"]))
        return codes

    def _vector_value(self, name: str, value: Any) -> float:
        if value is None:
            if name == "front_gap":
                return float(self.config.no_front_gap_value)
            if name == "nearest_actor_distance":
                return float(self.config.no_nearest_actor_value)
            return 0.0
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            if math.isfinite(float(value)):
                return float(value)
            return 0.0
        return 0.0

    def _angle_diff(self, current: float, previous: float) -> float:
        return (current - previous + math.pi) % (2.0 * math.pi) - math.pi


