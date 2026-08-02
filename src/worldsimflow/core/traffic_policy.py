from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

from .lane_graph import FrenetProjection, LaneGraph
from .types import MapFeature, ReplayActor, Scenario, VehicleState


@dataclass(frozen=True)
class IDMConfig:
    desired_speed: float = 10.0
    min_gap: float = 4.0
    time_headway: float = 1.2
    max_accel: float = 1.5
    comfortable_brake: float = 2.5
    delta: float = 4.0
    initial_gap: float = 16.0
    lateral_offset: float = 0.0
    length: float = 4.6
    width: float = 1.9
    object_type: str = "VEHICLE"


@dataclass(frozen=True)
class MapAwareTrafficConfig:
    desired_speed: float = 9.0
    min_gap: float = 4.0
    time_headway: float = 1.3
    max_accel: float = 1.2
    comfortable_brake: float = 2.5
    lane_snap_ratio: float = 0.25
    max_actor_count: int | None = None
    actor_ids: tuple[str, ...] | None = None
    vehicle_only: bool = True


class IDMLitePolicy:
    """A deterministic lightweight IDM follower used to generate reactive traffic tracks."""

    def rollout_follower(
        self,
        leader_states: list[VehicleState],
        dt: float,
        actor_id: str,
        config: IDMConfig | None = None,
    ) -> list[VehicleState]:
        cfg = config or IDMConfig()
        if not leader_states:
            raise ValueError("leader_states cannot be empty")

        leader0 = leader_states[0]
        x, y = self._from_leader_local(leader0, -cfg.initial_gap, cfg.lateral_offset)
        yaw = leader0.yaw
        speed = min(cfg.desired_speed, max(0.0, leader0.speed * 0.85))
        states: list[VehicleState] = []

        for step, leader in enumerate(leader_states):
            if step > 0:
                gap = max(0.1, self._longitudinal_gap(x, y, cfg.length, leader))
                rel_speed = speed - leader.speed
                accel = self._acceleration(speed, rel_speed, gap, cfg)
                speed = max(0.0, speed + accel * dt)
                yaw = leader.yaw
                x += speed * dt * math.cos(yaw)
                y += speed * dt * math.sin(yaw)
                desired_y = self._from_leader_local(leader, -gap, cfg.lateral_offset)[1]
                y = 0.85 * y + 0.15 * desired_y
            states.append(
                VehicleState(
                    actor_id=actor_id,
                    x=x,
                    y=y,
                    yaw=yaw,
                    speed=speed,
                    length=cfg.length,
                    width=cfg.width,
                    object_type=cfg.object_type,
                )
            )
        return states

    def _acceleration(self, speed: float, rel_speed: float, gap: float, cfg: IDMConfig) -> float:
        desired_speed = max(cfg.desired_speed, 0.1)
        brake_term = 2.0 * math.sqrt(max(cfg.max_accel * cfg.comfortable_brake, 1e-6))
        desired_gap = cfg.min_gap + max(0.0, speed * cfg.time_headway + speed * rel_speed / brake_term)
        free_road = (speed / desired_speed) ** cfg.delta
        interaction = (desired_gap / max(gap, 0.1)) ** 2
        accel = cfg.max_accel * (1.0 - free_road - interaction)
        return max(-cfg.comfortable_brake * 2.0, min(cfg.max_accel, accel))

    def _longitudinal_gap(self, follower_x: float, follower_y: float, follower_length: float, leader: VehicleState) -> float:
        dx = leader.x - follower_x
        dy = leader.y - follower_y
        forward = math.cos(leader.yaw) * dx + math.sin(leader.yaw) * dy
        return forward - (leader.length + follower_length) / 2.0

    def _from_leader_local(self, leader: VehicleState, backward: float, lateral: float) -> tuple[float, float]:
        forward = backward - leader.length / 2.0
        x = leader.x + math.cos(leader.yaw) * forward - math.sin(leader.yaw) * lateral
        y = leader.y + math.sin(leader.yaw) * forward + math.cos(leader.yaw) * lateral
        return x, y


class LaneCenterlineIndex:
    """Legacy helper kept for compatibility; B3 uses LaneGraph/Frenet instead."""

    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.centerlines = self._extract_centerlines(scenario)

    def assign_lane(self, state: VehicleState) -> int:
        best = (float("inf"), 0)
        for index, line in enumerate(self.centerlines):
            y, _ = self.pose_at(index, state.x)
            distance = abs(state.y - y)
            if distance < best[0]:
                best = (distance, index)
        return best[1]

    def pose_at(self, lane_id: int, x: float) -> tuple[float, float]:
        line = self.centerlines[max(0, min(lane_id, len(self.centerlines) - 1))]
        if len(line) < 2:
            return line[0][1], 0.0
        points = sorted(line, key=lambda p: p[0])
        if x <= points[0][0]:
            a, b = points[0], points[1]
        elif x >= points[-1][0]:
            a, b = points[-2], points[-1]
        else:
            a, b = points[0], points[-1]
            for left, right in zip(points, points[1:]):
                if left[0] <= x <= right[0]:
                    a, b = left, right
                    break
        span = b[0] - a[0]
        ratio = 0.0 if abs(span) < 1e-6 else (x - a[0]) / span
        y = a[1] + (b[1] - a[1]) * max(0.0, min(1.0, ratio))
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        return y, yaw

    def _extract_centerlines(self, scenario: Scenario) -> list[list[tuple[float, float]]]:
        features = [feature for feature in scenario.map_features if self._is_centerline(feature)]
        if not features:
            features = [feature for feature in scenario.map_features if "ROAD_LINE" in feature.feature_type]
        lines = [feature.polyline for feature in features if len(feature.polyline) >= 2]
        if lines:
            return sorted(lines, key=lambda line: sum(point[1] for point in line) / len(line))
        return self._fallback_lines(scenario)

    def _fallback_lines(self, scenario: Scenario) -> list[list[tuple[float, float]]]:
        half_width = scenario.road.half_width
        starts = [-half_width + scenario.road.lane_width * (index + 0.5) for index in range(scenario.road.lane_count)]
        return [[(-20.0, y), (scenario.road.length + 20.0, y)] for y in starts]

    def _is_centerline(self, feature: MapFeature) -> bool:
        return "LANE_CENTER" in feature.feature_type or "CENTER" in feature.feature_type


class TrafficPolicyRunner:
    """Apply lightweight traffic policies to a Scenario and return a replayable Scenario."""

    def __init__(self, idm_policy: IDMLitePolicy | None = None):
        self.idm_policy = idm_policy or IDMLitePolicy()

    def add_idm_follower(
        self,
        scenario: Scenario,
        actor_id: str = "idm_follower",
        leader_id: str = "ego",
        config: IDMConfig | None = None,
    ) -> Scenario:
        leader_states = self._leader_states(scenario, leader_id)
        follower_states = self.idm_policy.rollout_follower(leader_states, scenario.dt, actor_id, config)
        actors = [copy.deepcopy(actor) for actor in scenario.actors if actor.actor_id != actor_id]
        actors.append(ReplayActor(actor_id, follower_states))
        cfg = config or IDMConfig()
        policy_record: dict[str, Any] = {
            "kind": "idm_lite_follower",
            "actor_id": actor_id,
            "leader_id": leader_id,
            "desired_speed": cfg.desired_speed,
            "min_gap": cfg.min_gap,
            "time_headway": cfg.time_headway,
            "initial_gap": cfg.initial_gap,
        }
        metadata = {
            **scenario.metadata,
            "traffic_policies": [*scenario.metadata.get("traffic_policies", []), policy_record],
            "tags": sorted(set([*scenario.metadata.get("tags", []), "traffic_policy", "idm_lite"])),
        }
        return Scenario(
            scenario_id=f"{scenario.scenario_id}__idm_lite_{actor_id}",
            seed=scenario.seed,
            dt=scenario.dt,
            road=scenario.road,
            ego=scenario.ego,
            actors=sorted(actors, key=lambda actor: actor.actor_id),
            max_steps=scenario.max_steps,
            map_features=scenario.map_features,
            drivable_area=scenario.drivable_area,
            metadata=metadata,
        )

    def rebuild_map_aware_idm_traffic(
        self,
        scenario: Scenario,
        config: MapAwareTrafficConfig | None = None,
    ) -> Scenario:
        """Rebuild selected background vehicles with LaneGraph/Frenet + IDM.

        The method name is kept for compatibility with earlier scripts. Since B3, the
        implementation is lane-aware: each actor is projected to a LaneGraph segment,
        keeps its own s/l state, queries leaders in the same lane, and samples the next
        pose from the lane geometry instead of moving along global x/y.
        """
        cfg = config or MapAwareTrafficConfig()
        raw_candidates = [actor for actor in scenario.actors if self._actor_supported(actor, cfg)]
        if cfg.max_actor_count is not None:
            raw_candidates = raw_candidates[: cfg.max_actor_count]

        graph = LaneGraph.from_scenario(scenario)
        projections = {actor.actor_id: graph.project_state(actor.states[0]) for actor in raw_candidates}
        candidates = [actor for actor in raw_candidates if projections[actor.actor_id] is not None]
        skipped = [actor.actor_id for actor in raw_candidates if projections[actor.actor_id] is None]
        candidate_ids = {actor.actor_id for actor in candidates}
        untouched = [copy.deepcopy(actor) for actor in scenario.actors if actor.actor_id not in candidate_ids]
        ego_states = self._leader_states(scenario, "ego")

        current = {actor.actor_id: actor.states[0] for actor in candidates}
        generated: dict[str, list[VehicleState]] = {actor.actor_id: [actor.states[0]] for actor in candidates}
        lane_ids = {actor.actor_id: projections[actor.actor_id].lane_id for actor in candidates if projections[actor.actor_id] is not None}
        lane_s = {actor.actor_id: projections[actor.actor_id].s for actor in candidates if projections[actor.actor_id] is not None}
        lane_l = {actor.actor_id: projections[actor.actor_id].l for actor in candidates if projections[actor.actor_id] is not None}
        lane_direction = {
            actor.actor_id: self._lane_direction(actor.states[0], projections[actor.actor_id])
            for actor in candidates
            if projections[actor.actor_id] is not None
        }
        idm_cfg = IDMConfig(
            desired_speed=cfg.desired_speed,
            min_gap=cfg.min_gap,
            time_headway=cfg.time_headway,
            max_accel=cfg.max_accel,
            comfortable_brake=cfg.comfortable_brake,
        )

        for step in range(1, scenario.max_steps):
            previous = dict(current)
            ego_prev = ego_states[min(step - 1, len(ego_states) - 1)]
            ego_projection = graph.project_state(ego_prev)
            next_states: dict[str, VehicleState] = {}
            for actor in candidates:
                prev = previous[actor.actor_id]
                lane_id = lane_ids[actor.actor_id]
                direction = lane_direction[actor.actor_id]
                leader = self._nearest_lane_leader(
                    follower=prev,
                    follower_s=lane_s[actor.actor_id],
                    follower_lane=lane_id,
                    follower_direction=direction,
                    previous=previous,
                    lane_ids=lane_ids,
                    lane_s=lane_s,
                    ego=ego_prev,
                    ego_projection=ego_projection,
                )
                speed = prev.speed
                if leader is not None:
                    rel_speed = speed - leader.state.speed
                    accel = self.idm_policy._acceleration(speed, rel_speed, max(0.1, leader.gap), idm_cfg)
                else:
                    accel = self.idm_policy._acceleration(speed, 0.0, 1e6, idm_cfg)
                speed = max(0.0, speed + accel * scenario.dt)

                next_s = lane_s[actor.actor_id] + direction * speed * scenario.dt
                next_l = lane_l[actor.actor_id] * (1.0 - cfg.lane_snap_ratio)
                pose = graph.sample_pose(lane_id, next_s, next_l)
                lane_s[actor.actor_id] = pose.s
                lane_l[actor.actor_id] = pose.l
                lane_yaw = pose.heading if direction >= 0.0 else _wrap_to_pi(pose.heading + math.pi)
                state = VehicleState(
                    actor_id=prev.actor_id,
                    x=pose.x,
                    y=pose.y,
                    yaw=lane_yaw,
                    speed=speed,
                    length=prev.length,
                    width=prev.width,
                    object_type=prev.object_type,
                )
                generated[actor.actor_id].append(state)
                next_states[actor.actor_id] = state
            current = next_states

        actors = [*untouched, *[ReplayActor(actor.actor_id, generated[actor.actor_id]) for actor in candidates]]
        metadata = {
            **scenario.metadata,
            "traffic_mode": "lane_aware_idm",
            "traffic_policies": [
                *scenario.metadata.get("traffic_policies", []),
                {
                    "kind": "lane_aware_idm",
                    "actor_count": len(candidates),
                    "skipped_unprojected_actor_count": len(skipped),
                    "desired_speed": cfg.desired_speed,
                    "min_gap": cfg.min_gap,
                    "time_headway": cfg.time_headway,
                    "lane_snap_ratio": cfg.lane_snap_ratio,
                    "lane_graph": graph.to_summary(),
                    "map_feature_count": len(scenario.map_features),
                },
            ],
            "tags": sorted(
                set(
                    [
                        *scenario.metadata.get("tags", []),
                        "traffic_policy",
                        "map_aware_idm",
                        "lane_aware_idm",
                        "non_log_traffic",
                    ]
                )
            ),
        }
        return Scenario(
            scenario_id=f"{scenario.scenario_id}__lane_aware_idm",
            seed=scenario.seed,
            dt=scenario.dt,
            road=scenario.road,
            ego=scenario.ego,
            actors=sorted(actors, key=lambda actor: actor.actor_id),
            max_steps=scenario.max_steps,
            map_features=scenario.map_features,
            drivable_area=scenario.drivable_area,
            metadata=metadata,
        )

    def _actor_supported(self, actor: ReplayActor, config: MapAwareTrafficConfig) -> bool:
        if not actor.states:
            return False
        if config.actor_ids is not None and actor.actor_id not in config.actor_ids:
            return False
        if config.vehicle_only and actor.states[0].object_type != "VEHICLE":
            return False
        return True

    def _lane_direction(self, state: VehicleState, projection: FrenetProjection | None) -> float:
        if projection is None:
            return 1.0
        return 1.0 if math.cos(state.yaw - projection.heading) >= 0.0 else -1.0

    def _nearest_lane_leader(
        self,
        follower: VehicleState,
        follower_s: float,
        follower_lane: str,
        follower_direction: float,
        previous: dict[str, VehicleState],
        lane_ids: dict[str, str],
        lane_s: dict[str, float],
        ego: VehicleState,
        ego_projection: FrenetProjection | None,
    ) -> "_LaneLeader | None":
        candidates: list[tuple[VehicleState, float]] = []
        if ego_projection is not None and ego_projection.lane_id == follower_lane:
            candidates.append((ego, ego_projection.s))
        for actor_id, state in previous.items():
            if actor_id == follower.actor_id or lane_ids.get(actor_id) != follower_lane:
                continue
            candidates.append((state, lane_s[actor_id]))
        best: _LaneLeader | None = None
        for candidate, candidate_s in candidates:
            longitudinal = (candidate_s - follower_s) * follower_direction
            gap = longitudinal - (candidate.length + follower.length) / 2.0
            if gap <= 0.0:
                continue
            if best is None or gap < best.gap:
                best = _LaneLeader(candidate, gap)
        return best

    def _nearest_leader(
        self,
        follower: VehicleState,
        follower_lane: int,
        previous: dict[str, VehicleState],
        lane_ids: dict[str, int],
        ego: VehicleState,
    ) -> VehicleState | None:
        candidates = [ego]
        candidates.extend(state for actor_id, state in previous.items() if actor_id != follower.actor_id and lane_ids.get(actor_id) == follower_lane)
        best: tuple[float, VehicleState] | None = None
        for candidate in candidates:
            gap = self._forward_gap(follower, candidate)
            if gap <= 0.0:
                continue
            if best is None or gap < best[0]:
                best = (gap, candidate)
        return best[1] if best else None

    def _forward_gap(self, follower: VehicleState, leader: VehicleState) -> float:
        dx = leader.x - follower.x
        dy = leader.y - follower.y
        forward = math.cos(follower.yaw) * dx + math.sin(follower.yaw) * dy
        return forward - (leader.length + follower.length) / 2.0

    def _leader_states(self, scenario: Scenario, leader_id: str) -> list[VehicleState]:
        if leader_id == "ego":
            raw_states = scenario.metadata.get("ego_replay_states")
            if raw_states:
                states = [VehicleState(actor_id="ego", **state) for state in raw_states]
                if len(states) >= scenario.max_steps:
                    return states[: scenario.max_steps]
                return states + [states[-1]] * (scenario.max_steps - len(states))
            return [
                VehicleState(
                    actor_id="ego",
                    x=scenario.ego.x + scenario.ego.speed * scenario.dt * step * math.cos(scenario.ego.yaw),
                    y=scenario.ego.y + scenario.ego.speed * scenario.dt * step * math.sin(scenario.ego.yaw),
                    yaw=scenario.ego.yaw,
                    speed=scenario.ego.speed,
                    length=scenario.ego.length,
                    width=scenario.ego.width,
                    object_type=scenario.ego.object_type,
                )
                for step in range(scenario.max_steps)
            ]
        for actor in scenario.actors:
            if actor.actor_id == leader_id:
                return actor.states
        raise KeyError(f"leader_id={leader_id!r} not found")


@dataclass(frozen=True)
class _LaneLeader:
    state: VehicleState
    gap: float


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi