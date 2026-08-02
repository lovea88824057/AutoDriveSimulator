from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .lane_graph import LaneGraph
from .types import ReplayActor, Scenario, VehicleState


@dataclass(frozen=True)
class TargetInterventionSpec:
    kind: str
    actor_id: str
    start_step: int = 0
    duration: int = 30
    seed: int = 0
    params: dict[str, Any] = field(default_factory=dict)


class TargetActorInterventionEngine:
    """Deterministically modify one chosen actor in a log-like scenario."""

    def apply(self, scenario: Scenario, spec: TargetInterventionSpec) -> Scenario:
        kind = spec.kind.strip().lower().replace("-", "_")
        if kind == "hard_brake":
            return self.hard_brake(scenario, spec)
        if kind == "cut_in":
            return self.cut_in(scenario, spec)
        if kind == "speed_change":
            return self.speed_change(scenario, spec)
        if kind == "lateral_shift":
            return self.lateral_shift(scenario, spec)
        raise ValueError(f"Unsupported target intervention kind: {spec.kind}")

    def hard_brake(self, scenario: Scenario, spec: TargetInterventionSpec) -> Scenario:
        deceleration = float(spec.params.get("deceleration", -4.0))
        start = self._clamp_step(spec.start_step, scenario.max_steps)
        actors = self._replace_actor(
            scenario,
            spec.actor_id,
            lambda actor: self._hard_brake_states(actor.states, start, scenario.dt, deceleration),
        )
        return self._with_metadata(scenario, actors, spec, {"deceleration": deceleration})

    def cut_in(self, scenario: Scenario, spec: TargetInterventionSpec) -> Scenario:
        start = self._clamp_step(spec.start_step, scenario.max_steps)
        duration = max(1, int(spec.duration))
        target_lateral = float(spec.params.get("target_lateral", 0.0))
        target_lane_l = spec.params.get("target_lane_l")
        min_route_speed = float(spec.params.get("min_route_speed", 4.0))
        route_aware = bool(spec.params.get("route_aware", True))
        details: dict[str, Any] = {
            "duration": duration,
            "target_lateral": target_lateral,
            "target_lane_l": 0.0 if target_lane_l is None else float(target_lane_l),
            "min_route_speed": min_route_speed,
            "route_aware": route_aware,
        }

        def update(actor: ReplayActor) -> list[VehicleState]:
            if route_aware:
                states, route_details = self._route_aware_cut_in_states(
                    scenario,
                    actor.states,
                    start,
                    duration,
                    target_lateral,
                    None if target_lane_l is None else float(target_lane_l),
                    min_route_speed,
                )
                details.update(route_details)
                return states
            details["route_aware_status"] = "disabled"
            return self._cut_in_states(scenario, actor.states, start, duration, target_lateral)

        actors = self._replace_actor(scenario, spec.actor_id, update)
        return self._with_metadata(scenario, actors, spec, details)

    def speed_change(self, scenario: Scenario, spec: TargetInterventionSpec) -> Scenario:
        start = self._clamp_step(spec.start_step, scenario.max_steps)
        duration = max(1, int(spec.duration))
        actors = self._replace_actor(
            scenario,
            spec.actor_id,
            lambda actor: self._speed_change_states(actor.states, start, duration, scenario.dt, spec.params),
        )
        return self._with_metadata(scenario, actors, spec, {"duration": duration, **self._speed_change_record(spec.params)})

    def lateral_shift(self, scenario: Scenario, spec: TargetInterventionSpec) -> Scenario:
        start = self._clamp_step(spec.start_step, scenario.max_steps)
        duration = max(1, int(spec.duration))
        shift = float(spec.params.get("shift", spec.params.get("lateral_shift", 0.0)))
        actors = self._replace_actor(
            scenario,
            spec.actor_id,
            lambda actor: self._lateral_shift_states(scenario, actor.states, start, duration, shift),
        )
        return self._with_metadata(scenario, actors, spec, {"duration": duration, "shift": shift})

    def _hard_brake_states(self, states: list[VehicleState], start: int, dt: float, deceleration: float) -> list[VehicleState]:
        changed: list[VehicleState] = []
        for step, state in enumerate(states):
            if step <= start or not changed:
                changed.append(state)
                continue
            prev = changed[-1]
            speed = max(0.0, prev.speed + deceleration * dt)
            yaw = state.yaw
            changed.append(
                VehicleState(
                    actor_id=state.actor_id,
                    x=prev.x + speed * dt * math.cos(yaw),
                    y=prev.y + speed * dt * math.sin(yaw),
                    yaw=yaw,
                    speed=speed,
                    length=state.length,
                    width=state.width,
                    object_type=state.object_type,
                )
            )
        return changed

    def _speed_change_states(
        self,
        states: list[VehicleState],
        start: int,
        duration: int,
        dt: float,
        params: dict[str, Any],
    ) -> list[VehicleState]:
        changed: list[VehicleState] = []
        start_speed = states[start].speed
        if "target_speed" in params and params["target_speed"] is not None:
            final_speed = max(0.0, float(params["target_speed"]))
        elif "speed_delta" in params and params["speed_delta"] is not None:
            final_speed = max(0.0, start_speed + float(params["speed_delta"]))
        else:
            final_speed = max(0.0, start_speed * float(params.get("speed_scale", 0.7)))

        for step, state in enumerate(states):
            if step <= start:
                changed.append(state)
                continue
            progress = min(1.0, (step - start) / duration)
            smooth = progress * progress * (3.0 - 2.0 * progress)
            speed = start_speed + (final_speed - start_speed) * smooth
            prev = changed[-1]
            yaw = state.yaw
            changed.append(
                VehicleState(
                    actor_id=state.actor_id,
                    x=prev.x + speed * dt * math.cos(yaw),
                    y=prev.y + speed * dt * math.sin(yaw),
                    yaw=yaw,
                    speed=speed,
                    length=state.length,
                    width=state.width,
                    object_type=state.object_type,
                )
            )
        return changed

    def _cut_in_states(
        self,
        scenario: Scenario,
        states: list[VehicleState],
        start: int,
        duration: int,
        target_lateral: float,
    ) -> list[VehicleState]:
        start_local = self._to_local(states[start].x, states[start].y, self._ego_state_at(scenario, start))
        return self._rewrite_route_relative_lateral(scenario, states, start, duration, start_local, target_lateral)

    def _route_aware_cut_in_states(
        self,
        scenario: Scenario,
        states: list[VehicleState],
        start: int,
        duration: int,
        target_lateral: float,
        target_lane_l: float | None = None,
        min_route_speed: float = 4.0,
    ) -> tuple[list[VehicleState], dict[str, Any]]:
        graph = LaneGraph.from_scenario(scenario)
        start_state = states[start]
        start_ego = self._ego_state_at(scenario, start)
        start_local = self._to_local(start_state.x, start_state.y, start_ego)
        target_x, target_y = self._from_local(start_local.forward, target_lateral, start_ego)
        target_projection = graph.project(target_x, target_y)
        if target_projection is None:
            return self._cut_in_states(scenario, states, start, duration, target_lateral), {"route_aware_status": "fallback_no_target_projection"}
        source_projection = graph.project_state_to_lane(target_projection.lane_id, start_state)
        if source_projection is None:
            return self._cut_in_states(scenario, states, start, duration, target_lateral), {"route_aware_status": "fallback_no_source_projection"}
        lane_target_l = 0.0 if target_lane_l is None else float(target_lane_l)
        lane_direction = 1.0 if math.cos(start_state.yaw - target_projection.heading) >= 0.0 else -1.0
        lane_length = graph.lane_length(target_projection.lane_id)
        positive_margin = max(0.0, lane_length - source_projection.s)
        negative_margin = max(0.0, source_projection.s)
        required_margin = max(min_route_speed * scenario.dt * min(duration, scenario.max_steps - start - 1), graph.lane_width)
        preferred_margin = positive_margin if lane_direction >= 0.0 else negative_margin
        opposite_margin = negative_margin if lane_direction >= 0.0 else positive_margin
        if preferred_margin < required_margin and opposite_margin > preferred_margin:
            lane_direction *= -1.0

        changed: list[VehicleState] = []
        actor_distance = 0.0
        previous_actor = start_state
        for step, state in enumerate(states):
            if step < start:
                changed.append(state)
                continue
            if step > start:
                route_speed = max(0.0, previous_actor.speed, state.speed)
                if step <= start + duration:
                    route_speed = max(route_speed, min_route_speed)
                actor_distance += route_speed * scenario.dt
            progress = min(1.0, (step - start) / duration)
            smooth = progress * progress * (3.0 - 2.0 * progress)
            source_pose = graph.sample_pose(target_projection.lane_id, source_projection.s + lane_direction * actor_distance, source_projection.l)
            target_pose = graph.sample_pose(target_projection.lane_id, target_projection.s + lane_direction * actor_distance, lane_target_l)
            x = source_pose.x + (target_pose.x - source_pose.x) * smooth
            y = source_pose.y + (target_pose.y - source_pose.y) * smooth
            changed.append(
                VehicleState(
                    actor_id=state.actor_id,
                    x=x,
                    y=y,
                    yaw=state.yaw,
                    speed=state.speed,
                    length=state.length,
                    width=state.width,
                    object_type=state.object_type,
                )
            )
            previous_actor = state
        return self._with_consistent_kinematics(changed, start, scenario.dt), {
            "route_aware_status": "ok",
            "target_lane_id": target_projection.lane_id,
            "target_lane_source_s": round(source_projection.s, 4),
            "target_lane_target_s": round(target_projection.s, 4),
            "target_lane_source_l": round(source_projection.l, 4),
            "target_lane_target_l": round(lane_target_l, 4),
            "target_lane_direction": lane_direction,
            "target_lane_positive_margin": round(positive_margin, 4),
            "target_lane_negative_margin": round(negative_margin, 4),
            "target_lane_required_margin": round(required_margin, 4),
        }

    def _lateral_shift_states(
        self,
        scenario: Scenario,
        states: list[VehicleState],
        start: int,
        duration: int,
        shift: float,
    ) -> list[VehicleState]:
        start_local = self._to_local(states[start].x, states[start].y, self._ego_state_at(scenario, start))
        rewritten = self._rewrite_lateral(scenario, states, start, duration, start_local.lateral, start_local.lateral + shift)
        return self._with_consistent_kinematics(rewritten, start, scenario.dt)

    def _rewrite_route_relative_lateral(
        self,
        scenario: Scenario,
        states: list[VehicleState],
        start: int,
        duration: int,
        start_local: _LocalPoint,
        target_lateral: float,
    ) -> list[VehicleState]:
        changed: list[VehicleState] = []
        ego_distance = 0.0
        actor_distance = 0.0
        previous_ego = self._ego_state_at(scenario, start)
        previous_actor = states[start]
        for step, state in enumerate(states):
            if step < start:
                changed.append(state)
                continue
            ego = self._ego_state_at(scenario, step)
            if step > start:
                ego_distance += math.hypot(ego.x - previous_ego.x, ego.y - previous_ego.y)
                actor_distance += max(0.0, previous_actor.speed) * scenario.dt
            progress = min(1.0, (step - start) / duration)
            smooth = progress * progress * (3.0 - 2.0 * progress)
            forward = start_local.forward + actor_distance - ego_distance
            lateral = start_local.lateral + (target_lateral - start_local.lateral) * smooth
            x, y = self._from_local(forward, lateral, ego)
            changed.append(
                VehicleState(
                    actor_id=state.actor_id,
                    x=x,
                    y=y,
                    yaw=state.yaw,
                    speed=state.speed,
                    length=state.length,
                    width=state.width,
                    object_type=state.object_type,
                )
            )
            previous_ego = ego
            previous_actor = state
        return self._with_consistent_kinematics(changed, start, scenario.dt)

    def _rewrite_lateral(
        self,
        scenario: Scenario,
        states: list[VehicleState],
        start: int,
        duration: int,
        source_lateral: float,
        target_lateral: float,
    ) -> list[VehicleState]:
        changed: list[VehicleState] = []
        for step, state in enumerate(states):
            if step < start:
                changed.append(state)
                continue
            ego = self._ego_state_at(scenario, step)
            local = self._to_local(state.x, state.y, ego)
            progress = min(1.0, (step - start) / duration)
            smooth = progress * progress * (3.0 - 2.0 * progress)
            lateral = source_lateral + (target_lateral - source_lateral) * smooth
            x, y = self._from_local(local.forward, lateral, ego)
            changed.append(
                VehicleState(
                    actor_id=state.actor_id,
                    x=x,
                    y=y,
                    yaw=state.yaw,
                    speed=state.speed,
                    length=state.length,
                    width=state.width,
                    object_type=state.object_type,
                )
            )
        return changed

    def _with_consistent_kinematics(
        self,
        states: list[VehicleState],
        start: int,
        dt: float,
        min_motion: float = 1e-6,
    ) -> list[VehicleState]:
        if len(states) < 2:
            return states
        synced: list[VehicleState] = []
        for step, state in enumerate(states):
            if step < start:
                synced.append(state)
                continue
            if step < len(states) - 1:
                reference = states[step + 1]
                dx = reference.x - state.x
                dy = reference.y - state.y
            else:
                reference = states[step - 1]
                dx = state.x - reference.x
                dy = state.y - reference.y
            distance = math.hypot(dx, dy)
            if distance > min_motion:
                yaw = math.atan2(dy, dx)
                speed = distance / max(dt, 1e-6)
            elif synced:
                yaw = synced[-1].yaw
                speed = 0.0
            else:
                yaw = state.yaw
                speed = 0.0
            synced.append(
                VehicleState(
                    actor_id=state.actor_id,
                    x=state.x,
                    y=state.y,
                    yaw=yaw,
                    speed=speed,
                    length=state.length,
                    width=state.width,
                    object_type=state.object_type,
                )
            )
        return synced

    def _replace_actor(self, scenario: Scenario, actor_id: str, update) -> list[ReplayActor]:
        found = False
        actors: list[ReplayActor] = []
        for actor in scenario.actors:
            if actor.actor_id != actor_id:
                actors.append(copy.deepcopy(actor))
                continue
            found = True
            actors.append(ReplayActor(actor.actor_id, update(actor)))
        if not found:
            available = ", ".join(actor.actor_id for actor in scenario.actors[:12])
            raise KeyError(f"actor_id={actor_id!r} not found. First available actors: {available}")
        return actors

    def _with_metadata(
        self,
        scenario: Scenario,
        actors: list[ReplayActor],
        spec: TargetInterventionSpec,
        details: dict[str, Any],
    ) -> Scenario:
        kind = spec.kind.strip().lower().replace("-", "_")
        actor_token = self._safe_token(spec.actor_id)
        record = {
            "phase": 2,
            "kind": kind,
            "target_actor_id": spec.actor_id,
            "seed": spec.seed,
            "start_step": self._clamp_step(spec.start_step, scenario.max_steps),
            **details,
        }
        metadata = {
            **scenario.metadata,
            "phase2": True,
            "targeted_intervention": True,
            "mutation": record,
            "interventions": [*scenario.metadata.get("interventions", []), record],
            "tags": sorted(set([*scenario.metadata.get("tags", []), "phase2", "targeted", kind])),
        }
        return Scenario(
            scenario_id=f"{scenario.scenario_id}__target_{actor_token}_{kind}_{spec.seed}",
            seed=spec.seed,
            dt=scenario.dt,
            road=scenario.road,
            ego=scenario.ego,
            actors=sorted(actors, key=lambda actor: actor.actor_id),
            max_steps=scenario.max_steps,
            map_features=scenario.map_features,
            drivable_area=scenario.drivable_area,
            metadata=metadata,
        )

    def _ego_state_at(self, scenario: Scenario, step: int) -> VehicleState:
        raw_states = scenario.metadata.get("ego_replay_states")
        if raw_states:
            item = raw_states[min(step, len(raw_states) - 1)]
            return VehicleState(actor_id="ego", **item)
        return VehicleState(
            actor_id="ego",
            x=scenario.ego.x + scenario.ego.speed * scenario.dt * step * math.cos(scenario.ego.yaw),
            y=scenario.ego.y + scenario.ego.speed * scenario.dt * step * math.sin(scenario.ego.yaw),
            yaw=scenario.ego.yaw,
            speed=scenario.ego.speed,
            length=scenario.ego.length,
            width=scenario.ego.width,
            object_type=scenario.ego.object_type,
        )

    def _to_local(self, x: float, y: float, ego: VehicleState) -> _LocalPoint:
        dx = x - ego.x
        dy = y - ego.y
        cos_yaw = math.cos(ego.yaw)
        sin_yaw = math.sin(ego.yaw)
        return _LocalPoint(cos_yaw * dx + sin_yaw * dy, -sin_yaw * dx + cos_yaw * dy)

    def _from_local(self, forward: float, lateral: float, ego: VehicleState) -> tuple[float, float]:
        cos_yaw = math.cos(ego.yaw)
        sin_yaw = math.sin(ego.yaw)
        x = ego.x + cos_yaw * forward - sin_yaw * lateral
        y = ego.y + sin_yaw * forward + cos_yaw * lateral
        return x, y

    def _speed_change_record(self, params: dict[str, Any]) -> dict[str, Any]:
        return {
            key: float(params[key])
            for key in ["target_speed", "speed_delta", "speed_scale"]
            if key in params and params[key] is not None
        }

    def _clamp_step(self, step: int, max_steps: int) -> int:
        return max(0, min(int(step), max_steps - 1))

    def _safe_token(self, value: str) -> str:
        token = re.sub(r"[^0-9A-Za-z_\-]+", "_", str(value)).strip("_")
        return token[:48] or "actor"


@dataclass(frozen=True)
class _LocalPoint:
    forward: float
    lateral: float
