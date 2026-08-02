from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .types import DrivableArea, MapFeature, ReplayActor, Road, Scenario, VehicleState


@dataclass(frozen=True)
class InterventionSpec:
    kind: str
    seed: int = 0
    intensity: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


class ScenarioInterventionEngine:
    """Deterministic scene intervention and long-tail variant generation."""

    def apply(self, scenario: Scenario, spec: InterventionSpec) -> Scenario:
        kind = spec.kind.strip().lower().replace("-", "_")
        if kind == "hard_brake":
            return self.hard_brake(scenario, spec)
        if kind == "cut_in":
            return self.cut_in(scenario, spec)
        if kind == "close_follow":
            return self.close_follow(scenario, spec)
        if kind == "pedestrian_crossing":
            return self.pedestrian_crossing(scenario, spec)
        raise ValueError(f"Unsupported intervention kind: {spec.kind}")

    def generate_suite(self, scenario: Scenario, seed: int = 20260721) -> list[Scenario]:
        specs = [
            InterventionSpec("hard_brake", seed=seed + 1, intensity=1.0, params={"trigger_step": 24}),
            InterventionSpec("cut_in", seed=seed + 2, intensity=1.0, params={"start_step": 16, "duration": 28}),
            InterventionSpec("close_follow", seed=seed + 3, intensity=1.0, params={"gap": 8.5}),
            InterventionSpec(
                "pedestrian_crossing",
                seed=seed + 4,
                intensity=1.0,
                params={"start_step": 22, "duration": 36, "forward": 20.0},
            ),
        ]
        return [self.apply(scenario, spec) for spec in specs]

    def hard_brake(self, scenario: Scenario, spec: InterventionSpec) -> Scenario:
        rng = random.Random(spec.seed)
        trigger_step = int(spec.params.get("trigger_step", scenario.max_steps // 3))
        decel = float(spec.params.get("deceleration", -4.0 * spec.intensity))
        actor = self._select_front_actor(scenario, trigger_step, prefer_vehicle=True)
        actors = []
        for item in scenario.actors:
            if item.actor_id != actor.actor_id:
                actors.append(copy.deepcopy(item))
                continue
            states = []
            for step, state in enumerate(item.states):
                if step <= trigger_step or not states:
                    states.append(state)
                    continue
                prev = states[-1]
                yaw = state.yaw
                speed = max(0.0, prev.speed + decel * scenario.dt)
                states.append(
                    VehicleState(
                        actor_id=state.actor_id,
                        x=prev.x + speed * scenario.dt * math.cos(yaw),
                        y=prev.y + speed * scenario.dt * math.sin(yaw),
                        yaw=yaw,
                        speed=speed,
                        length=state.length,
                        width=state.width,
                        object_type=state.object_type,
                    )
                )
            actors.append(ReplayActor(item.actor_id, states))
        return self._with_mutation_metadata(
            scenario,
            actors,
            kind="hard_brake",
            seed=spec.seed,
            details={"actor_id": actor.actor_id, "trigger_step": trigger_step, "deceleration": decel},
        )

    def cut_in(self, scenario: Scenario, spec: InterventionSpec) -> Scenario:
        rng = random.Random(spec.seed)
        start_step = int(spec.params.get("start_step", scenario.max_steps // 5))
        duration = max(1, int(spec.params.get("duration", scenario.max_steps // 3)))
        target_lateral = float(spec.params.get("target_lateral", rng.uniform(-0.35, 0.35)))
        actor = self._select_cut_in_actor(scenario, start_step)
        actors = []
        for item in scenario.actors:
            if item.actor_id != actor.actor_id:
                actors.append(copy.deepcopy(item))
                continue
            original_start = self._to_ego_local(scenario, item.states[start_step], start_step).lateral
            states = []
            for step, state in enumerate(item.states):
                if step < start_step:
                    states.append(state)
                    continue
                ego = self._ego_state_at(scenario, step)
                local = self._to_local(state.x, state.y, ego)
                progress = min(1.0, (step - start_step) / duration)
                smooth = progress * progress * (3.0 - 2.0 * progress)
                lateral = original_start + (target_lateral - original_start) * smooth
                x, y = self._from_local(local.forward, lateral, ego)
                states.append(
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
            actors.append(ReplayActor(item.actor_id, states))
        return self._with_mutation_metadata(
            scenario,
            actors,
            kind="cut_in",
            seed=spec.seed,
            details={
                "actor_id": actor.actor_id,
                "start_step": start_step,
                "duration": duration,
                "target_lateral": round(target_lateral, 4),
            },
        )

    def close_follow(self, scenario: Scenario, spec: InterventionSpec) -> Scenario:
        gap = float(spec.params.get("gap", 8.5))
        speed_scale = float(spec.params.get("speed_scale", 0.88))
        actor_id = str(spec.params.get("actor_id", f"phase2_close_follow_{spec.seed}"))
        states = []
        for step in range(scenario.max_steps):
            ego = self._ego_state_at(scenario, step)
            x, y = self._from_local(gap, 0.0, ego)
            states.append(
                VehicleState(
                    actor_id=actor_id,
                    x=x,
                    y=y,
                    yaw=ego.yaw,
                    speed=max(0.0, ego.speed * speed_scale),
                    length=4.6,
                    width=1.9,
                    object_type="VEHICLE",
                )
            )
        actors = [copy.deepcopy(item) for item in scenario.actors]
        actors.append(ReplayActor(actor_id, states))
        return self._with_mutation_metadata(
            scenario,
            actors,
            kind="close_follow",
            seed=spec.seed,
            details={"actor_id": actor_id, "gap": gap, "speed_scale": speed_scale},
        )

    def pedestrian_crossing(self, scenario: Scenario, spec: InterventionSpec) -> Scenario:
        start_step = int(spec.params.get("start_step", scenario.max_steps // 4))
        duration = max(1, int(spec.params.get("duration", scenario.max_steps // 3)))
        forward = float(spec.params.get("forward", 20.0))
        lateral_start = float(spec.params.get("lateral_start", -5.5))
        lateral_end = float(spec.params.get("lateral_end", 5.5))
        actor_id = str(spec.params.get("actor_id", f"phase2_pedestrian_{spec.seed}"))
        states = []
        for step in range(scenario.max_steps):
            ego = self._ego_state_at(scenario, step)
            if step < start_step:
                lateral = lateral_start
            elif step > start_step + duration:
                lateral = lateral_end
            else:
                progress = (step - start_step) / duration
                lateral = lateral_start + (lateral_end - lateral_start) * progress
            x, y = self._from_local(forward, lateral, ego)
            states.append(
                VehicleState(
                    actor_id=actor_id,
                    x=x,
                    y=y,
                    yaw=ego.yaw + math.pi / 2.0,
                    speed=abs(lateral_end - lateral_start) / max(duration * scenario.dt, 1e-6),
                    length=0.8,
                    width=0.8,
                    object_type="PEDESTRIAN",
                )
            )
        actors = [copy.deepcopy(item) for item in scenario.actors]
        actors.append(ReplayActor(actor_id, states))
        return self._with_mutation_metadata(
            scenario,
            actors,
            kind="pedestrian_crossing",
            seed=spec.seed,
            details={
                "actor_id": actor_id,
                "start_step": start_step,
                "duration": duration,
                "forward": forward,
                "lateral_start": lateral_start,
                "lateral_end": lateral_end,
            },
        )

    def _with_mutation_metadata(
        self,
        scenario: Scenario,
        actors: list[ReplayActor],
        kind: str,
        seed: int,
        details: dict[str, Any],
    ) -> Scenario:
        history = list(scenario.metadata.get("interventions", []))
        record = {"phase": 2, "kind": kind, "seed": seed, **details}
        history.append(record)
        metadata = {
            **scenario.metadata,
            "phase2": True,
            "interventions": history,
            "mutation": record,
            "tags": sorted(set([*scenario.metadata.get("tags", []), "phase2", kind])),
        }
        return Scenario(
            scenario_id=f"{scenario.scenario_id}__{kind}_{seed}",
            seed=seed,
            dt=scenario.dt,
            road=scenario.road,
            ego=scenario.ego,
            actors=sorted(actors, key=lambda item: item.actor_id),
            max_steps=scenario.max_steps,
            map_features=scenario.map_features,
            drivable_area=scenario.drivable_area,
            metadata=metadata,
        )

    def _select_front_actor(self, scenario: Scenario, step: int, prefer_vehicle: bool) -> ReplayActor:
        step = min(step, scenario.max_steps - 1)
        candidates = []
        for actor in scenario.actors:
            state = actor.states[step]
            local = self._to_ego_local(scenario, state, step)
            is_vehicle = state.object_type in {"VEHICLE", "EGO"}
            if prefer_vehicle and not is_vehicle:
                continue
            if local.forward > 5.0 and abs(local.lateral) <= max(4.5, scenario.road.lane_width):
                candidates.append((local.forward, actor))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1]
        vehicle_fallback = [actor for actor in scenario.actors if actor.states[step].object_type == "VEHICLE"]
        if vehicle_fallback:
            return vehicle_fallback[0]
        if not scenario.actors:
            raise ValueError("Scenario has no actors to intervene on")
        return scenario.actors[0]

    def _select_cut_in_actor(self, scenario: Scenario, step: int) -> ReplayActor:
        step = min(step, scenario.max_steps - 1)
        candidates = []
        for actor in scenario.actors:
            state = actor.states[step]
            if state.object_type != "VEHICLE":
                continue
            local = self._to_ego_local(scenario, state, step)
            if 8.0 <= local.forward <= 65.0 and abs(local.lateral) >= scenario.road.lane_width * 0.7:
                candidates.append((abs(local.lateral), local.forward, actor))
        if candidates:
            candidates.sort(key=lambda item: (-item[0], item[1]))
            return candidates[0][2]
        return self._select_front_actor(scenario, step, prefer_vehicle=True)

    def _ego_state_at(self, scenario: Scenario, step: int) -> VehicleState:
        raw_states = scenario.metadata.get("ego_replay_states")
        if raw_states:
            item = raw_states[min(step, len(raw_states) - 1)]
            return VehicleState(actor_id="ego", **item)
        if step == 0:
            return scenario.ego
        dt = scenario.dt
        speed = scenario.ego.speed
        return VehicleState(
            actor_id="ego",
            x=scenario.ego.x + speed * dt * step * math.cos(scenario.ego.yaw),
            y=scenario.ego.y + speed * dt * step * math.sin(scenario.ego.yaw),
            yaw=scenario.ego.yaw,
            speed=speed,
            length=scenario.ego.length,
            width=scenario.ego.width,
            object_type=scenario.ego.object_type,
        )

    def _to_ego_local(self, scenario: Scenario, state: VehicleState, step: int) -> _LocalPoint:
        ego = self._ego_state_at(scenario, step)
        local = self._to_local(state.x, state.y, ego)
        return _LocalPoint(local.forward, local.lateral)

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


@dataclass(frozen=True)
class _LocalPoint:
    forward: float
    lateral: float


def scenario_to_dict(scenario: Scenario) -> dict[str, Any]:
    return {
        "scenario_id": scenario.scenario_id,
        "seed": scenario.seed,
        "dt": scenario.dt,
        "max_steps": scenario.max_steps,
        "road": road_to_dict(scenario.road),
        "ego": vehicle_to_dict(scenario.ego, include_actor_id=False),
        "actors": [actor_to_dict(actor) for actor in scenario.actors],
        "map_features": [map_feature_to_dict(feature) for feature in scenario.map_features],
        "drivable_area": drivable_area_to_dict(scenario.drivable_area),
        "metadata": scenario.metadata,
    }


def road_to_dict(road: Road) -> dict[str, Any]:
    return {"length": road.length, "lane_width": road.lane_width, "lane_count": road.lane_count}


def map_feature_to_dict(feature: MapFeature) -> dict[str, Any]:
    return {
        "feature_id": feature.feature_id,
        "type": feature.feature_type,
        "polyline": [[round(float(x), 4), round(float(y), 4)] for x, y in feature.polyline],
    }


def drivable_area_to_dict(area: DrivableArea | None) -> dict[str, Any] | None:
    if area is None:
        return None
    return {
        "min_x": round(float(area.min_x), 4),
        "max_x": round(float(area.max_x), 4),
        "min_y": round(float(area.min_y), 4),
        "max_y": round(float(area.max_y), 4),
        "polygons": [
            [[round(float(x), 4), round(float(y), 4)] for x, y in polygon]
            for polygon in area.polygons
        ],
    }


def actor_to_dict(actor: ReplayActor) -> dict[str, Any]:
    return {"actor_id": actor.actor_id, "states": [vehicle_to_dict(state, include_actor_id=False) for state in actor.states]}


def vehicle_to_dict(state: VehicleState, include_actor_id: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "x": round(float(state.x), 4),
        "y": round(float(state.y), 4),
        "yaw": round(float(state.yaw), 6),
        "speed": round(float(state.speed), 4),
        "length": round(float(state.length), 4),
        "width": round(float(state.width), 4),
        "object_type": state.object_type,
    }
    if include_actor_id:
        result["actor_id"] = state.actor_id
    return result


def write_scenario_json(scenario: Scenario, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scenario_to_dict(scenario), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_many_scenarios(scenarios: Iterable[Scenario], output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return [write_scenario_json(scenario, root / f"{scenario.scenario_id}.json") for scenario in scenarios]
