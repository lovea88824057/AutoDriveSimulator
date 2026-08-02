from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

from .traffic_policy import MapAwareTrafficConfig, TrafficPolicyRunner
from .types import ReplayActor, Scenario

TrafficMode = Literal["replay", "lane_aware_idm", "hybrid"]


@dataclass(frozen=True)
class TrafficManagerConfig:
    mode: TrafficMode = "replay"
    desired_speed: float = 9.0
    min_gap: float = 4.0
    time_headway: float = 1.3
    lane_snap_ratio: float = 0.25
    max_reactive_actors: int | None = None
    reactive_actor_ids: tuple[str, ...] | None = None
    vehicle_only: bool = True


class TrafficManagerLite:
    """Lightweight traffic orchestration layer for replay and reactive actors.

    The manager decides whether background actors are pure replay, lane-aware IDM,`r`n    or hybrid controlled actors. It uses a compact deterministic implementation:`r`n    select actors,
    delegates reactive motion to LaneGraph + IDM, then returns a normal replayable
    Scenario so the rest of the pipeline stays unchanged.
    """

    def __init__(self, runner: TrafficPolicyRunner | None = None):
        self.runner = runner or TrafficPolicyRunner()

    def build_scenario(self, scenario: Scenario, config: TrafficManagerConfig | None = None) -> Scenario:
        cfg = config or TrafficManagerConfig()
        mode = self._normalize_mode(cfg.mode)
        if mode == "replay":
            return self._replay_scenario(scenario, cfg)
        if mode == "lane_aware_idm":
            return self._lane_aware_scenario(scenario, cfg, actor_ids=cfg.reactive_actor_ids, suffix="traffic_lane_aware_idm")
        if mode == "hybrid":
            actor_ids = cfg.reactive_actor_ids or self._default_reactive_ids(scenario, cfg)
            return self._lane_aware_scenario(scenario, cfg, actor_ids=actor_ids, suffix="traffic_hybrid", manager_mode="hybrid")
        raise ValueError(f"Unsupported traffic mode: {cfg.mode!r}")

    def _replay_scenario(self, scenario: Scenario, config: TrafficManagerConfig) -> Scenario:
        metadata = self._manager_metadata(scenario, config, mode="replay", reactive_actor_ids=())
        return Scenario(
            scenario_id=f"{scenario.scenario_id}__traffic_replay",
            seed=scenario.seed,
            dt=scenario.dt,
            road=scenario.road,
            ego=scenario.ego,
            actors=[ReplayActor(actor.actor_id, list(actor.states)) for actor in copy.deepcopy(scenario.actors)],
            max_steps=scenario.max_steps,
            map_features=scenario.map_features,
            drivable_area=scenario.drivable_area,
            metadata=metadata,
        )

    def _lane_aware_scenario(
        self,
        scenario: Scenario,
        config: TrafficManagerConfig,
        actor_ids: tuple[str, ...] | None,
        suffix: str,
        manager_mode: TrafficMode = "lane_aware_idm",
    ) -> Scenario:
        policy_config = MapAwareTrafficConfig(
            desired_speed=config.desired_speed,
            min_gap=config.min_gap,
            time_headway=config.time_headway,
            lane_snap_ratio=config.lane_snap_ratio,
            max_actor_count=config.max_reactive_actors if actor_ids is None else None,
            actor_ids=actor_ids,
            vehicle_only=config.vehicle_only,
        )
        generated = self.runner.rebuild_map_aware_idm_traffic(scenario, policy_config)
        reactive_ids = actor_ids or tuple(record.actor_id for record in scenario.actors if self._is_reactive_candidate(record, config))
        if config.max_reactive_actors is not None and actor_ids is None:
            reactive_ids = reactive_ids[: config.max_reactive_actors]
        metadata = self._manager_metadata(generated, config, mode=manager_mode, reactive_actor_ids=reactive_ids)
        return Scenario(
            scenario_id=f"{scenario.scenario_id}__{suffix}",
            seed=generated.seed,
            dt=generated.dt,
            road=generated.road,
            ego=generated.ego,
            actors=generated.actors,
            max_steps=generated.max_steps,
            map_features=generated.map_features,
            drivable_area=generated.drivable_area,
            metadata=metadata,
        )

    def _manager_metadata(
        self,
        scenario: Scenario,
        config: TrafficManagerConfig,
        mode: TrafficMode,
        reactive_actor_ids: tuple[str, ...],
    ) -> dict:
        record = {
            "kind": "traffic_manager_lite",
            "mode": mode,
            "reactive_actor_ids": list(reactive_actor_ids),
            "reactive_actor_count": len(reactive_actor_ids),
            "desired_speed": config.desired_speed,
            "min_gap": config.min_gap,
            "time_headway": config.time_headway,
            "lane_snap_ratio": config.lane_snap_ratio,
        }
        return {
            **scenario.metadata,
            "traffic_manager_mode": mode,
            "traffic_manager": record,
            "traffic_managers": [*scenario.metadata.get("traffic_managers", []), record],
            "tags": sorted(set([*scenario.metadata.get("tags", []), "traffic_manager_lite", mode])),
        }

    def _default_reactive_ids(self, scenario: Scenario, config: TrafficManagerConfig) -> tuple[str, ...]:
        ids = [actor.actor_id for actor in scenario.actors if self._is_reactive_candidate(actor, config)]
        if config.max_reactive_actors is not None:
            ids = ids[: config.max_reactive_actors]
        return tuple(ids)

    def _is_reactive_candidate(self, actor: ReplayActor, config: TrafficManagerConfig) -> bool:
        if not actor.states:
            return False
        if not config.vehicle_only:
            return True
        return actor.states[0].object_type == "VEHICLE"

    def _normalize_mode(self, mode: str) -> TrafficMode:
        normalized = mode.strip().lower().replace("-", "_")
        aliases = {
            "idm": "lane_aware_idm",
            "lane_idm": "lane_aware_idm",
            "map_aware_idm": "lane_aware_idm",
            "log": "replay",
            "log_replay": "replay",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"replay", "lane_aware_idm", "hybrid"}:
            raise ValueError(f"Unsupported traffic mode: {mode!r}")
        return normalized  # type: ignore[return-value]


