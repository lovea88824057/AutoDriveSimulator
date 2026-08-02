from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any

from worldsimflow.backends import ReplayBackend, SimulationBackend

from .bev_observation import BEVRasterConfig, BEVRasterObservationBuilder
from .observation import ObservationBuilder, ObservationConfig
from .spaces import BoxSpaceSpec, action_space_spec
from .success_criterion import SuccessCriterion
from .termination import infer_done_reason
from .types import Action, HealthEvent, Scenario


class WorldSimFlowEnv:
    """Gymnasium-style evaluation wrapper without adding a gym dependency."""

    def __init__(
        self,
        scenario: Scenario,
        backend: SimulationBackend | None = None,
        max_steps: int | None = None,
        observation_builder: ObservationBuilder | None = None,
        observation_config: ObservationConfig | None = None,
        include_bev: bool = False,
        bev_config: BEVRasterConfig | None = None,
        bev_history_length: int = 1,
    ):
        self.scenario = scenario
        self.backend = backend or ReplayBackend(scenario)
        self.max_steps = min(max_steps or scenario.max_steps, scenario.max_steps)
        self.observation_builder = observation_builder or ObservationBuilder(scenario, observation_config)
        self.include_bev = bool(include_bev)
        self.bev_history_length = max(1, int(bev_history_length or 1))
        self.bev_builder = BEVRasterObservationBuilder(scenario, bev_config) if self.include_bev else None
        self._bev_history: list[dict[str, Any]] = []
        self._action_space = action_space_spec()
        self.success_criterion = SuccessCriterion(scenario)
        self.step_id = 0
        self.done = False
        self.trace_hashes: list[str] = []
        self.last_raw_observation: dict[str, Any] | None = None
        self.last_observation: dict[str, Any] | None = None

    @property
    def observation_space(self) -> BoxSpaceSpec:
        return self.observation_builder.observation_space_spec()

    @property
    def action_space(self) -> BoxSpaceSpec:
        return self._action_space

    def bev_observation_space_spec(self) -> dict[str, Any] | None:
        return self.bev_builder.observation_space_spec() if self.bev_builder else None

    def sample_action(self, seed: int | None = None) -> dict[str, float]:
        values = self.action_space.sample(seed=seed)
        return self._action_from_vector(values)

    def contains_action(self, action: Action | dict[str, float] | list[float] | tuple[float, float]) -> bool:
        parsed = self._action(action)
        return self.action_space.contains([parsed.acceleration, parsed.steering])

    def clip_action(self, action: Action | dict[str, float] | list[float] | tuple[float, float]) -> dict[str, float]:
        parsed = self._action(action)
        return self._action_from_vector(self.action_space.clip([parsed.acceleration, parsed.steering]))


    @property
    def observation_schema_version(self) -> str:
        return self.observation_builder.config.schema_version

    @property
    def feature_names(self) -> list[str]:
        return list(self.observation_builder.feature_names)

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self.step_id = 0
        self.done = False
        self.trace_hashes = []
        raw = self.backend.reset()
        obs = self._observation(raw, previous_raw=None, last_action=None, events=[])
        self._reset_bev_history(obs)
        self.last_raw_observation = raw
        self.last_observation = obs
        reset_info = {
            "scenario_id": self.scenario.scenario_id,
            "seed": seed if seed is not None else self.scenario.seed,
            "observation_schema_version": self.observation_schema_version,
            "feature_names": self.feature_names,
            "observation_space_spec": self.observation_space.to_dict(),
            "action_space_spec": self.action_space.to_dict(),
            "route_goal_spec": self.success_criterion.goal_spec(),
        }
        if self.bev_builder:
            reset_info["bev_observation_space_spec"] = self.bev_builder.observation_space_spec()
        return obs, reset_info

    def step(self, action: Action | dict[str, float] | list[float] | tuple[float, float]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.done:
            obs = self.last_observation or self._observation(self.last_raw_observation or self.backend.reset(), previous_raw=None, last_action=None, events=[])
            done_reason, success_info = infer_done_reason(
                terminated=True,
                truncated=False,
                event_codes=[],
                step_id=self.step_id,
                max_steps=self.max_steps,
                scenario_max_steps=self.scenario.max_steps,
                already_done=True,
            )
            return obs, 0.0, True, False, {
                "events": [],
                "event_codes": [],
                "already_done": True,
                "done_reason": done_reason.to_dict(),
                "success_info": success_info.to_dict(),
                "trace_hash": self.final_trace_hash(),
                "observation_schema_version": self.observation_schema_version,
                "observation_space_spec": self.observation_space.to_dict(),
                "action_space_spec": self.action_space.to_dict(),
                **({"bev_observation_space_spec": self.bev_builder.observation_space_spec()} if self.bev_builder else {}),
            }

        parsed = self._action(action)
        previous_raw = self.last_raw_observation
        result = self.backend.step(parsed)
        obs = self._observation(result.observation, previous_raw=previous_raw, last_action=parsed, events=result.events)
        self._append_bev_history(obs)
        self.step_id += 1
        event_codes = [event.code for event in result.events]
        route_goal = self.success_criterion.evaluate(obs, event_codes).to_dict()
        backend_info = {**result.info, "route_goal": route_goal}
        terminated = bool(result.done) or bool(route_goal.get("reached"))
        truncated = self.step_id >= self.max_steps and not terminated
        self.done = terminated or truncated
        self.last_raw_observation = result.observation
        self.last_observation = obs
        frame_hash = self._frame_hash(result.observation, obs, parsed, result.reward, result.events)
        self.trace_hashes.append(frame_hash)
        done_reason, success_info = infer_done_reason(
            terminated=terminated,
            truncated=truncated,
            event_codes=event_codes,
            step_id=self.step_id,
            max_steps=self.max_steps,
            scenario_max_steps=self.scenario.max_steps,
            backend_info=backend_info,
        )
        info = {
            **backend_info,
            "events": [asdict(event) for event in result.events],
            "event_codes": event_codes,
            "done_reason": done_reason.to_dict(),
            "success_info": success_info.to_dict(),
            "trace_hash": frame_hash,
            "final_trace_hash": self.final_trace_hash(),
            "step_id": self.step_id,
            "observation_schema_version": self.observation_schema_version,
            "feature_names": self.feature_names,
            "observation_space_spec": self.observation_space.to_dict(),
            "action_space_spec": self.action_space.to_dict(),
            "action_within_space": self.contains_action(parsed),
            "route_goal_spec": self.success_criterion.goal_spec(),
        }
        if self.bev_builder:
            info["bev_observation_space_spec"] = self.bev_builder.observation_space_spec()
        return obs, float(result.reward), terminated, truncated, info

    def close(self) -> None:
        self.backend.close()

    def final_trace_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.trace_hashes, separators=(",", ":")).encode("utf-8")).hexdigest()


    def _action_from_vector(self, values: list[float]) -> dict[str, float]:
        return {"acceleration": float(values[0]), "steering": float(values[1])}

    def _action(self, action: Action | dict[str, float] | list[float] | tuple[float, float]) -> Action:
        if isinstance(action, Action):
            return action
        if isinstance(action, dict):
            return Action(acceleration=float(action.get("acceleration", 0.0)), steering=float(action.get("steering", 0.0)))
        if isinstance(action, (list, tuple)) and len(action) >= 2:
            return Action(acceleration=float(action[0]), steering=float(action[1]))
        raise TypeError("action must be Action, dict, or [acceleration, steering]")

    def _observation(
        self,
        raw: dict[str, Any],
        *,
        previous_raw: dict[str, Any] | None,
        last_action: Action | None,
        events: list[HealthEvent],
    ) -> dict[str, Any]:
        obs = self.observation_builder.build(
            raw,
            step=int(raw.get("step", self.step_id)),
            last_action=last_action,
            previous_observation=previous_raw,
            events=events,
        )
        if self.bev_builder:
            obs["bev"] = self.bev_builder.build(raw)
        return obs


    def _reset_bev_history(self, obs: dict[str, Any]) -> None:
        if not self.bev_builder or "bev" not in obs:
            self._bev_history = []
            return
        self._bev_history = [obs["bev"] for _ in range(self.bev_history_length)]
        self._attach_bev_history(obs)

    def _append_bev_history(self, obs: dict[str, Any]) -> None:
        if not self.bev_builder or "bev" not in obs:
            return
        if not self._bev_history:
            self._bev_history = [obs["bev"] for _ in range(self.bev_history_length)]
        else:
            self._bev_history.append(obs["bev"])
            self._bev_history = self._bev_history[-self.bev_history_length:]
            while len(self._bev_history) < self.bev_history_length:
                self._bev_history.insert(0, self._bev_history[0])
        self._attach_bev_history(obs)

    def _attach_bev_history(self, obs: dict[str, Any]) -> None:
        if not self.bev_builder or not self._bev_history:
            return
        latest = self._bev_history[-1]
        obs["bev_history"] = {
            "schema_version": "bev_history_v1",
            "history_length": self.bev_history_length,
            "shape": [self.bev_history_length] + list(latest.get("shape", [])),
            "channels": list(latest.get("channels", [])),
            "meters_per_pixel": latest.get("meters_per_pixel"),
            "ego_origin_pixel": latest.get("ego_origin_pixel"),
            "frame_steps": [int(frame["step"] if frame.get("step") is not None else obs.get("step", 0)) for frame in self._bev_history],
            "frames": [frame.get("raster", []) for frame in self._bev_history],
        }

    def _frame_hash(self, raw_observation: dict[str, Any], observation: dict[str, Any], action: Action, reward: float, events: list[HealthEvent]) -> str:
        payload = {
            "scenario_id": self.scenario.scenario_id,
            "raw_observation": self._jsonable(raw_observation),
            "observation": self._jsonable(observation),
            "action": asdict(action),
            "reward": round(float(reward), 8),
            "events": [asdict(event) for event in events],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    def _jsonable(self, value: Any) -> Any:
        if is_dataclass(value):
            return {key: self._jsonable(item) for key, item in asdict(value).items()}
        if isinstance(value, dict):
            return {key: self._jsonable(item) for key, item in value.items() if key != "raw_observation"}
        if isinstance(value, list):
            return [self._jsonable(item) for item in value]
        if isinstance(value, tuple):
            return [self._jsonable(item) for item in value]
        if isinstance(value, float):
            return round(value, 8)
        return value



