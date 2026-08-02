from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .bev_observation import BEVRasterConfig
from .rl_env import WorldSimFlowEnv
from .types import Action, Scenario

PolicyFn = Callable[[dict[str, Any]], Action | dict[str, float] | list[float] | tuple[float, float]]


@dataclass(frozen=True)
class TransitionExportConfig:
    """Configuration for exporting model-facing transition JSONL datasets."""

    dataset_schema_version: str = "transition_v1"
    episodes: int = 1
    max_steps: int | None = None
    seed: int | None = None
    include_full_observation: bool = True
    float_digits: int = 8
    include_bev: bool = False
    bev_history_length: int = 1
    bev_width: int = 64
    bev_height: int = 64
    bev_meters_per_pixel: float = 1.0


class TransitionDatasetExporter:
    """Export obs/action/reward/next_obs transitions from WorldSimFlowEnv.

    The JSONL format is intentionally framework-neutral: RL, imitation learning and
    world-model code can all read the same transition records without importing
    WorldSimFlow internals.
    """

    def __init__(self, scenario: Scenario, config: TransitionExportConfig | None = None):
        self.scenario = scenario
        self.config = config or TransitionExportConfig()

    def export(self, output: str | Path, policy: PolicyFn) -> dict[str, Any]:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        transition_count = 0
        reward_curve: list[float] = []
        final_trace_hashes: list[str] = []
        observation_schema_version = None
        feature_names: list[str] = []
        observation_space_spec = None
        action_space_spec = None
        bev_observation_space_spec = None

        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            for episode in range(self.config.episodes):
                env = WorldSimFlowEnv(
                    self.scenario,
                    max_steps=self.config.max_steps,
                    include_bev=self.config.include_bev,
                    bev_config=BEVRasterConfig(
                        width=self.config.bev_width,
                        height=self.config.bev_height,
                        meters_per_pixel=self.config.bev_meters_per_pixel,
                    ) if self.config.include_bev else None,
                    bev_history_length=self.config.bev_history_length,
                )
                episode_seed = None if self.config.seed is None else self.config.seed + episode
                obs, reset_info = env.reset(seed=episode_seed)
                observation_schema_version = reset_info.get("observation_schema_version", observation_schema_version)
                feature_names = list(reset_info.get("feature_names", feature_names))
                observation_space_spec = reset_info.get("observation_space_spec", observation_space_spec)
                action_space_spec = reset_info.get("action_space_spec", action_space_spec)
                bev_observation_space_spec = reset_info.get("bev_observation_space_spec", bev_observation_space_spec)
                total_reward = 0.0
                try:
                    max_steps = self.config.max_steps or self.scenario.max_steps
                    for step in range(max_steps):
                        action = policy(obs)
                        next_obs, reward, terminated, truncated, info = env.step(action)
                        total_reward += float(reward)
                        record = self._transition_record(
                            episode=episode,
                            step=step,
                            obs=obs,
                            action=env._action(action),
                            reward=float(reward),
                            next_obs=next_obs,
                            terminated=terminated,
                            truncated=truncated,
                            info=info,
                        )
                        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
                        handle.write("\n")
                        transition_count += 1
                        obs = next_obs
                        if terminated or truncated:
                            break
                    final_trace_hashes.append(env.final_trace_hash())
                    reward_curve.append(round(total_reward, 6))
                finally:
                    env.close()

        report = {
            "dataset_schema_version": self.config.dataset_schema_version,
            "scenario_id": self.scenario.scenario_id,
            "output": str(output_path),
            "episodes": self.config.episodes,
            "max_steps": self.config.max_steps or self.scenario.max_steps,
            "transition_count": transition_count,
            "observation_schema_version": observation_schema_version,
            "feature_names": feature_names,
            "feature_count": len(feature_names),
            "observation_space_spec": observation_space_spec,
            "action_space_spec": action_space_spec,
            "include_bev": self.config.include_bev,
            "bev_history_length": self.config.bev_history_length if self.config.include_bev else 0,
            "bev_observation_space_spec": bev_observation_space_spec,
            "reward_curve": reward_curve,
            "final_trace_hashes": final_trace_hashes,
        }
        return report

    def _transition_record(
        self,
        *,
        episode: int,
        step: int,
        obs: dict[str, Any],
        action: Action,
        reward: float,
        next_obs: dict[str, Any],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "record_type": "transition",
            "dataset_schema_version": self.config.dataset_schema_version,
            "scenario_id": self.scenario.scenario_id,
            "episode": episode,
            "step": step,
            "obs": self._observation_payload(obs),
            "action": asdict(action),
            "reward": round(reward, self.config.float_digits),
            "next_obs": self._observation_payload(next_obs),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "event_codes": list(info.get("event_codes", [])),
            "done_reason": self._jsonable(info.get("done_reason", {})),
            "success_info": self._jsonable(info.get("success_info", {})),
            "reward_breakdown": self._jsonable(info.get("reward_breakdown", {})),
            "cost_info": self._jsonable(info.get("cost_info", {})),
            "trace_hash": info.get("trace_hash"),
            "final_trace_hash_so_far": info.get("final_trace_hash"),
        }

    def _observation_payload(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self.config.include_full_observation:
            return self._jsonable(obs)
        payload = {
            "schema_version": obs.get("schema_version"),
            "step": obs.get("step"),
            "feature_names": list(obs.get("feature_names", [])),
            "state_vector": self._jsonable(obs.get("state_vector", [])),
            "normalized_vector": self._jsonable(obs.get("normalized_vector", [])),
            "normalized_range": self._jsonable(obs.get("normalized_range", [-1.0, 1.0])),
        }
        if self.config.include_bev:
            if "bev" in obs:
                payload["bev"] = self._jsonable(obs.get("bev"))
            if "bev_history" in obs:
                payload["bev_history"] = self._jsonable(obs.get("bev_history"))
        return payload

    def _jsonable(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): self._jsonable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._jsonable(item) for item in value]
        if isinstance(value, tuple):
            return [self._jsonable(item) for item in value]
        if isinstance(value, float):
            return round(value, self.config.float_digits)
        return value
