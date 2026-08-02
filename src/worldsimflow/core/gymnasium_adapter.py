from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from .bev_observation import BEVRasterConfig
from .rl_env import WorldSimFlowEnv
from .scenario import ScenarioLoader
from .types import Scenario

try:  # optional dependency
    import gymnasium as gym
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - exercised by environments without gymnasium
    gym = None
    np = None

ObservationMode = Literal["state", "bev", "dict"]


class GymnasiumWorldSimFlowEnv(gym.Env if gym is not None else object):
    """Optional Gymnasium wrapper around WorldSimFlowEnv.

    This adapter keeps the core project dependency-free. Install gymnasium only
    when you want to connect PPO/RLlib/SB3-style training code.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: Scenario | str | Path,
        *,
        max_steps: int | None = None,
        observation_mode: ObservationMode = "state",
        bev_config: BEVRasterConfig | None = None,
    ):
        if gym is None or np is None:
            raise ImportError("GymnasiumWorldSimFlowEnv requires optional dependency: pip install gymnasium numpy")
        if observation_mode not in {"state", "bev", "dict"}:
            raise ValueError("observation_mode must be one of: state, bev, dict")
        self.scenario = ScenarioLoader().load(scenario) if isinstance(scenario, (str, Path)) else scenario
        self.observation_mode = observation_mode
        self._env = WorldSimFlowEnv(
            self.scenario,
            max_steps=max_steps,
            include_bev=observation_mode in {"bev", "dict"},
            bev_config=bev_config,
        )
        self.action_space = self._make_action_space()
        self.observation_space = self._make_observation_space()
        self.last_info: dict[str, Any] = {}

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, info = self._env.reset(seed=seed)
        self.last_info = info
        return self._format_observation(obs), info

    def step(self, action: Any):
        action_values = self._action_values(action)
        obs, reward, terminated, truncated, info = self._env.step(action_values)
        self.last_info = info
        return self._format_observation(obs), float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        self._env.close()

    @property
    def unwrapped_worldsimflow_env(self) -> WorldSimFlowEnv:
        return self._env

    def _make_action_space(self):
        spec = self._env.action_space
        return gym.spaces.Box(
            low=np.asarray(spec.low, dtype=np.float32),
            high=np.asarray(spec.high, dtype=np.float32),
            shape=tuple(spec.shape),
            dtype=np.float32,
        )

    def _make_observation_space(self):
        state_spec = self._env.observation_space
        state_space = gym.spaces.Box(
            low=np.full(tuple(state_spec.shape), -1.0, dtype=np.float32),
            high=np.full(tuple(state_spec.shape), 1.0, dtype=np.float32),
            shape=tuple(state_spec.shape),
            dtype=np.float32,
        )
        if self.observation_mode == "state":
            return state_space
        bev_spec = self._env.bev_observation_space_spec()
        if not bev_spec:
            raise RuntimeError("BEV observation space was not initialized")
        bev_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=tuple(bev_spec["shape"]),
            dtype=np.float32,
        )
        if self.observation_mode == "bev":
            return bev_space
        return gym.spaces.Dict({"state": state_space, "bev": bev_space})

    def _format_observation(self, obs: dict[str, Any]):
        state = np.asarray(obs.get("normalized_vector", []), dtype=np.float32)
        if self.observation_mode == "state":
            return state
        bev = np.asarray(obs["bev"]["raster"], dtype=np.float32)
        if self.observation_mode == "bev":
            return bev
        return {"state": state, "bev": bev}

    def _action_values(self, action: Any) -> list[float]:
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.shape[0] < 2:
            raise ValueError("Gymnasium action must contain acceleration and steering")
        return [float(values[0]), float(values[1])]


def gymnasium_available() -> bool:
    return gym is not None and np is not None


def make_gymnasium_env(
    scenario: Scenario | str | Path,
    *,
    max_steps: int | None = None,
    observation_mode: ObservationMode = "state",
    bev_config: BEVRasterConfig | None = None,
) -> GymnasiumWorldSimFlowEnv:
    return GymnasiumWorldSimFlowEnv(
        scenario,
        max_steps=max_steps,
        observation_mode=observation_mode,
        bev_config=bev_config,
    )
