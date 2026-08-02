from __future__ import annotations

from worldsimflow.backends.base import BackendStepResult
from worldsimflow.core.sim import MiniDrivingSimulator
from worldsimflow.core.types import Action, Scenario


class ReplayBackend:
    """Deterministic lightweight LogSim backend used for fast local validation."""

    def __init__(self, scenario: Scenario):
        self._scenario = scenario
        self._sim = MiniDrivingSimulator(scenario)

    @property
    def scenario(self) -> Scenario:
        return self._scenario

    def reset(self) -> dict:
        return self._sim.reset()

    def step(self, action: Action) -> BackendStepResult:
        observation, reward, done, events = self._sim.step(action)
        return BackendStepResult(
            observation=observation,
            reward=reward,
            done=done,
            events=events,
            info={"backend": "replay", **self._sim.reward_info()},
        )

    def close(self) -> None:
        return None
