from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, is_dataclass
from typing import Protocol

from worldsimflow.backends import ReplayBackend, SimulationBackend

from .types import Action, Scenario, StepResult


class Policy(Protocol):
    def act(self, observation: dict) -> Action:
        ...


class DeterministicFlowController:
    """Owns the closed-loop order: observe, plan, apply, monitor, trace."""

    def __init__(self, scenario: Scenario, policy: Policy, backend: SimulationBackend | None = None):
        self.scenario = scenario
        self.policy = policy
        self.backend = backend or ReplayBackend(scenario)
        self.rng = random.Random(scenario.seed)
        self.trace: list[StepResult] = []

    def run(self, steps: int | None = None) -> list[StepResult]:
        max_steps = min(steps or self.scenario.max_steps, self.scenario.max_steps)
        observation = self.backend.reset()
        self.trace = []
        for _ in range(max_steps):
            action = self.policy.act(observation)
            backend_result = self.backend.step(action)
            observation = backend_result.observation
            trace_hash = self._hash_frame(observation, action, backend_result.reward, backend_result.events)
            result = StepResult(
                step=observation["step"],
                observation=observation,
                reward=backend_result.reward,
                done=backend_result.done,
                events=backend_result.events,
                trace_hash=trace_hash,
                action=action,
            )
            self.trace.append(result)
            if backend_result.done:
                break
        return self.trace

    def close(self) -> None:
        self.backend.close()

    def final_trace_hash(self) -> str:
        payload = [item.trace_hash for item in self.trace]
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()

    def _hash_frame(self, observation: dict, action: Action, reward: float, events: list) -> str:
        payload = {
            "scenario_id": self.scenario.scenario_id,
            "backend": self.backend.__class__.__name__,
            "observation": self._jsonable(observation),
            "action": self._jsonable(action),
            "reward": round(reward, 8),
            "events": self._jsonable(events),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _jsonable(self, value):
        if is_dataclass(value):
            return {k: self._jsonable(v) for k, v in asdict(value).items()}
        if isinstance(value, dict):
            return {k: self._jsonable(v) for k, v in value.items() if k != "raw_observation"}
        if isinstance(value, list):
            return [self._jsonable(v) for v in value]
        if isinstance(value, float):
            return round(value, 8)
        return value
