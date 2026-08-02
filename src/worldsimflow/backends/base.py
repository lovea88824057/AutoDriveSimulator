from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from worldsimflow.core.types import Action, HealthEvent, Scenario


@dataclass(frozen=True)
class BackendStepResult:
    observation: dict
    reward: float
    done: bool
    events: list[HealthEvent]
    info: dict


class SimulationBackend(Protocol):
    """Common backend interface for replay, lightweight closed-loop, and learned-world rollouts."""

    @property
    def scenario(self) -> Scenario:
        ...

    def reset(self) -> dict:
        ...

    def step(self, action: Action) -> BackendStepResult:
        ...

    def close(self) -> None:
        ...

