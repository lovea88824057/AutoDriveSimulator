from __future__ import annotations

import math

from .types import DrivableArea, HealthEvent, Road, VehicleState


class HealthMonitor:
    """Runtime checks for abnormal termination and root-cause labeling."""

    def __init__(
        self,
        road: Road,
        max_steps: int,
        min_gap: float = 0.3,
        non_terminal_codes: set[str] | None = None,
        drivable_area: DrivableArea | None = None,
    ):
        self._road = road
        self._max_steps = max_steps
        self._min_gap = min_gap
        self._non_terminal_codes = non_terminal_codes or set()
        self._drivable_area = drivable_area

    def check(
        self,
        step: int,
        ego: VehicleState,
        actors: list[VehicleState],
        replay_exhausted: bool,
    ) -> list[HealthEvent]:
        events: list[HealthEvent] = []
        if step >= self._max_steps:
            events.append(self._event(step, "info", "timeout", "episode time limit reached"))
        if replay_exhausted:
            events.append(self._event(step, "error", "stale_replay", "background actor replay ended"))
        if not self._finite(ego):
            events.append(self._event(step, "fatal", "non_finite_state", "ego state contains NaN or Inf"))
        if self._is_offroad(ego):
            events.append(self._event(step, "error", "offroad", "ego left drivable area"))
        for actor in actors:
            if self._overlap_1d(ego.x, ego.length, actor.x, actor.length) and self._overlap_1d(
                ego.y, ego.width, actor.y, actor.width
            ):
                events.append(self._event(step, "fatal", "collision", f"ego collided with {actor.actor_id}"))
        return events

    def is_terminal(self, events: list[HealthEvent]) -> bool:
        return any(
            event.code not in self._non_terminal_codes and event.severity in {"fatal", "error"}
            for event in events
        ) or any(event.code == "timeout" for event in events)

    def _event(self, step: int, severity: str, code: str, message: str) -> HealthEvent:
        return HealthEvent(step=step, severity=severity, code=code, message=message)

    def _finite(self, state: VehicleState) -> bool:
        return all(math.isfinite(value) for value in [state.x, state.y, state.yaw, state.speed])

    def _is_offroad(self, ego: VehicleState) -> bool:
        if self._drivable_area is not None:
            return not self._drivable_area.contains(ego.x, ego.y)
        return abs(ego.y) > self._road.half_width

    def _overlap_1d(self, center_a: float, size_a: float, center_b: float, size_b: float) -> bool:
        return abs(center_a - center_b) <= (size_a + size_b) / 2.0 + self._min_gap
