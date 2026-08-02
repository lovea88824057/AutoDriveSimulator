from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from .types import Scenario


@dataclass(frozen=True)
class RouteGoal:
    """Route-level success target for a WorldSimFlow scenario.

    The goal can be progress-based, point-based, or both. Progress is measured by
    ObservationBuilder's route_progress in [0, 1]. Point distance is measured in
    world coordinates from ego to (target_x, target_y).
    """

    enabled: bool = False
    target_progress: float | None = None
    target_x: float | None = None
    target_y: float | None = None
    success_radius: float = 3.0
    max_abs_lane_l: float | None = None
    min_step: int = 0
    require_clean: bool = True
    label: str = "route_goal"

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "RouteGoal":
        raw = scenario.metadata.get("route_goal") or scenario.metadata.get("success_criterion") or {}
        if raw is True:
            raw = {"enabled": True, "target_progress": 1.0}
        if not isinstance(raw, dict):
            return cls(enabled=False)
        enabled = bool(raw.get("enabled", True)) if raw else False
        return cls(
            enabled=enabled,
            target_progress=_optional_float(raw.get("target_progress")),
            target_x=_optional_float(raw.get("target_x")),
            target_y=_optional_float(raw.get("target_y")),
            success_radius=float(raw.get("success_radius", 3.0)),
            max_abs_lane_l=_optional_float(raw.get("max_abs_lane_l")),
            min_step=int(raw.get("min_step", 0) or 0),
            require_clean=bool(raw.get("require_clean", True)),
            label=str(raw.get("label", "route_goal")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteGoalResult:
    enabled: bool
    reached: bool
    reason: str
    label: str
    route_progress: float
    target_progress: float | None
    distance_to_goal: float | None
    success_radius: float
    lane_l: float | None
    max_abs_lane_l: float | None
    clean_required: bool
    blocking_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SuccessCriterion:
    """Evaluate route-goal success from model-facing observations."""

    def __init__(self, scenario: Scenario, goal: RouteGoal | None = None):
        self.scenario = scenario
        self.goal = goal or RouteGoal.from_scenario(scenario)

    def goal_spec(self) -> dict[str, Any]:
        return self.goal.to_dict()

    def evaluate(self, observation: dict[str, Any], event_codes: list[str] | tuple[str, ...] = ()) -> RouteGoalResult:
        goal = self.goal
        if not goal.enabled:
            return RouteGoalResult(
                enabled=False,
                reached=False,
                reason="disabled",
                label=goal.label,
                route_progress=float(observation.get("route_progress") or 0.0),
                target_progress=goal.target_progress,
                distance_to_goal=None,
                success_radius=goal.success_radius,
                lane_l=_optional_float(observation.get("ego_lane_l")),
                max_abs_lane_l=goal.max_abs_lane_l,
                clean_required=goal.require_clean,
            )

        blocking: list[str] = []
        step = int(observation.get("step", 0) or 0)
        progress = float(observation.get("route_progress") or 0.0)
        lane_l = _optional_float(observation.get("ego_lane_l"))
        distance = self._distance_to_goal(observation)
        progress_ok = goal.target_progress is not None and progress >= float(goal.target_progress)
        distance_ok = distance is not None and distance <= goal.success_radius

        if step < goal.min_step:
            blocking.append("min_step")
        if goal.require_clean and any(code in {"collision", "offroad", "non_finite_state", "stale_replay", "timeout"} for code in event_codes):
            blocking.append("failure_event")
        if goal.max_abs_lane_l is not None and lane_l is not None and abs(lane_l) > goal.max_abs_lane_l:
            blocking.append("lane_l")
        if not progress_ok and not distance_ok:
            blocking.append("target")

        reached = len(blocking) == 0
        if reached:
            reason = "route_goal_reached"
        elif "target" in blocking:
            reason = "target_not_reached"
        else:
            reason = "blocked"
        return RouteGoalResult(
            enabled=True,
            reached=reached,
            reason=reason,
            label=goal.label,
            route_progress=progress,
            target_progress=goal.target_progress,
            distance_to_goal=distance,
            success_radius=goal.success_radius,
            lane_l=lane_l,
            max_abs_lane_l=goal.max_abs_lane_l,
            clean_required=goal.require_clean,
            blocking_reasons=blocking,
        )

    def _distance_to_goal(self, observation: dict[str, Any]) -> float | None:
        if self.goal.target_x is None or self.goal.target_y is None:
            return None
        ego = observation.get("ego") or {}
        if not isinstance(ego, dict):
            return None
        return math.hypot(float(ego.get("x", 0.0)) - float(self.goal.target_x), float(ego.get("y", 0.0)) - float(self.goal.target_y))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
