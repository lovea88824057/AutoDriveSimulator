from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


FAILURE_PRIORITY = ("collision", "offroad", "non_finite_state", "stale_replay", "timeout")


@dataclass(frozen=True)
class DoneReason:
    """Episode-level termination reason used by reports, datasets, and dashboards."""

    reason: str
    terminated: bool
    truncated: bool
    success: bool
    message: str
    event_codes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SuccessInfo:
    """Human-readable success/failure metadata for reports and dashboards."""

    success: bool
    reason: str
    clean_end: bool
    reached_eval_horizon: bool
    reached_scenario_horizon: bool
    failure: bool
    failure_code: str | None
    route_goal_reached: bool = False
    route_goal_info: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_done_reason(
    *,
    terminated: bool,
    truncated: bool,
    event_codes: Iterable[str],
    step_id: int,
    max_steps: int,
    scenario_max_steps: int,
    backend_info: dict[str, Any] | None = None,
    already_done: bool = False,
) -> tuple[DoneReason, SuccessInfo]:
    """Convert low-level events into one clear episode reason.

    Current WorldSimFlow scenarios do not yet have a route destination. Therefore
    reaching only the evaluation horizon is reported as ``max_steps`` instead of
    pretending it is destination success. Real success can be supplied by backend
    info or by a future route-goal event.
    """

    backend_info = backend_info or {}
    codes = [str(code) for code in event_codes]
    code_set = set(codes)
    failure_code = next((code for code in FAILURE_PRIORITY if code in code_set), None)
    route_goal_info = backend_info.get("route_goal") if isinstance(backend_info.get("route_goal"), dict) else None
    route_goal_reached = bool(route_goal_info.get("reached")) if route_goal_info else False
    explicit_success = bool(backend_info.get("success")) or route_goal_reached or "success" in code_set
    reached_eval_horizon = bool(truncated and step_id >= max_steps)
    reached_scenario_horizon = bool(step_id >= scenario_max_steps)

    if already_done:
        reason = "already_done"
        message = "step() was called after the episode had already ended."
    elif failure_code:
        reason = failure_code
        message = _message_for(reason)
    elif explicit_success and (terminated or truncated):
        reason = "success"
        message = _message_for(reason)
    elif truncated:
        reason = "max_steps"
        message = _message_for(reason)
    elif terminated:
        reason = codes[0] if codes else "terminated"
        message = _message_for(reason)
    else:
        reason = "running"
        message = "Episode is still running."

    success = reason == "success"
    clean_end = not failure_code and not codes
    done_reason = DoneReason(
        reason=reason,
        terminated=bool(terminated),
        truncated=bool(truncated),
        success=success,
        message=message,
        event_codes=codes,
    )
    success_info = SuccessInfo(
        success=success,
        reason=reason,
        clean_end=clean_end,
        reached_eval_horizon=reached_eval_horizon,
        reached_scenario_horizon=reached_scenario_horizon,
        failure=bool(failure_code),
        failure_code=failure_code,
        route_goal_reached=route_goal_reached,
        route_goal_info=route_goal_info,
    )
    return done_reason, success_info


def _message_for(reason: str) -> str:
    messages = {
        "collision": "Episode ended because ego collided with another actor.",
        "offroad": "Episode ended because ego left the drivable area.",
        "non_finite_state": "Episode ended because a state value became non-finite.",
        "stale_replay": "Episode ended because replay data was exhausted.",
        "timeout": "Episode ended because simulation exceeded the scenario time limit.",
        "max_steps": "Episode reached the evaluation max_steps limit without an explicit route success signal.",
        "success": "Episode reached the configured route goal.",
        "terminated": "Episode ended by backend termination.",
    }
    return messages.get(reason, f"Episode ended because of {reason}.")

