from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .types import HealthEvent


@dataclass(frozen=True)
class RewardBreakdown:
    """Explain one scalar reward as named components."""

    speed_reward: float = 0.0
    lane_penalty: float = 0.0
    gap_penalty: float = 0.0
    collision_penalty: float = 0.0
    offroad_penalty: float = 0.0
    stale_replay_penalty: float = 0.0

    @property
    def total(self) -> float:
        return round(
            self.speed_reward
            + self.lane_penalty
            + self.gap_penalty
            + self.collision_penalty
            + self.offroad_penalty
            + self.stale_replay_penalty,
            8,
        )

    def to_dict(self) -> dict[str, float]:
        data = asdict(self)
        data["total"] = self.total
        return data


@dataclass(frozen=True)
class CostInfo:
    """Cost-style safety signals for RL evaluation and diagnostics."""

    collision_cost: float = 0.0
    offroad_cost: float = 0.0
    stale_replay_cost: float = 0.0
    close_gap_cost: float = 0.0

    @property
    def total_cost(self) -> float:
        return round(self.collision_cost + self.offroad_cost + self.stale_replay_cost + self.close_gap_cost, 8)

    def to_dict(self) -> dict[str, float]:
        data = asdict(self)
        data["total_cost"] = self.total_cost
        return data


def compute_reward_breakdown(obs: dict[str, Any], events: Iterable[HealthEvent | dict[str, Any]]) -> tuple[RewardBreakdown, CostInfo]:
    ego = obs.get("ego")
    if isinstance(ego, dict):
        speed = float(ego.get("speed", 0.0) or 0.0)
    else:
        speed = float(getattr(ego, "speed", 0.0) if ego is not None else 0.0)
    lane_offset = float(obs.get("lane_center_offset") or 0.0)
    front_gap = obs.get("front_gap")
    event_codes = _event_codes(events)

    speed_reward = speed * 0.1
    lane_penalty = -abs(lane_offset) * 0.2
    gap_penalty = 0.0
    close_gap_cost = 0.0
    if front_gap is not None and float(front_gap) < 8.0:
        gap_penalty = -(8.0 - float(front_gap)) * 0.5
        close_gap_cost = 1.0

    collision_penalty = -100.0 if "collision" in event_codes else 0.0
    offroad_penalty = -50.0 if "offroad" in event_codes else 0.0
    stale_replay_penalty = 0.0

    breakdown = RewardBreakdown(
        speed_reward=round(speed_reward, 8),
        lane_penalty=round(lane_penalty, 8),
        gap_penalty=round(gap_penalty, 8),
        collision_penalty=collision_penalty,
        offroad_penalty=offroad_penalty,
        stale_replay_penalty=stale_replay_penalty,
    )
    cost = CostInfo(
        collision_cost=1.0 if "collision" in event_codes else 0.0,
        offroad_cost=1.0 if "offroad" in event_codes else 0.0,
        stale_replay_cost=1.0 if "stale_replay" in event_codes else 0.0,
        close_gap_cost=close_gap_cost,
    )
    return breakdown, cost


def _event_codes(events: Iterable[HealthEvent | dict[str, Any]]) -> set[str]:
    codes: set[str] = set()
    for event in events:
        if isinstance(event, HealthEvent):
            codes.add(event.code)
        elif isinstance(event, dict) and event.get("code") is not None:
            codes.add(str(event["code"]))
    return codes
