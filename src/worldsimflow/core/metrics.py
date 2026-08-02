from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .types import StepResult


@dataclass(frozen=True)
class RunMetrics:
    steps: int
    done: bool
    total_reward: float
    avg_speed: float
    min_front_gap: float | None
    event_counts: dict[str, int]


def summarize_trace(trace: list[StepResult]) -> RunMetrics:
    if not trace:
        return RunMetrics(
            steps=0,
            done=False,
            total_reward=0.0,
            avg_speed=0.0,
            min_front_gap=None,
            event_counts={},
        )

    event_counter: Counter[str] = Counter()
    speeds: list[float] = []
    front_gaps: list[float] = []
    total_reward = 0.0

    for frame in trace:
        total_reward += frame.reward
        speeds.append(float(frame.observation["ego"].speed))
        gap = frame.observation.get("front_gap")
        if gap is not None:
            front_gaps.append(float(gap))
        for event in frame.events:
            event_counter[event.code] += 1

    return RunMetrics(
        steps=len(trace),
        done=trace[-1].done,
        total_reward=round(total_reward, 6),
        avg_speed=round(sum(speeds) / len(speeds), 6),
        min_front_gap=round(min(front_gaps), 6) if front_gaps else None,
        event_counts=dict(sorted(event_counter.items())),
    )
