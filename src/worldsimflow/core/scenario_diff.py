from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .types import Scenario, VehicleState


@dataclass(frozen=True)
class ActorDiffSummary:
    actor_id: str
    status: str
    object_type: str | None = None
    max_position_delta: float = 0.0
    max_speed_delta: float = 0.0
    changed_steps: int = 0


@dataclass(frozen=True)
class ScenarioDiffReport:
    base_scenario_id: str
    variant_scenario_id: str
    added_actors: list[str]
    removed_actors: list[str]
    changed_actors: list[ActorDiffSummary]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def changed_actor_count(self) -> int:
        return len(self.changed_actors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_scenario_id": self.base_scenario_id,
            "variant_scenario_id": self.variant_scenario_id,
            "added_actors": self.added_actors,
            "removed_actors": self.removed_actors,
            "changed_actors": [item.__dict__ for item in self.changed_actors],
            "changed_actor_count": self.changed_actor_count,
            "metadata": self.metadata,
        }


class ScenarioDiff:
    """Compare an original scenario with an intervention variant."""

    def compare(self, base: Scenario, variant: Scenario, position_eps: float = 1e-3, speed_eps: float = 1e-3) -> ScenarioDiffReport:
        base_actors = {actor.actor_id: actor for actor in base.actors}
        variant_actors = {actor.actor_id: actor for actor in variant.actors}
        added = sorted(set(variant_actors) - set(base_actors))
        removed = sorted(set(base_actors) - set(variant_actors))
        changed: list[ActorDiffSummary] = []

        for actor_id in sorted(set(base_actors) & set(variant_actors)):
            summary = self._compare_actor(actor_id, base_actors[actor_id].states, variant_actors[actor_id].states)
            if summary.max_position_delta > position_eps or summary.max_speed_delta > speed_eps or summary.changed_steps > 0:
                changed.append(summary)

        return ScenarioDiffReport(
            base_scenario_id=base.scenario_id,
            variant_scenario_id=variant.scenario_id,
            added_actors=added,
            removed_actors=removed,
            changed_actors=changed,
            metadata={"base_actor_count": len(base.actors), "variant_actor_count": len(variant.actors)},
        )

    def _compare_actor(self, actor_id: str, base_states: list[VehicleState], variant_states: list[VehicleState]) -> ActorDiffSummary:
        steps = min(len(base_states), len(variant_states))
        max_pos = 0.0
        max_speed = 0.0
        changed_steps = abs(len(base_states) - len(variant_states))
        object_type = variant_states[0].object_type if variant_states else None
        for index in range(steps):
            before = base_states[index]
            after = variant_states[index]
            pos_delta = math.hypot(after.x - before.x, after.y - before.y)
            speed_delta = abs(after.speed - before.speed)
            max_pos = max(max_pos, pos_delta)
            max_speed = max(max_speed, speed_delta)
            if pos_delta > 1e-3 or speed_delta > 1e-3 or abs(after.yaw - before.yaw) > 1e-3:
                changed_steps += 1
        return ActorDiffSummary(
            actor_id=actor_id,
            status="changed",
            object_type=object_type,
            max_position_delta=round(max_pos, 4),
            max_speed_delta=round(max_speed, 4),
            changed_steps=changed_steps,
        )