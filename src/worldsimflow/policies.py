from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .core.types import Action


class LaneKeepPolicy:
    """Simple baseline policy for closed-loop evaluation."""

    def __init__(self, target_speed: float = 8.0):
        self.target_speed = target_speed

    def act(self, observation: dict) -> Action:
        ego = observation["ego"]
        front_gap = observation["front_gap"]
        acceleration = 1.0 if ego.speed < self.target_speed else -0.5
        if front_gap is not None and front_gap < 10.0:
            acceleration = -2.0
        steering = -0.8 * observation["lane_center_offset"]
        return Action(acceleration=acceleration, steering=steering)


@dataclass(frozen=True)
class PolicyAction:
    """Action plus a stable label for evaluation summaries."""

    name: str
    action: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "action": dict(self.action)}


class EvaluationPolicy(Protocol):
    name: str

    def reset(self, seed: int | None = None) -> None: ...

    def act(self, observation: dict[str, Any]) -> PolicyAction: ...

    def observe(
        self,
        observation: dict[str, Any],
        action: PolicyAction,
        reward: float,
        next_observation: dict[str, Any],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None: ...

    def summary(self) -> dict[str, Any]: ...


DISCRETE_ACTIONS: list[PolicyAction] = [
    PolicyAction("brake", {"acceleration": -2.0, "steering": 0.0}),
    PolicyAction("keep", {"acceleration": 0.0, "steering": 0.0}),
    PolicyAction("accelerate", {"acceleration": 1.5, "steering": 0.0}),
    PolicyAction("left", {"acceleration": 0.0, "steering": 0.25}),
    PolicyAction("right", {"acceleration": 0.0, "steering": -0.25}),
]


class RuleEvaluationPolicy:
    """Deterministic lane-centering and target-speed baseline."""

    name = "rule"

    def reset(self, seed: int | None = None) -> None:
        return None

    def act(self, observation: dict[str, Any]) -> PolicyAction:
        speed = float(observation.get("ego_speed") or 0.0)
        front_gap = observation.get("front_gap")
        lane_l = float(observation.get("ego_lane_l") or observation.get("lane_center_offset") or 0.0)
        target_speed = 8.0
        acceleration = max(-2.5, min(2.0, (target_speed - speed) * 0.45))
        label = "accelerate" if acceleration > 0.2 else "brake" if acceleration < -0.2 else "keep"
        if front_gap is not None and float(front_gap) < 10.0:
            acceleration = min(acceleration, -1.5)
            label = "brake"
        steering = max(-0.35, min(0.35, -lane_l * 0.16))
        if abs(steering) > 0.05:
            label = f"{label}+{'left' if steering > 0 else 'right'}"
        return PolicyAction(label, {"acceleration": acceleration, "steering": steering})

    def observe(self, *args: Any, **kwargs: Any) -> None:
        return None

    def summary(self) -> dict[str, Any]:
        return {"type": "rule", "description": "target-speed + lane-centering baseline"}


class RandomEvaluationPolicy:
    """Seeded random policy over a small discrete action set."""

    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng.seed(seed)

    def act(self, observation: dict[str, Any]) -> PolicyAction:
        return self.rng.choice(DISCRETE_ACTIONS)

    def observe(self, *args: Any, **kwargs: Any) -> None:
        return None

    def summary(self) -> dict[str, Any]:
        return {"type": "random", "action_count": len(DISCRETE_ACTIONS)}


class MinimalQPolicy:
    """Tiny online Q-learning policy using normalized_vector buckets."""

    name = "minimal-q"

    def __init__(self, seed: int | None = None, epsilon: float = 0.20, alpha: float = 0.35, gamma: float = 0.90):
        self.seed = seed
        self.rng = random.Random(seed)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.q_table: dict[str, list[float]] = defaultdict(lambda: [0.0 for _ in DISCRETE_ACTIONS])

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng.seed(seed)

    def act(self, observation: dict[str, Any]) -> PolicyAction:
        state_key = self.discretize(observation)
        values = self.q_table[state_key]
        if self.rng.random() < self.epsilon:
            index = self.rng.randrange(len(DISCRETE_ACTIONS))
        else:
            best = max(values)
            choices = [idx for idx, value in enumerate(values) if value == best]
            index = self.rng.choice(choices)
        return DISCRETE_ACTIONS[index]

    def observe(
        self,
        observation: dict[str, Any],
        action: PolicyAction,
        reward: float,
        next_observation: dict[str, Any],
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> None:
        state_key = self.discretize(observation)
        next_key = self.discretize(next_observation)
        action_index = self._action_index(action.name)
        bootstrap = 0.0 if terminated else max(self.q_table[next_key])
        target = float(reward) + self.gamma * bootstrap
        self.q_table[state_key][action_index] += self.alpha * (target - self.q_table[state_key][action_index])

    def summary(self) -> dict[str, Any]:
        learned = {}
        for state, values in sorted(self.q_table.items()):
            best_index = max(range(len(values)), key=lambda idx: values[idx])
            learned[state] = {
                "best_action": DISCRETE_ACTIONS[best_index].name,
                "q_values": {DISCRETE_ACTIONS[idx].name: round(value, 6) for idx, value in enumerate(values)},
            }
        return {
            "type": "minimal-q",
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "q_state_count": len(self.q_table),
            "learned_policy": learned,
        }

    def discretize(self, observation: dict[str, Any]) -> str:
        feature_names = observation.get("feature_names") or []
        values = observation.get("normalized_vector") or []
        lookup = {name: float(values[idx]) for idx, name in enumerate(feature_names) if idx < len(values)}
        speed = self._bucket(lookup.get("ego_speed", 0.0), [-0.70, -0.40, 0.0], ["very_slow", "slow", "target", "fast"])
        lane = self._bucket(lookup.get("ego_lane_l", 0.0), [-0.18, 0.18], ["right", "center", "left"])
        gap = self._bucket(lookup.get("front_gap", 1.0), [-0.96, -0.90, -0.70], ["danger", "close", "medium", "open"])
        return f"speed={speed}|lane={lane}|gap={gap}"

    def _bucket(self, value: float, thresholds: list[float], labels: list[str]) -> str:
        for threshold, label in zip(thresholds, labels):
            if value < threshold:
                return label
        return labels[-1]

    def _action_index(self, action_name: str) -> int:
        for index, item in enumerate(DISCRETE_ACTIONS):
            if item.name == action_name:
                return index
        return 1


def make_evaluation_policy(name: str, seed: int | None = None) -> EvaluationPolicy:
    if name == "rule":
        return RuleEvaluationPolicy()
    if name == "random":
        return RandomEvaluationPolicy(seed=seed)
    if name in {"minimal-q", "minimal_q", "q"}:
        return MinimalQPolicy(seed=seed)
    raise ValueError(f"Unsupported evaluation policy: {name}")


def policy_action_to_action(value: PolicyAction | Action | dict[str, float] | list[float] | tuple[float, float]) -> dict[str, float]:
    if isinstance(value, PolicyAction):
        return dict(value.action)
    if isinstance(value, Action):
        return asdict(value)
    if isinstance(value, dict):
        return {"acceleration": float(value.get("acceleration", 0.0)), "steering": float(value.get("steering", 0.0))}
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return {"acceleration": float(value[0]), "steering": float(value[1])}
    raise TypeError("Unsupported policy action payload")
