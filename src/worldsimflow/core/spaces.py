from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class BoxSpaceSpec:
    """A tiny dependency-free Box-space contract inspired by gym.spaces.Box."""

    name: str
    shape: tuple[int, ...]
    low: list[float]
    high: list[float]
    dtype: str = "float32"
    labels: list[str] = field(default_factory=list)
    schema_version: str = "box_space_v1"
    normalized_low: float = -1.0
    normalized_high: float = 1.0

    def __post_init__(self) -> None:
        size = self.flat_size
        if len(self.low) != size or len(self.high) != size:
            raise ValueError("low/high length must match shape flat size")
        if self.labels and len(self.labels) != size:
            raise ValueError("labels length must match shape flat size")
        for idx, (lo, hi) in enumerate(zip(self.low, self.high)):
            if not math.isfinite(float(lo)) or not math.isfinite(float(hi)):
                raise ValueError(f"bounds must be finite at index {idx}")
            if float(hi) <= float(lo):
                raise ValueError(f"high must be greater than low at index {idx}")

    @property
    def flat_size(self) -> int:
        size = 1
        for dim in self.shape:
            size *= int(dim)
        return size

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["shape"] = list(self.shape)
        return data

    def contains(self, value: Iterable[float], *, atol: float = 1e-9) -> bool:
        values = self._values(value)
        if len(values) != self.flat_size:
            return False
        for item, lo, hi in zip(values, self.low, self.high):
            if not math.isfinite(item):
                return False
            if item < lo - atol or item > hi + atol:
                return False
        return True

    def clip(self, value: Iterable[float]) -> list[float]:
        values = self._values(value)
        if len(values) != self.flat_size:
            raise ValueError(f"expected {self.flat_size} values, got {len(values)}")
        return [max(lo, min(hi, item)) for item, lo, hi in zip(values, self.low, self.high)]

    def sample(self, seed: int | None = None, rng: random.Random | None = None) -> list[float]:
        generator = rng or random.Random(seed)
        return [generator.uniform(lo, hi) for lo, hi in zip(self.low, self.high)]

    def normalize(self, value: Iterable[float], *, clip: bool = True) -> list[float]:
        values = self.clip(value) if clip else self._values(value)
        if len(values) != self.flat_size:
            raise ValueError(f"expected {self.flat_size} values, got {len(values)}")
        out = []
        target_span = self.normalized_high - self.normalized_low
        for item, lo, hi in zip(values, self.low, self.high):
            ratio = (item - lo) / (hi - lo)
            out.append(self.normalized_low + ratio * target_span)
        return out

    def denormalize(self, value: Iterable[float], *, clip: bool = True) -> list[float]:
        values = self._values(value)
        if len(values) != self.flat_size:
            raise ValueError(f"expected {self.flat_size} values, got {len(values)}")
        out = []
        target_span = self.normalized_high - self.normalized_low
        for item, lo, hi in zip(values, self.low, self.high):
            normalized = item
            if clip:
                normalized = max(self.normalized_low, min(self.normalized_high, normalized))
            ratio = (normalized - self.normalized_low) / target_span
            out.append(lo + ratio * (hi - lo))
        return out

    def _values(self, value: Iterable[float]) -> list[float]:
        return [float(item) for item in value]


def action_space_spec() -> BoxSpaceSpec:
    """Default closed-loop ego action space: acceleration and yaw-rate-like steering."""

    return BoxSpaceSpec(
        name="action_v1",
        shape=(2,),
        low=[-6.0, -1.0],
        high=[3.0, 1.0],
        labels=["acceleration", "steering"],
    )
