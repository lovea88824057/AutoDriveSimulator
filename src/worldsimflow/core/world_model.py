from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


Grid3 = list[list[list[float]]]
Grid4 = list[list[list[list[float]]]]


@dataclass(frozen=True)
class WorldModelConfig:
    """Config for a tiny dependency-free BEV forward predictor."""

    epochs: int = 40
    learning_rate: float = 0.02
    seed: int = 20260726
    train_fraction: float = 0.8
    threshold: float = 0.5
    schema_version: str = "minimal_bev_world_model_v1"


@dataclass(frozen=True)
class WorldModelSample:
    sample_id: str
    scenario_id: str
    variant_id: str | None
    step: int
    action: dict[str, float]
    obs_history: Grid4
    target_history: Grid4
    channels: list[str]
    frame_steps: list[int]

    @property
    def last_observed_frame(self) -> Grid3:
        return self.obs_history[-1]

    @property
    def target_next_frame(self) -> Grid3:
        return self.target_history[-1]


@dataclass
class MinimalBEVForwardModel:
    """Per-channel action-conditioned linear BEV predictor.

    This is intentionally tiny: for each channel and pixel it predicts the next
    occupancy value from the previous pixel value plus ego action. The point is to
    prove the world-model data loop, not to compete with neural BEV models.
    """

    channels: list[str]
    weights_last: list[float] = field(default_factory=list)
    weights_acceleration: list[float] = field(default_factory=list)
    weights_steering: list[float] = field(default_factory=list)
    bias: list[float] = field(default_factory=list)

    @classmethod
    def initialize(cls, channels: list[str]) -> "MinimalBEVForwardModel":
        count = len(channels)
        return cls(
            channels=list(channels),
            weights_last=[1.0 for _ in range(count)],
            weights_acceleration=[0.0 for _ in range(count)],
            weights_steering=[0.0 for _ in range(count)],
            bias=[0.0 for _ in range(count)],
        )

    def predict_next_frame(self, last_frame: Grid3, action: dict[str, float]) -> Grid3:
        acc = float(action.get("acceleration", 0.0))
        steering = float(action.get("steering", 0.0))
        out: Grid3 = []
        for channel_index, channel_grid in enumerate(last_frame):
            predicted_channel = []
            for row in channel_grid:
                predicted_row = []
                for value in row:
                    raw = (
                        self.weights_last[channel_index] * float(value)
                        + self.weights_acceleration[channel_index] * acc
                        + self.weights_steering[channel_index] * steering
                        + self.bias[channel_index]
                    )
                    predicted_row.append(_clip01(raw))
                predicted_channel.append(predicted_row)
            out.append(predicted_channel)
        return out

    def predict_history(self, sample: WorldModelSample) -> Grid4:
        predicted_next = self.predict_next_frame(sample.last_observed_frame, sample.action)
        return sample.obs_history[1:] + [predicted_next]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MinimalWorldModelTrainer:
    """Train and evaluate a tiny BEV world-model smoke demo."""

    def __init__(self, config: WorldModelConfig | None = None):
        self.config = config or WorldModelConfig()

    def load_samples(self, path: str | Path, *, max_samples: int | None = None) -> list[WorldModelSample]:
        rows = []
        for line_index, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            sample = self._sample_from_row(row, line_index)
            if sample is not None:
                rows.append(sample)
            if max_samples is not None and len(rows) >= max_samples:
                break
        if not rows:
            raise ValueError("No BEV history transitions found. Export with --include-bev first.")
        return rows

    def train(self, samples: list[WorldModelSample]) -> dict[str, Any]:
        rng = random.Random(self.config.seed)
        shuffled = list(samples)
        rng.shuffle(shuffled)
        split = max(1, min(len(shuffled), int(round(len(shuffled) * self.config.train_fraction))))
        if len(shuffled) > 1 and split == len(shuffled):
            split = len(shuffled) - 1
        train_samples = shuffled[:split]
        eval_samples = shuffled[split:] or shuffled[:]
        model = MinimalBEVForwardModel.initialize(samples[0].channels)
        baseline_train = self.evaluate_persistence(train_samples)
        baseline_eval = self.evaluate_persistence(eval_samples)
        loss_curve = []

        for epoch in range(self.config.epochs):
            rng.shuffle(train_samples)
            total_loss = 0.0
            total_count = 0
            for sample in train_samples:
                loss, count = self._sgd_step(model, sample)
                total_loss += loss
                total_count += count
            train_metrics = self.evaluate(model, train_samples)
            eval_metrics = self.evaluate(model, eval_samples)
            loss_curve.append(
                {
                    "epoch": epoch + 1,
                    "sgd_mse": round(total_loss / max(1, total_count), 8),
                    "train_mse": train_metrics["mse"],
                    "eval_mse": eval_metrics["mse"],
                    "eval_iou": eval_metrics["occupancy_iou"],
                }
            )

        final_train = self.evaluate(model, train_samples)
        final_eval = self.evaluate(model, eval_samples)
        return {
            "world_model_demo": "ok",
            "schema_version": self.config.schema_version,
            "sample_count": len(samples),
            "train_sample_count": len(train_samples),
            "eval_sample_count": len(eval_samples),
            "channels": samples[0].channels,
            "history_shape": _history_shape(samples[0].obs_history),
            "input_contract": "bev_history_t + action_t -> next_bev_history_t_plus_1",
            "model_type": "per_channel_action_conditioned_linear_predictor",
            "config": asdict(self.config),
            "baseline_persistence_train": baseline_train,
            "baseline_persistence_eval": baseline_eval,
            "final_train": final_train,
            "final_eval": final_eval,
            "loss_curve": loss_curve,
            "model": model.to_dict(),
            "examples": [self.example_payload(model, sample) for sample in eval_samples[: min(3, len(eval_samples))]],
        }

    def evaluate(self, model: MinimalBEVForwardModel, samples: list[WorldModelSample]) -> dict[str, Any]:
        mse_sum = 0.0
        mae_sum = 0.0
        count = 0
        intersection = 0
        union = 0
        channel_totals = {channel: {"mse_sum": 0.0, "count": 0} for channel in model.channels}
        for sample in samples:
            predicted = model.predict_next_frame(sample.last_observed_frame, sample.action)
            target = sample.target_next_frame
            metrics = _grid_metrics(predicted, target, threshold=self.config.threshold)
            mse_sum += metrics["mse_sum"]
            mae_sum += metrics["mae_sum"]
            count += metrics["count"]
            intersection += metrics["intersection"]
            union += metrics["union"]
            for channel_index, channel in enumerate(model.channels):
                channel_mse, channel_count = _channel_mse(predicted[channel_index], target[channel_index])
                channel_totals[channel]["mse_sum"] += channel_mse
                channel_totals[channel]["count"] += channel_count
        return {
            "mse": round(mse_sum / max(1, count), 8),
            "mae": round(mae_sum / max(1, count), 8),
            "occupancy_iou": round(intersection / max(1, union), 8),
            "positive_pixel_ratio": self._positive_ratio(samples),
            "channel_mse": {
                channel: round(values["mse_sum"] / max(1, values["count"]), 8)
                for channel, values in channel_totals.items()
            },
        }

    def evaluate_persistence(self, samples: list[WorldModelSample]) -> dict[str, Any]:
        mse_sum = 0.0
        mae_sum = 0.0
        count = 0
        intersection = 0
        union = 0
        for sample in samples:
            metrics = _grid_metrics(sample.last_observed_frame, sample.target_next_frame, threshold=self.config.threshold)
            mse_sum += metrics["mse_sum"]
            mae_sum += metrics["mae_sum"]
            count += metrics["count"]
            intersection += metrics["intersection"]
            union += metrics["union"]
        return {
            "mse": round(mse_sum / max(1, count), 8),
            "mae": round(mae_sum / max(1, count), 8),
            "occupancy_iou": round(intersection / max(1, union), 8),
        }

    def example_payload(self, model: MinimalBEVForwardModel, sample: WorldModelSample) -> dict[str, Any]:
        predicted_next = model.predict_next_frame(sample.last_observed_frame, sample.action)
        metrics = _grid_metrics(predicted_next, sample.target_next_frame, threshold=self.config.threshold)
        return {
            "sample_id": sample.sample_id,
            "scenario_id": sample.scenario_id,
            "variant_id": sample.variant_id,
            "step": sample.step,
            "action": sample.action,
            "channels": sample.channels,
            "frame_steps": sample.frame_steps,
            "metrics": {
                "mse": round(metrics["mse_sum"] / max(1, metrics["count"]), 8),
                "mae": round(metrics["mae_sum"] / max(1, metrics["count"]), 8),
                "occupancy_iou": round(metrics["intersection"] / max(1, metrics["union"]), 8),
            },
            "last_observed_frame": sample.last_observed_frame,
            "target_next_frame": sample.target_next_frame,
            "predicted_next_frame": predicted_next,
        }

    def _sgd_step(self, model: MinimalBEVForwardModel, sample: WorldModelSample) -> tuple[float, int]:
        acc = float(sample.action.get("acceleration", 0.0))
        steering = float(sample.action.get("steering", 0.0))
        lr = float(self.config.learning_rate)
        total_loss = 0.0
        total_count = 0
        last = sample.last_observed_frame
        target = sample.target_next_frame
        for channel_index in range(len(model.channels)):
            grad_last = 0.0
            grad_acc = 0.0
            grad_steering = 0.0
            grad_bias = 0.0
            count = 0
            for y, row in enumerate(last[channel_index]):
                for x, value in enumerate(row):
                    pred = _clip01(
                        model.weights_last[channel_index] * float(value)
                        + model.weights_acceleration[channel_index] * acc
                        + model.weights_steering[channel_index] * steering
                        + model.bias[channel_index]
                    )
                    truth = float(target[channel_index][y][x])
                    error = pred - truth
                    total_loss += error * error
                    grad = 2.0 * error
                    grad_last += grad * float(value)
                    grad_acc += grad * acc
                    grad_steering += grad * steering
                    grad_bias += grad
                    count += 1
            scale = lr / max(1, count)
            model.weights_last[channel_index] -= scale * grad_last
            model.weights_acceleration[channel_index] -= scale * grad_acc
            model.weights_steering[channel_index] -= scale * grad_steering
            model.bias[channel_index] -= scale * grad_bias
            total_count += count
        return total_loss, total_count

    def _sample_from_row(self, row: dict[str, Any], line_index: int) -> WorldModelSample | None:
        obs_history = row.get("obs", {}).get("bev_history")
        target_history = row.get("next_obs", {}).get("bev_history")
        if not obs_history or not target_history:
            return None
        return WorldModelSample(
            sample_id=f"line_{line_index}",
            scenario_id=str(row.get("scenario_id", "unknown")),
            variant_id=row.get("variant_id"),
            step=int(row.get("step", line_index)),
            action={
                "acceleration": float(row.get("action", {}).get("acceleration", 0.0)),
                "steering": float(row.get("action", {}).get("steering", 0.0)),
            },
            obs_history=_float_grid4(obs_history.get("frames", [])),
            target_history=_float_grid4(target_history.get("frames", [])),
            channels=list(obs_history.get("channels", [])),
            frame_steps=list(obs_history.get("frame_steps", [])),
        )

    def _positive_ratio(self, samples: list[WorldModelSample]) -> float:
        positive = 0
        count = 0
        for sample in samples:
            for channel in sample.target_next_frame:
                for row in channel:
                    for value in row:
                        positive += int(float(value) >= self.config.threshold)
                        count += 1
        return round(positive / max(1, count), 8)


def _float_grid4(value: Any) -> Grid4:
    return [[[[float(item) for item in row] for row in channel] for channel in frame] for frame in value]


def _history_shape(history: Grid4) -> list[int]:
    if not history:
        return [0, 0, 0, 0]
    return [len(history), len(history[0]), len(history[0][0]), len(history[0][0][0])]


def _grid_metrics(predicted: Grid3, target: Grid3, *, threshold: float) -> dict[str, Any]:
    mse_sum = 0.0
    mae_sum = 0.0
    count = 0
    intersection = 0
    union = 0
    for channel_index, channel in enumerate(predicted):
        for y, row in enumerate(channel):
            for x, value in enumerate(row):
                pred = float(value)
                truth = float(target[channel_index][y][x])
                error = pred - truth
                mse_sum += error * error
                mae_sum += abs(error)
                pred_on = pred >= threshold
                truth_on = truth >= threshold
                intersection += int(pred_on and truth_on)
                union += int(pred_on or truth_on)
                count += 1
    return {"mse_sum": mse_sum, "mae_sum": mae_sum, "count": count, "intersection": intersection, "union": union}


def _channel_mse(predicted: list[list[float]], target: list[list[float]]) -> tuple[float, int]:
    mse_sum = 0.0
    count = 0
    for y, row in enumerate(predicted):
        for x, value in enumerate(row):
            error = float(value) - float(target[y][x])
            mse_sum += error * error
            count += 1
    return mse_sum, count


def _clip01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    if not math.isfinite(value):
        return 0.0
    return value
