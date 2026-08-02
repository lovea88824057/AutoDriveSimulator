from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .types import Scenario, VehicleState


@dataclass(frozen=True)
class BEVRasterConfig:
    """Lightweight ego-centric BEV raster config.

    The raster is intentionally dependency-free: it returns nested Python lists so
    RL, VLA, or world-model code can decide later whether to convert it to numpy,
    torch, or image tensors.
    """

    width: int = 64
    height: int = 64
    meters_per_pixel: float = 1.0
    ego_anchor_x: float = 0.35
    channels: tuple[str, ...] = ("drivable", "lane_center", "ego", "actor_vehicle", "actor_vru")
    schema_version: str = "bev_raster_v1"
    draw_actor_boxes: bool = True

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("BEV width/height must be positive")
        if self.meters_per_pixel <= 0.0:
            raise ValueError("meters_per_pixel must be positive")
        if not self.channels:
            raise ValueError("channels must not be empty")
        if not 0.0 <= self.ego_anchor_x <= 1.0:
            raise ValueError("ego_anchor_x must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["channels"] = list(self.channels)
        data["shape"] = [len(self.channels), self.height, self.width]
        return data


class BEVRasterObservationBuilder:
    """Build a small ego-centric BEV raster from WorldSimFlow raw observations."""

    def __init__(self, scenario: Scenario, config: BEVRasterConfig | None = None):
        self.scenario = scenario
        self.config = config or BEVRasterConfig()
        self._channel_index = {name: idx for idx, name in enumerate(self.config.channels)}

    @property
    def channels(self) -> list[str]:
        return list(self.config.channels)

    @property
    def shape(self) -> list[int]:
        return [len(self.config.channels), self.config.height, self.config.width]

    def observation_space_spec(self) -> dict[str, Any]:
        return {
            "name": self.config.schema_version,
            "shape": self.shape,
            "low": 0.0,
            "high": 1.0,
            "dtype": "float32",
            "channels": self.channels,
            "meters_per_pixel": self.config.meters_per_pixel,
            "ego_origin_pixel": self._ego_origin_pixel(),
        }

    def build(self, raw: dict[str, Any]) -> dict[str, Any]:
        ego = raw.get("ego") or self.scenario.ego
        actors = list(raw.get("actors") or [])
        grid = self._empty_grid()
        self._draw_drivable(grid, ego)
        self._draw_lane_centers(grid, ego)
        self._draw_vehicle(grid, ego, ego, "ego")
        for actor in actors:
            channel = "actor_vru" if self._is_vru(actor) else "actor_vehicle"
            self._draw_vehicle(grid, ego, actor, channel)
        return {
            "schema_version": self.config.schema_version,
            "step": int(raw.get("step", 0)),
            "shape": self.shape,
            "channels": self.channels,
            "meters_per_pixel": self.config.meters_per_pixel,
            "ego_origin_pixel": self._ego_origin_pixel(),
            "raster": grid,
        }

    def _empty_grid(self) -> list[list[list[float]]]:
        return [
            [[0.0 for _x in range(self.config.width)] for _y in range(self.config.height)]
            for _c in self.config.channels
        ]

    def _draw_drivable(self, grid: list[list[list[float]]], ego: VehicleState) -> None:
        if "drivable" not in self._channel_index:
            return
        road = self.scenario.road
        half_width = road.lane_count * road.lane_width * 0.5
        forward_min = -self.config.ego_anchor_x * self.config.width * self.config.meters_per_pixel
        forward_max = (1.0 - self.config.ego_anchor_x) * self.config.width * self.config.meters_per_pixel
        x0 = max(0.0, ego.x + forward_min)
        x1 = min(float(road.length), ego.x + forward_max)
        self._draw_world_rect(grid, "drivable", ego, x0, -half_width, x1, half_width)

    def _draw_lane_centers(self, grid: list[list[list[float]]], ego: VehicleState) -> None:
        if "lane_center" not in self._channel_index:
            return
        road = self.scenario.road
        lane_offsets = self._lane_offsets(road.lane_count, road.lane_width)
        forward_min = -self.config.ego_anchor_x * self.config.width * self.config.meters_per_pixel
        forward_max = (1.0 - self.config.ego_anchor_x) * self.config.width * self.config.meters_per_pixel
        samples = max(2, int((forward_max - forward_min) / max(0.5, self.config.meters_per_pixel)))
        for lane_y in lane_offsets:
            last_pixel: tuple[int, int] | None = None
            for idx in range(samples + 1):
                x = ego.x + forward_min + (forward_max - forward_min) * idx / samples
                if 0.0 <= x <= road.length:
                    pixel = self._world_to_pixel(ego, x, lane_y)
                    if pixel:
                        self._set(grid, "lane_center", pixel[0], pixel[1], 1.0)
                        if last_pixel:
                            self._draw_pixel_line(grid, "lane_center", last_pixel, pixel)
                        last_pixel = pixel

    def _draw_vehicle(self, grid: list[list[list[float]]], ego: VehicleState, vehicle: VehicleState, channel: str) -> None:
        if channel not in self._channel_index:
            return
        center = self._world_to_pixel(ego, float(vehicle.x), float(vehicle.y))
        if not center:
            return
        half_l = max(1, int(round(float(getattr(vehicle, "length", 4.5)) * 0.5 / self.config.meters_per_pixel)))
        half_w = max(1, int(round(float(getattr(vehicle, "width", 1.9)) * 0.5 / self.config.meters_per_pixel)))
        if not self.config.draw_actor_boxes:
            self._set(grid, channel, center[0], center[1], 1.0)
            return
        yaw_delta = float(getattr(vehicle, "yaw", 0.0)) - float(getattr(ego, "yaw", 0.0))
        cos_y = math.cos(yaw_delta)
        sin_y = math.sin(yaw_delta)
        for lx in range(-half_l, half_l + 1):
            for ly in range(-half_w, half_w + 1):
                px = int(round(center[0] + lx * cos_y - ly * sin_y))
                py = int(round(center[1] - (lx * sin_y + ly * cos_y)))
                self._set(grid, channel, px, py, 1.0)

    def _draw_world_rect(self, grid: list[list[list[float]]], channel: str, ego: VehicleState, x0: float, y0: float, x1: float, y1: float) -> None:
        step = max(0.5, self.config.meters_per_pixel)
        xs = self._sample_range(min(x0, x1), max(x0, x1), step)
        ys = self._sample_range(min(y0, y1), max(y0, y1), step)
        for x in xs:
            for y in ys:
                pixel = self._world_to_pixel(ego, x, y)
                if pixel:
                    self._set(grid, channel, pixel[0], pixel[1], 1.0)

    def _draw_pixel_line(self, grid: list[list[list[float]]], channel: str, start: tuple[int, int], end: tuple[int, int]) -> None:
        x0, y0 = start
        x1, y1 = end
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for idx in range(steps + 1):
            t = idx / steps
            self._set(grid, channel, int(round(x0 + (x1 - x0) * t)), int(round(y0 + (y1 - y0) * t)), 1.0)

    def _world_to_pixel(self, ego: VehicleState, x: float, y: float) -> tuple[int, int] | None:
        dx = x - float(ego.x)
        dy = y - float(ego.y)
        cos_yaw = math.cos(float(ego.yaw))
        sin_yaw = math.sin(float(ego.yaw))
        forward = cos_yaw * dx + sin_yaw * dy
        lateral = -sin_yaw * dx + cos_yaw * dy
        origin_x, origin_y = self._ego_origin_pixel()
        px = int(round(origin_x + forward / self.config.meters_per_pixel))
        py = int(round(origin_y - lateral / self.config.meters_per_pixel))
        if 0 <= px < self.config.width and 0 <= py < self.config.height:
            return px, py
        return None

    def _ego_origin_pixel(self) -> list[int]:
        return [int(round(self.config.width * self.config.ego_anchor_x)), self.config.height // 2]

    def _set(self, grid: list[list[list[float]]], channel: str, x: int, y: int, value: float) -> None:
        if 0 <= x < self.config.width and 0 <= y < self.config.height:
            grid[self._channel_index[channel]][y][x] = float(value)

    def _lane_offsets(self, lane_count: int, lane_width: float) -> list[float]:
        center = (lane_count - 1) / 2.0
        return [(idx - center) * lane_width for idx in range(lane_count)]

    def _sample_range(self, start: float, end: float, step: float) -> Iterable[float]:
        count = max(1, int(math.ceil((end - start) / step)))
        for idx in range(count + 1):
            yield start + (end - start) * idx / count

    def _is_vru(self, actor: VehicleState) -> bool:
        value = str(getattr(actor, "object_type", "vehicle") or "vehicle").lower()
        return value in {"pedestrian", "person", "cyclist", "bicycle", "vru"}
