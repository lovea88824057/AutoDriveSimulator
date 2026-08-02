from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from .types import Scenario, VehicleState


@dataclass(frozen=True)
class LanePose:
    lane_id: str
    s: float
    l: float
    x: float
    y: float
    heading: float


@dataclass(frozen=True)
class FrenetProjection:
    lane_id: str
    s: float
    l: float
    x: float
    y: float
    heading: float
    distance: float
    segment_index: int


@dataclass(frozen=True)
class LaneSegment:
    lane_id: str
    polyline: list[tuple[float, float]]
    source: str
    source_feature_ids: tuple[str, ...]
    length: float
    cumulative_s: tuple[float, ...]

    @classmethod
    def create(
        cls,
        lane_id: str,
        polyline: Iterable[tuple[float, float]],
        source: str,
        source_feature_ids: Iterable[str] = (),
    ) -> "LaneSegment":
        cleaned = _clean_polyline(list(polyline))
        cumulative = _cumulative_s(cleaned)
        return cls(
            lane_id=lane_id,
            polyline=cleaned,
            source=source,
            source_feature_ids=tuple(source_feature_ids),
            length=cumulative[-1] if cumulative else 0.0,
            cumulative_s=tuple(cumulative),
        )

    def project(self, x: float, y: float) -> FrenetProjection | None:
        if len(self.polyline) < 2 or self.length <= 0.0:
            return None
        best: tuple[float, FrenetProjection] | None = None
        for index, (a, b) in enumerate(zip(self.polyline, self.polyline[1:])):
            ax, ay = a
            bx, by = b
            vx = bx - ax
            vy = by - ay
            seg_len_sq = vx * vx + vy * vy
            if seg_len_sq <= 1e-12:
                continue
            raw_t = ((x - ax) * vx + (y - ay) * vy) / seg_len_sq
            t = max(0.0, min(1.0, raw_t))
            px = ax + vx * t
            py = ay + vy * t
            heading = math.atan2(vy, vx)
            dx = x - px
            dy = y - py
            distance = math.hypot(dx, dy)
            seg_len = math.sqrt(seg_len_sq)
            signed_l = (-math.sin(heading) * dx) + (math.cos(heading) * dy)
            projection = FrenetProjection(
                lane_id=self.lane_id,
                s=self.cumulative_s[index] + t * seg_len,
                l=signed_l,
                x=px,
                y=py,
                heading=heading,
                distance=distance,
                segment_index=index,
            )
            if best is None or distance < best[0]:
                best = (distance, projection)
        return best[1] if best else None

    def sample_pose(self, s: float, l: float = 0.0) -> LanePose:
        if len(self.polyline) < 2 or self.length <= 0.0:
            raise ValueError(f"LaneSegment {self.lane_id!r} has no valid polyline")
        target_s = max(0.0, min(float(s), self.length))
        for index, (a, b) in enumerate(zip(self.polyline, self.polyline[1:])):
            start_s = self.cumulative_s[index]
            end_s = self.cumulative_s[index + 1]
            if target_s > end_s and index < len(self.polyline) - 2:
                continue
            span = max(end_s - start_s, 1e-12)
            t = (target_s - start_s) / span
            ax, ay = a
            bx, by = b
            heading = math.atan2(by - ay, bx - ax)
            cx = ax + (bx - ax) * t
            cy = ay + (by - ay) * t
            x = cx - math.sin(heading) * l
            y = cy + math.cos(heading) * l
            return LanePose(lane_id=self.lane_id, s=target_s, l=l, x=x, y=y, heading=heading)
        raise RuntimeError(f"LaneSegment {self.lane_id!r} sampling failed")


class LaneGraph:
    """Lightweight lane graph and Frenet projection helper built from Scenario map features."""

    def __init__(self, lanes: list[LaneSegment], lane_width: float = 3.5):
        self.lanes = [lane for lane in lanes if len(lane.polyline) >= 2 and lane.length > 0.0]
        self.lane_width = lane_width
        self._by_id = {lane.lane_id: lane for lane in self.lanes}

    @classmethod
    def from_scenario(cls, scenario: Scenario) -> "LaneGraph":
        lane_width = scenario.road.lane_width
        lanes: list[LaneSegment] = []
        center_features = [feature for feature in scenario.map_features if _is_centerline(feature.feature_type)]
        for feature in center_features:
            lanes.append(LaneSegment.create(f"center:{feature.feature_id}", feature.polyline, "map_centerline", [feature.feature_id]))

        if not lanes:
            lanes.extend(_derive_centerlines_from_boundaries(scenario, lane_width))

        lanes.extend(_reference_lines_from_boundaries(scenario))

        if not lanes:
            lanes.extend(_fallback_straight_lanes(scenario))

        return cls(lanes=lanes, lane_width=lane_width)

    @property
    def driving_lanes(self) -> list[LaneSegment]:
        lanes = [lane for lane in self.lanes if lane.source != "map_reference_line"]
        return lanes or self.lanes

    def project(
        self,
        x: float,
        y: float,
        max_distance: float | None = None,
        include_reference: bool = False,
    ) -> FrenetProjection | None:
        best: FrenetProjection | None = None
        lanes = self.lanes if include_reference else self.driving_lanes
        for lane in lanes:
            projection = lane.project(x, y)
            if projection is None:
                continue
            if max_distance is not None and projection.distance > max_distance:
                continue
            if best is None or projection.distance < best.distance:
                best = projection
        return best

    def project_state(
        self,
        state: VehicleState,
        max_distance: float | None = None,
        include_reference: bool = False,
    ) -> FrenetProjection | None:
        return self.project(state.x, state.y, max_distance=max_distance, include_reference=include_reference)

    def project_to_lane(self, lane_id: str, x: float, y: float) -> FrenetProjection | None:
        if lane_id not in self._by_id:
            raise KeyError(f"lane_id={lane_id!r} not found")
        return self._by_id[lane_id].project(x, y)

    def project_state_to_lane(self, lane_id: str, state: VehicleState) -> FrenetProjection | None:
        return self.project_to_lane(lane_id, state.x, state.y)

    def lane_length(self, lane_id: str) -> float:
        if lane_id not in self._by_id:
            raise KeyError(f"lane_id={lane_id!r} not found")
        return self._by_id[lane_id].length

    def sample_pose(self, lane_id: str, s: float, l: float = 0.0) -> LanePose:
        if lane_id not in self._by_id:
            raise KeyError(f"lane_id={lane_id!r} not found")
        return self._by_id[lane_id].sample_pose(s, l)

    def heading_mismatch(self, state: VehicleState, projection: FrenetProjection | None = None) -> float | None:
        projection = projection or self.project_state(state)
        if projection is None:
            return None
        # Map polylines converted from datasets may not encode driving direction, so diagnostics use
        # directionless tangent alignment. A vehicle heading 180 degrees from the stored polyline order
        # is still geometrically aligned with that lane/reference line.
        return _parallel_angle_diff(state.yaw, projection.heading)

    def lane_deviation(self, state: VehicleState, projection: FrenetProjection | None = None) -> float | None:
        projection = projection or self.project_state(state)
        if projection is None:
            return None
        return abs(projection.l)

    def find_front_back_actors(
        self,
        ego: VehicleState,
        actors: Iterable[VehicleState],
        max_lateral: float | None = None,
    ) -> tuple[tuple[VehicleState, FrenetProjection, float] | None, tuple[VehicleState, FrenetProjection, float] | None]:
        ego_projection = self.project_state(ego)
        if ego_projection is None:
            return None, None
        lateral_limit = max_lateral if max_lateral is not None else self.lane_width * 0.6
        front: tuple[VehicleState, FrenetProjection, float] | None = None
        back: tuple[VehicleState, FrenetProjection, float] | None = None
        for actor in actors:
            projection = self.project_state(actor)
            if projection is None or projection.lane_id != ego_projection.lane_id:
                continue
            if abs(projection.l - ego_projection.l) > lateral_limit:
                continue
            ds = projection.s - ego_projection.s
            if ds >= 0.0:
                if front is None or ds < front[2]:
                    front = (actor, projection, ds)
            else:
                back_dist = abs(ds)
                if back is None or back_dist < back[2]:
                    back = (actor, projection, back_dist)
        return front, back

    def to_summary(self) -> dict:
        by_source: dict[str, int] = {}
        for lane in self.lanes:
            by_source[lane.source] = by_source.get(lane.source, 0) + 1
        return {"lane_count": len(self.lanes), "driving_lane_count": len(self.driving_lanes), "lane_width": self.lane_width, "sources": dict(sorted(by_source.items()))}



def _reference_lines_from_boundaries(scenario: Scenario) -> list[LaneSegment]:
    lanes = []
    for feature in scenario.map_features:
        if len(feature.polyline) < 2 or not _is_boundary(feature.feature_type):
            continue
        lanes.append(
            LaneSegment.create(
                f"reference:{feature.feature_id}",
                feature.polyline,
                "map_reference_line",
                [feature.feature_id],
            )
        )
    return lanes

def _derive_centerlines_from_boundaries(scenario: Scenario, lane_width: float) -> list[LaneSegment]:
    features = [feature for feature in scenario.map_features if len(feature.polyline) >= 2 and _is_boundary(feature.feature_type)]
    lanes: list[LaneSegment] = []
    used_pairs: set[tuple[str, str]] = set()
    max_sep = max(lane_width * 3.0, 8.5)
    min_sep = max(1.2, lane_width * 0.35)
    for left_index, left in enumerate(features):
        for right in features[left_index + 1 :]:
            pair_key = tuple(sorted((left.feature_id, right.feature_id)))
            if pair_key in used_pairs:
                continue
            centerline = _centerline_between(left.polyline, right.polyline, min_sep=min_sep, max_sep=max_sep)
            if centerline is None:
                continue
            lane = LaneSegment.create(
                f"derived:{left.feature_id}:{right.feature_id}",
                centerline,
                "derived_from_boundaries",
                [left.feature_id, right.feature_id],
            )
            if lane.length >= 5.0:
                lanes.append(lane)
                used_pairs.add(pair_key)
    if lanes:
        return _dedupe_lanes(lanes)
    return []


def _centerline_between(
    first: list[tuple[float, float]],
    second: list[tuple[float, float]],
    min_sep: float,
    max_sep: float,
    samples: int = 16,
) -> list[tuple[float, float]] | None:
    if len(first) < 2 or len(second) < 2:
        return None
    heading_a = _overall_heading(first)
    heading_b = _overall_heading(second)
    if _parallel_angle_diff(heading_a, heading_b) > math.radians(25.0):
        return None
    a_points = [_sample_polyline(first, i / max(samples - 1, 1)) for i in range(samples)]
    b_forward = [_sample_polyline(second, i / max(samples - 1, 1)) for i in range(samples)]
    b_reverse = [_sample_polyline(list(reversed(second)), i / max(samples - 1, 1)) for i in range(samples)]
    forward_dist = _mean_distance(a_points, b_forward)
    reverse_dist = _mean_distance(a_points, b_reverse)
    b_points = b_reverse if reverse_dist < forward_dist else b_forward
    separation = _mean_distance(a_points, b_points)
    if separation < min_sep or separation > max_sep:
        return None
    if max(_distance(a, b) for a, b in zip(a_points, b_points)) > max_sep * 1.6:
        return None
    return [((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0) for a, b in zip(a_points, b_points)]


def _fallback_straight_lanes(scenario: Scenario) -> list[LaneSegment]:
    lanes = []
    half_width = scenario.road.half_width
    for index in range(scenario.road.lane_count):
        y = -half_width + scenario.road.lane_width * (index + 0.5)
        lanes.append(
            LaneSegment.create(
                lane_id=f"fallback:{index}",
                polyline=[(-20.0, y), (scenario.road.length + 20.0, y)],
                source="fallback_road",
            )
        )
    return lanes


def _dedupe_lanes(lanes: list[LaneSegment], key_precision: float = 1.0) -> list[LaneSegment]:
    seen: set[tuple[int, int, int, int]] = set()
    result: list[LaneSegment] = []
    for lane in sorted(lanes, key=lambda item: (-item.length, item.lane_id)):
        start = lane.polyline[0]
        end = lane.polyline[-1]
        key = (
            round(start[0] / key_precision),
            round(start[1] / key_precision),
            round(end[0] / key_precision),
            round(end[1] / key_precision),
        )
        reverse_key = (key[2], key[3], key[0], key[1])
        if key in seen or reverse_key in seen:
            continue
        seen.add(key)
        result.append(lane)
    return sorted(result, key=lambda item: item.lane_id)


def _is_centerline(feature_type: str) -> bool:
    upper = feature_type.upper()
    return "LANE_CENTER" in upper or "CENTERLINE" in upper or upper.endswith("CENTER")


def _is_boundary(feature_type: str) -> bool:
    upper = feature_type.upper()
    return "ROAD_LINE" in upper or "ROAD_EDGE" in upper or "LANE_MARK" in upper or "BOUNDARY" in upper


def _clean_polyline(points: list[tuple[float, float]], min_dist: float = 1e-6) -> list[tuple[float, float]]:
    cleaned: list[tuple[float, float]] = []
    for x, y in points:
        point = (float(x), float(y))
        if not cleaned or _distance(cleaned[-1], point) > min_dist:
            cleaned.append(point)
    return cleaned


def _cumulative_s(points: list[tuple[float, float]]) -> list[float]:
    if not points:
        return []
    cumulative = [0.0]
    for a, b in zip(points, points[1:]):
        cumulative.append(cumulative[-1] + _distance(a, b))
    return cumulative


def _sample_polyline(points: list[tuple[float, float]], ratio: float) -> tuple[float, float]:
    cumulative = _cumulative_s(points)
    if not cumulative or cumulative[-1] <= 0.0:
        return points[0]
    target = max(0.0, min(1.0, ratio)) * cumulative[-1]
    for index, (a, b) in enumerate(zip(points, points[1:])):
        if target > cumulative[index + 1] and index < len(points) - 2:
            continue
        span = max(cumulative[index + 1] - cumulative[index], 1e-12)
        t = (target - cumulative[index]) / span
        return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
    return points[-1]


def _overall_heading(points: list[tuple[float, float]]) -> float:
    start = points[0]
    end = points[-1]
    return math.atan2(end[1] - start[1], end[0] - start[0])


def _parallel_angle_diff(a: float, b: float) -> float:
    diff = abs(_wrap_to_pi(a - b))
    return min(diff, abs(math.pi - diff))


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _mean_distance(first: list[tuple[float, float]], second: list[tuple[float, float]]) -> float:
    return sum(_distance(a, b) for a, b in zip(first, second)) / max(1, min(len(first), len(second)))


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
