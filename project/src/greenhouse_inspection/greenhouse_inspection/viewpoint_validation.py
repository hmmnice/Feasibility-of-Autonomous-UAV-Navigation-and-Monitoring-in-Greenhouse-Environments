"""Independent safety checks for candidate imaging viewpoints."""

from dataclasses import dataclass
from enum import Enum
import math
from statistics import median
from typing import Iterable, Optional, Sequence

from .free_space import DRONE_HALF, DRONE_HALF_Z, EAVE, TRACKING_ALLOWANCE
from .gimbal_geometry import JointLimits
from .viewpoint_geometry import (
    CameraConfig,
    CandidateViewpoint,
    Point3,
    vehicle_position_for_camera,
)


Box = tuple[float, float, float, float, float, float]


class RejectionReason(str, Enum):
    """Stable, presentation-friendly outcome for one candidate."""

    VALID = "VALID"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    COLLISION = "COLLISION"
    DISTANCE_INVALID = "DISTANCE_INVALID"
    GIMBAL_LIMIT = "GIMBAL_LIMIT"
    OCCLUDED = "OCCLUDED"


@dataclass(frozen=True)
class FlightBounds:
    """Inclusive map-frame envelope for the nominal camera/vehicle centre."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def __post_init__(self):
        values = (
            self.x_min, self.x_max, self.y_min,
            self.y_max, self.z_min, self.z_max)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("flight bounds must be finite")
        if (self.x_min > self.x_max or self.y_min > self.y_max
                or self.z_min > self.z_max):
            raise ValueError("flight-bound minimums must not exceed maximums")


@dataclass(frozen=True)
class ValidationConfig:
    """Configurable physical assumptions used by all candidate checks."""

    bounds: FlightBounds
    camera: CameraConfig = CameraConfig()
    gimbal_limits: JointLimits = JointLimits()
    target_terminal_depth: float = 0.25

    def __post_init__(self):
        if (not math.isfinite(self.target_terminal_depth)
                or self.target_terminal_depth < 0.0):
            raise ValueError("target_terminal_depth must be finite and >= 0")


@dataclass(frozen=True)
class ValidationResult:
    """All independent checks plus one deterministic headline outcome."""

    candidate: CandidateViewpoint
    inside_bounds: bool
    collision_free: bool
    distance_valid: bool
    gimbal_valid: bool
    target_visible: bool
    reason: RejectionReason
    vehicle_position: Point3
    collision_box_index: Optional[int] = None
    occluding_box_index: Optional[int] = None

    @property
    def valid(self) -> bool:
        return self.reason is RejectionReason.VALID


def _point(values: Sequence[float]) -> Point3:
    if len(values) != 3:
        raise ValueError("point must contain x, y and z")
    point = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in point):
        raise ValueError("point must contain finite values")
    return point


def _box(values: Sequence[float]) -> Box:
    if len(values) != 6:
        raise ValueError("box must contain six limits")
    box = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in box):
        raise ValueError("box must contain finite values")
    if box[0] > box[1] or box[2] > box[3] or box[4] > box[5]:
        raise ValueError("box minimums must not exceed maximums")
    return box


def point_in_bounds(point: Sequence[float], bounds: FlightBounds) -> bool:
    x, y, z = _point(point)
    return (
        bounds.x_min <= x <= bounds.x_max
        and bounds.y_min <= y <= bounds.y_max
        and bounds.z_min <= z <= bounds.z_max)


def point_in_box(point: Sequence[float], box: Sequence[float]) -> bool:
    x, y, z = _point(point)
    xlo, xhi, ylo, yhi, zlo, zhi = _box(box)
    return xlo <= x <= xhi and ylo <= y <= yhi and zlo <= z <= zhi


def collision_box_index(
        point: Sequence[float], boxes: Iterable[Sequence[float]]) -> Optional[int]:
    """Return the first inflated obstacle containing the candidate centre."""
    for index, box in enumerate(boxes):
        if point_in_box(point, box):
            return index
    return None


def segment_box_interval(
        start: Sequence[float], end: Sequence[float],
        box: Sequence[float]) -> Optional[tuple[float, float]]:
    """Return the closed segment's entry/exit parameters for an AABB."""
    start = _point(start)
    end = _point(end)
    box = _box(box)
    lower = (box[0], box[2], box[4])
    upper = (box[1], box[3], box[5])
    t_enter, t_exit = 0.0, 1.0
    for origin, finish, lo, hi in zip(start, end, lower, upper):
        delta = finish - origin
        if abs(delta) <= 1e-12:
            if origin < lo or origin > hi:
                return None
            continue
        first = (lo - origin) / delta
        second = (hi - origin) / delta
        axis_enter, axis_exit = sorted((first, second))
        t_enter = max(t_enter, axis_enter)
        t_exit = min(t_exit, axis_exit)
        if t_enter > t_exit:
            return None
    return t_enter, t_exit


def visibility_blocker(
        camera_position: Sequence[float], target: Sequence[float],
        optical_boxes: Iterable[Sequence[float]],
        target_box_index: Optional[int] = None,
        target_terminal_depth: float = 0.25) -> Optional[int]:
    """Return the first box that blocks the camera-to-target ray."""
    camera_position = _point(camera_position)
    target = _point(target)
    intersections = []
    for index, box in enumerate(optical_boxes):
        interval = segment_box_interval(camera_position, target, box)
        if interval is not None:
            intersections.append((interval[0], index))
    for t_enter, index in sorted(intersections):
        if index != target_box_index:
            return index
        entry = tuple(
            start + t_enter * (finish - start)
            for start, finish in zip(camera_position, target))
        terminal_distance = math.dist(entry, target)
        if terminal_distance > target_terminal_depth + 1e-9:
            return index
    return None


def deflate_box(
        box: Sequence[float], xy: float, top: float = 0.0) -> Box:
    """Remove known map-planner inflation to form an optical obstacle."""
    xlo, xhi, ylo, yhi, zlo, zhi = _box(box)
    result = (xlo + xy, xhi - xy, ylo + xy, yhi - xy, zlo, zhi - top)
    return _box(result)


def optical_boxes_from_geometry(
        map_geometry, ceiling: float = EAVE) -> tuple[Box, ...]:
    """Convert planner collision boxes to less-inflated LOS geometry."""
    crop_count = len(map_geometry.rows)
    result = []
    for index, box in enumerate(map_geometry.boxes):
        if index < crop_count:
            result.append(deflate_box(box, DRONE_HALF, DRONE_HALF_Z))
        else:
            optical = deflate_box(
                box, DRONE_HALF + TRACKING_ALLOWANCE)
            result.append((*optical[:5], max(optical[5], ceiling)))
    return tuple(result)


def collision_boxes_from_geometry(
        map_geometry, ceiling: float = EAVE) -> tuple[Box, ...]:
    """Return inflated vehicle boxes, extending tall structures to ceiling."""
    if not math.isfinite(ceiling):
        raise ValueError("ceiling must be finite")
    crop_count = len(map_geometry.rows)
    result = []
    for index, values in enumerate(map_geometry.boxes):
        box = _box(values)
        if index < crop_count:
            result.append(box)
        else:
            result.append((*box[:5], max(box[5], ceiling)))
    return tuple(result)


def associate_target_box(
        target: Sequence[float], crop_boxes: Iterable[Sequence[float]],
        tolerance: float = 0.05) -> Optional[int]:
    """Associate a selected surface point with exactly one crop-row box."""
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("target association tolerance must be finite and >= 0")
    target = _point(target)
    matches = []
    for index, values in enumerate(crop_boxes):
        box = _box(values)
        expanded = (
            box[0] - tolerance, box[1] + tolerance,
            box[2] - tolerance, box[3] + tolerance,
            box[4] - tolerance, box[5] + tolerance)
        if point_in_box(target, expanded):
            matches.append(index)
    if len(matches) > 1:
        raise ValueError("target is ambiguous between multiple crop boxes")
    return matches[0] if matches else None


def flight_bounds_from_geometry(
        map_geometry, ceiling: float = EAVE,
        floor_clearance: float = DRONE_HALF,
        ceiling_clearance: float = 0.5) -> FlightBounds:
    """Build a conservative map-derived planning envelope."""
    if not map_geometry.rows or not map_geometry.extents or not map_geometry.boxes:
        raise ValueError("map geometry is empty")
    x_min = min(float(extent[0]) for extent in map_geometry.extents)
    x_max = max(float(extent[1]) for extent in map_geometry.extents)
    rows = sorted(float(value) for value in map_geometry.rows)
    spacings = [right - left for left, right in zip(rows, rows[1:])]
    row_padding = 0.5 * median(spacings) if spacings else DRONE_HALF
    crop_boxes = map_geometry.boxes[:len(rows)]
    clearances = (floor_clearance, ceiling_clearance)
    if (not math.isfinite(ceiling)
            or not all(math.isfinite(value) and value >= 0.0
                       for value in clearances)):
        raise ValueError("ceiling/vertical clearances must be finite and >= 0")
    z_min = min(float(box[4]) for box in crop_boxes) + floor_clearance
    z_max = ceiling - ceiling_clearance
    return FlightBounds(
        x_min, x_max, rows[0] - row_padding, rows[-1] + row_padding,
        z_min, z_max)


def validate_candidate(
        candidate: CandidateViewpoint, config: ValidationConfig,
        collision_boxes: Iterable[Sequence[float]],
        optical_boxes: Iterable[Sequence[float]],
        target_box_index: Optional[int] = None) -> ValidationResult:
    """Evaluate every independent check and assign one stable reason."""
    collision_boxes = tuple(collision_boxes)
    optical_boxes = tuple(optical_boxes)
    camera_position = candidate.camera_position
    vehicle_position = vehicle_position_for_camera(
        camera_position, candidate.vehicle_yaw_map)
    inside = point_in_bounds(vehicle_position, config.bounds)
    collision_index = collision_box_index(vehicle_position, collision_boxes)
    actual_distance = math.dist(camera_position, candidate.target)
    distance_valid = (
        config.camera.minimum_standoff <= actual_distance
        <= config.camera.maximum_standoff)
    gimbal_valid = (
        config.gimbal_limits.yaw_min <= candidate.gimbal_yaw
        <= config.gimbal_limits.yaw_max
        and config.gimbal_limits.pitch_min <= candidate.gimbal_pitch
        <= config.gimbal_limits.pitch_max)
    blocker = visibility_blocker(
        camera_position, candidate.target, optical_boxes, target_box_index,
        config.target_terminal_depth)

    checks = (
        (not inside, RejectionReason.OUT_OF_BOUNDS),
        (collision_index is not None, RejectionReason.COLLISION),
        (not distance_valid, RejectionReason.DISTANCE_INVALID),
        (not gimbal_valid, RejectionReason.GIMBAL_LIMIT),
        (blocker is not None, RejectionReason.OCCLUDED),
    )
    reason = next(
        (candidate_reason for failed, candidate_reason in checks if failed),
        RejectionReason.VALID)
    return ValidationResult(
        candidate=candidate,
        inside_bounds=inside,
        collision_free=collision_index is None,
        distance_valid=distance_valid,
        gimbal_valid=gimbal_valid,
        target_visible=blocker is None,
        reason=reason,
        vehicle_position=vehicle_position,
        collision_box_index=collision_index,
        occluding_box_index=blocker,
    )


def validate_candidates(
        candidates: Iterable[CandidateViewpoint], config: ValidationConfig,
        collision_boxes: Iterable[Sequence[float]],
        optical_boxes: Iterable[Sequence[float]],
        target_box_index: Optional[int] = None) -> list[ValidationResult]:
    """Validate a deterministic candidate sequence without ranking it."""
    collision_boxes = tuple(collision_boxes)
    optical_boxes = tuple(optical_boxes)
    return [
        validate_candidate(
            candidate, config, collision_boxes, optical_boxes,
            target_box_index)
        for candidate in candidates
    ]
