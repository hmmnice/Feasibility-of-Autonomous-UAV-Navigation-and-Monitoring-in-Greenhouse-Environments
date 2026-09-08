"""Transparent selection of one already-valid camera viewpoint."""

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from .viewpoint_geometry import CandidateViewpoint, Point3
from .viewpoint_validation import Box, ValidationResult


class NoValidViewpointError(ValueError):
    """Raised when Phase 3 supplied no candidate eligible for selection."""


class NoAcceptableViewpointError(ValueError):
    """Raised when valid poses exist but none meets the request tolerances."""


@dataclass(frozen=True)
class SelectionConfig:
    """Simulation tolerances; these are not claimed biological requirements."""

    maximum_direction_error: float = math.radians(30.0)
    maximum_distance_error: float = 0.5

    def __post_init__(self):
        values = (self.maximum_direction_error, self.maximum_distance_error)
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("selection tolerances must be finite and >= 0")


@dataclass(frozen=True)
class PerspectiveRequest:
    """Requested target-to-camera direction and camera distance."""

    azimuth: float
    elevation: float
    distance: float

    def __post_init__(self):
        if not all(math.isfinite(value) for value in (
                self.azimuth, self.elevation, self.distance)):
            raise ValueError("requested perspective must be finite")
        if not -math.pi / 2.0 <= self.elevation <= math.pi / 2.0:
            raise ValueError("requested elevation must be between -90 and +90 deg")
        if self.distance <= 0.0:
            raise ValueError("requested distance must be positive")

    @property
    def direction(self) -> Point3:
        cos_elevation = math.cos(self.elevation)
        return (
            cos_elevation * math.cos(self.azimuth),
            cos_elevation * math.sin(self.azimuth),
            math.sin(self.elevation),
        )


@dataclass(frozen=True)
class CandidateScore:
    """Explainable selection metrics for one Phase 3-valid candidate."""

    validation: ValidationResult
    angular_error: float
    distance_error: float
    obstacle_clearance: float

    @property
    def candidate(self) -> CandidateViewpoint:
        return self.validation.candidate

    @property
    def vehicle_position(self) -> Point3:
        return self.validation.vehicle_position

    @property
    def sort_key(self) -> tuple[float, float, float, int]:
        # Equivalent spherical samples can differ by ~1e-8 rad after inverse trig.
        return (
            round(self.angular_error, 6),
            round(self.distance_error, 6),
            -round(self.obstacle_clearance, 6),
            self.candidate.index,
        )


@dataclass(frozen=True)
class SelectionResult:
    """The chosen viewpoint plus the complete ordered valid set."""

    request: PerspectiveRequest
    config: SelectionConfig
    selected: CandidateScore
    ranked: tuple[CandidateScore, ...]


def _point(values: Sequence[float]) -> Point3:
    if len(values) != 3:
        raise ValueError("point must contain x, y and z")
    point = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in point):
        raise ValueError("point must contain finite values")
    return point


def point_box_clearance(point: Sequence[float], box: Sequence[float]) -> float:
    """Euclidean distance from a point to a closed axis-aligned box."""
    point = _point(point)
    if len(box) != 6:
        raise ValueError("box must contain six limits")
    xlo, xhi, ylo, yhi, zlo, zhi = (float(value) for value in box)
    if (xlo > xhi or ylo > yhi or zlo > zhi
            or not all(math.isfinite(value) for value in (
                xlo, xhi, ylo, yhi, zlo, zhi))):
        raise ValueError("box limits must be finite and ordered")
    distances = (
        max(xlo - point[0], 0.0, point[0] - xhi),
        max(ylo - point[1], 0.0, point[1] - yhi),
        max(zlo - point[2], 0.0, point[2] - zhi),
    )
    return math.sqrt(sum(value * value for value in distances))


def minimum_obstacle_clearance(
        point: Sequence[float], boxes: Iterable[Box]) -> float:
    """Minimum clearance to an inflated obstacle, or infinity if none exist."""
    clearances = [point_box_clearance(point, box) for box in boxes]
    return min(clearances, default=math.inf)


def candidate_view_direction(candidate: CandidateViewpoint) -> Point3:
    """Unit target-to-camera vector represented by a candidate."""
    delta = tuple(
        camera - target
        for camera, target in zip(candidate.camera_position, candidate.target))
    length = math.sqrt(sum(value * value for value in delta))
    if length <= 1e-9:
        raise ValueError("candidate camera and target must differ")
    return tuple(value / length for value in delta)


def direction_error(candidate: CandidateViewpoint, request: PerspectiveRequest) -> float:
    """Return the spherical viewing-direction error in radians."""
    dot = sum(
        actual * desired
        for actual, desired in zip(
            candidate_view_direction(candidate), request.direction))
    return math.acos(min(1.0, max(-1.0, dot)))


def score_valid_candidates(
        validations: Iterable[ValidationResult], request: PerspectiveRequest,
        collision_boxes: Iterable[Box]) -> list[CandidateScore]:
    """Score and order only candidates accepted by every Phase 3 check."""
    collision_boxes = tuple(collision_boxes)
    scored = []
    for validation in validations:
        if not validation.valid:
            continue
        candidate = validation.candidate
        scored.append(CandidateScore(
            validation=validation,
            angular_error=direction_error(candidate, request),
            distance_error=abs(candidate.distance - request.distance),
            obstacle_clearance=minimum_obstacle_clearance(
                validation.vehicle_position, collision_boxes),
        ))
    return sorted(scored, key=lambda score: score.sort_key)


def select_viewpoint(
        validations: Iterable[ValidationResult], request: PerspectiveRequest,
        collision_boxes: Iterable[Box],
        config: SelectionConfig = SelectionConfig()) -> SelectionResult:
    """Choose one valid viewpoint using the documented lexicographic rule."""
    ranked = tuple(score_valid_candidates(
        validations, request, collision_boxes))
    if not ranked:
        raise NoValidViewpointError(
            "no Phase 3-valid viewpoint satisfies the selection precondition")
    acceptable = [
        score for score in ranked
        if score.angular_error <= config.maximum_direction_error + 1e-9
        and score.distance_error <= config.maximum_distance_error + 1e-9]
    if not acceptable:
        raise NoAcceptableViewpointError(
            "valid viewpoints exist, but none meets the configured requested-"
            "perspective tolerances")
    return SelectionResult(
        request=request, config=config, selected=acceptable[0], ranked=ranked)
