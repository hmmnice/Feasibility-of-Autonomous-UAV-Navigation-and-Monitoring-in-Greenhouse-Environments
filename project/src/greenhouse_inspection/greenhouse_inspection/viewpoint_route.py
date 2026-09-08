"""Offline routing from a start pose to a selected viewpoint."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, Sequence

from .free_space import TRANSIT_Z, safe_route, segment_blocked
from .viewpoint_selection import SelectionResult
from .viewpoint_validation import Box, collision_box_index


Waypoint = tuple[float, float, float, float]


class RouteStatus(str, Enum):
    """Phase 5 outcome, separate from Phase 3 candidate rejection."""

    ROUTE_VALID = "ROUTE_VALID"
    NO_SAFE_ROUTE = "NO_SAFE_ROUTE"


@dataclass(frozen=True)
class RoutePlan:
    """Verified offline route result for one selected inspection pose."""

    status: RouteStatus
    start: Waypoint
    goal: Waypoint
    waypoints: tuple[Waypoint, ...]
    length: float
    inserted_waypoints: int
    direct: bool
    detail: str = ""

    @property
    def valid(self) -> bool:
        return self.status is RouteStatus.ROUTE_VALID


def _start_pose(position: Sequence[float], yaw: float) -> Waypoint:
    if len(position) != 3:
        raise ValueError("start position must contain x, y and z")
    values = tuple(float(value) for value in position) + (float(yaw),)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("start pose must contain finite values")
    return values


def route_length(waypoints: Iterable[Sequence[float]]) -> float:
    """Return total 3D polyline length in metres."""
    points = [tuple(float(value) for value in point[:3])
              for point in waypoints]
    return sum(math.dist(start, end)
               for start, end in zip(points, points[1:]))


def blocked_route_segments(
        waypoints: Iterable[Sequence[float]],
        boxes: Iterable[Box]) -> list[tuple[int, Box]]:
    """Return every route leg that intersects an inflated obstacle box."""
    points = tuple(waypoints)
    boxes = tuple(boxes)
    blocked = []
    for index, (start, end) in enumerate(zip(points, points[1:])):
        box = segment_blocked(start[:3], end[:3], boxes)
        if box is not None:
            blocked.append((index, box))
    return blocked


def plan_route_to_viewpoint(
        selection: SelectionResult, start_position: Sequence[float],
        collision_boxes: Iterable[Box], rows: Iterable[float],
        start_yaw: float = 0.0, transit_z: float = TRANSIT_Z) -> RoutePlan:
    """Plan and independently verify a route to Phase 4's vehicle pose."""
    start = _start_pose(start_position, start_yaw)
    candidate = selection.selected.candidate
    goal = tuple(float(value) for value in selection.selected.vehicle_position)
    goal = goal + (float(candidate.vehicle_yaw_map),)
    boxes = tuple(collision_boxes)
    rows = tuple(float(value) for value in rows)

    if collision_box_index(start[:3], boxes) is not None:
        return RoutePlan(
            RouteStatus.NO_SAFE_ROUTE, start, goal, (), 0.0, 0, False,
            "start pose lies inside an inflated mapped obstacle")
    if collision_box_index(goal[:3], boxes) is not None:
        return RoutePlan(
            RouteStatus.NO_SAFE_ROUTE, start, goal, (), 0.0, 0, False,
            "selected vehicle pose lies inside an inflated mapped obstacle")

    try:
        planned = safe_route(
            [goal], rows, start=start[:3], transit_z=float(transit_z),
            boxes=boxes)
    except ValueError as error:
        return RoutePlan(
            RouteStatus.NO_SAFE_ROUTE, start, goal, (), 0.0, 0, False,
            str(error))

    waypoints = (start,) + tuple(
        tuple(float(value) for value in waypoint) for waypoint in planned)
    blocked = blocked_route_segments(waypoints, boxes)
    if blocked:
        raise RuntimeError(
            "safe_route returned an obstacle-intersecting leg; refusing the "
            "unverified route")
    inserted = max(0, len(planned) - 1)
    return RoutePlan(
        RouteStatus.ROUTE_VALID,
        start,
        goal,
        waypoints,
        route_length(waypoints),
        inserted,
        inserted == 0,
    )
