"""Deterministic 2.5D A* route option for planner comparisons."""

from dataclasses import dataclass
import heapq
import math
import time

import numpy as np

from greenhouse_inspection.free_space import segment_blocked
from greenhouse_inspection.viewpoint_route import route_length

from .static_greenhouse import HOUSE_X_BOUNDS, HOUSE_Y_BOUNDS


@dataclass(frozen=True)
class AStarRouteResult:
    route: np.ndarray
    planning_time_ms: float
    expanded_nodes: int
    raw_grid_nodes: int
    simplified_grid_nodes: int
    grid_resolution_m: float


def _inside_box(point, box):
    x_value, y_value, z_value = point
    return bool(
        box[0] <= x_value <= box[1]
        and box[2] <= y_value <= box[3]
        and box[4] <= z_value <= box[5]
    )


def _grid_index(value, lower, resolution):
    return int(round((float(value) - float(lower)) / float(resolution)))


def _grid_point(index, lower_x, lower_y, resolution, altitude):
    return np.asarray((
        lower_x + index[0] * resolution,
        lower_y + index[1] * resolution,
        altitude,
    ), dtype=float)


def _nearest_free(seed, is_free, maximum_radius=12):
    if is_free(seed):
        return seed
    for radius in range(1, maximum_radius + 1):
        candidates = []
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                candidates.append((seed[0] + dx, seed[1] + dy))
        for candidate in sorted(
                candidates,
                key=lambda item: (
                    (item[0] - seed[0]) ** 2 + (item[1] - seed[1]) ** 2,
                    item,
                )):
            if is_free(candidate):
                return candidate
    raise ValueError("no free A* cell exists near route endpoint")


def _reconstruct(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _simplify(points, boxes):
    if len(points) <= 2:
        return list(points)
    simplified = [points[0]]
    anchor = 0
    while anchor < len(points) - 1:
        furthest = anchor + 1
        for candidate in range(anchor + 2, len(points)):
            if segment_blocked(points[anchor], points[candidate], boxes):
                break
            furthest = candidate
        simplified.append(points[furthest])
        anchor = furthest
    return simplified


def plan_astar_route(
        start_pose, goal_pose, boxes, resolution=0.4,
        x_bounds=HOUSE_X_BOUNDS, y_bounds=HOUSE_Y_BOUNDS):
    """Plan and exactly revalidate a 2.5D route between two 4D poses."""
    started = time.perf_counter()
    start = np.asarray(start_pose, dtype=float)
    goal = np.asarray(goal_pose, dtype=float)
    if start.shape != (4,) or goal.shape != (4,):
        raise ValueError("start and goal must be x/y/z/yaw poses")
    if not np.isfinite(start).all() or not np.isfinite(goal).all():
        raise ValueError("start and goal poses must be finite")
    resolution = float(resolution)
    if resolution <= 0.0:
        raise ValueError("A* resolution must be positive")
    boxes = tuple(tuple(float(value) for value in box) for box in boxes)
    altitude = float(goal[2])
    lower_x, upper_x = map(float, x_bounds)
    lower_y, upper_y = map(float, y_bounds)
    max_x = int(math.floor((upper_x - lower_x) / resolution))
    max_y = int(math.floor((upper_y - lower_y) / resolution))

    def in_grid(index):
        return 0 <= index[0] <= max_x and 0 <= index[1] <= max_y

    def is_free(index):
        if not in_grid(index):
            return False
        point = _grid_point(
            index, lower_x, lower_y, resolution, altitude)
        return not any(_inside_box(point, box) for box in boxes)

    start_index = _nearest_free((
        _grid_index(start[0], lower_x, resolution),
        _grid_index(start[1], lower_y, resolution),
    ), is_free)
    goal_index = _nearest_free((
        _grid_index(goal[0], lower_x, resolution),
        _grid_index(goal[1], lower_y, resolution),
    ), is_free)
    neighbours = (
        (-1, -1, math.sqrt(2.0)), (-1, 0, 1.0),
        (-1, 1, math.sqrt(2.0)), (0, -1, 1.0),
        (0, 1, 1.0), (1, -1, math.sqrt(2.0)),
        (1, 0, 1.0), (1, 1, math.sqrt(2.0)),
    )

    def heuristic(index):
        return math.hypot(
            goal_index[0] - index[0], goal_index[1] - index[1])

    queue = [(heuristic(start_index), 0.0, start_index)]
    came_from = {}
    cost = {start_index: 0.0}
    expanded = 0
    found = None
    while queue:
        _, current_cost, current = heapq.heappop(queue)
        if current_cost > cost.get(current, math.inf) + 1e-12:
            continue
        expanded += 1
        if current == goal_index:
            found = current
            break
        current_point = _grid_point(
            current, lower_x, lower_y, resolution, altitude)
        for dx, dy, step_cost in neighbours:
            neighbour = (current[0] + dx, current[1] + dy)
            if not is_free(neighbour):
                continue
            neighbour_point = _grid_point(
                neighbour, lower_x, lower_y, resolution, altitude)
            if segment_blocked(current_point, neighbour_point, boxes):
                continue
            tentative = current_cost + step_cost
            if tentative + 1e-12 >= cost.get(neighbour, math.inf):
                continue
            came_from[neighbour] = current
            cost[neighbour] = tentative
            heapq.heappush(queue, (
                tentative + heuristic(neighbour), tentative, neighbour))
    if found is None:
        raise ValueError("A* found no collision-free route")

    grid_indices = _reconstruct(came_from, found)
    grid_points = [
        _grid_point(index, lower_x, lower_y, resolution, altitude)
        for index in grid_indices
    ]
    # Preserve exact endpoint x/y values when their vertical segments are safe.
    exact_start_air = np.asarray((start[0], start[1], altitude), dtype=float)
    exact_goal = goal[:3].copy()
    if segment_blocked(start[:3], exact_start_air, boxes):
        raise ValueError("A* launch vertical segment intersects an obstacle")
    if segment_blocked(grid_points[0], exact_start_air, boxes):
        raise ValueError("A* cannot connect its grid to the exact start")
    if segment_blocked(grid_points[-1], exact_goal, boxes):
        raise ValueError("A* cannot connect its grid to the exact goal")
    points = [exact_start_air] + grid_points + [exact_goal]
    points = _simplify(points, boxes)
    route_points = [start[:3].copy()]
    if np.linalg.norm(exact_start_air - start[:3]) > 1e-9:
        route_points.append(exact_start_air)
    route_points.extend(points[1:])
    route_points = [route_points[0]] + [
        point for index, point in enumerate(route_points[1:], start=1)
        if np.linalg.norm(point - route_points[index - 1]) > 1e-9
    ]
    route = np.empty((len(route_points), 4), dtype=float)
    for index, point in enumerate(route_points):
        route[index, :3] = point
        if index == len(route_points) - 1:
            route[index, 3] = goal[3]
        else:
            delta = route_points[index + 1] - point
            route[index, 3] = math.atan2(delta[1], delta[0])
    for start_row, end_row in zip(route, route[1:]):
        if segment_blocked(start_row[:3], end_row[:3], boxes):
            raise RuntimeError("simplified A* route failed exact validation")
    if not math.isfinite(route_length(route)):
        raise RuntimeError("A* produced a non-finite route")
    return AStarRouteResult(
        route=route,
        planning_time_ms=1000.0 * (time.perf_counter() - started),
        expanded_nodes=expanded,
        raw_grid_nodes=len(grid_points),
        simplified_grid_nodes=len(points),
        grid_resolution_m=resolution,
    )
