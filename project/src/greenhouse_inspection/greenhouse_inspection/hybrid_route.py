"""Route data, editing support, and static safety planning for hybrid flight."""

import math
import os
from collections import namedtuple

import numpy as np

from greenhouse_inspection.free_space import (
    TRANSIT_Z,
    _safe_leg_subdivided,
    segment_blocked,
)
from greenhouse_inspection.map_geometry import geometry
from greenhouse_inspection.path_follower import (
    MAX_STEP,
    simplify_path,
    taught_to_px4,
)
from greenhouse_inspection.relocalize import traj_path


ROUTE_COLUMNS = 5
MAX_TEACH_GAP = 3.0
CompiledRoute = namedtuple("CompiledRoute", "route geometry source")


def default_route_path(map_path):
    """Editable route paired with a saved map, without touching teach data."""
    base, _ = os.path.splitext(os.fspath(map_path))
    return base + "_hybrid_route.npy"


def _as_route(values, source):
    """Normalise route arrays to ``x, y, z, yaw, capture`` rows."""
    route = np.asarray(values, dtype=float)
    if route.ndim != 2 or route.shape[1] not in (3, 4, ROUTE_COLUMNS):
        raise ValueError(
            f"{source} must be Nx3, Nx4, or Nx{ROUTE_COLUMNS}; "
            f"got {route.shape}")
    if len(route) < 2:
        raise ValueError(f"{source} needs at least two waypoints")
    if not np.isfinite(route[:, :3]).all():
        raise ValueError(f"{source} has non-finite xyz coordinates")

    if route.shape[1] == 3:
        route = np.column_stack([route, np.full(len(route), np.nan),
                                 np.ones(len(route))])
    elif route.shape[1] == 4:
        route = np.column_stack([route, np.ones(len(route))])
    else:
        route = route.copy()

    if not np.isfinite(route[:, 3][~np.isnan(route[:, 3])]).all():
        raise ValueError(f"{source} has invalid yaw values")
    if not np.isfinite(route[:, 4]).all():
        raise ValueError(f"{source} has invalid capture values")
    route[:, 4] = (route[:, 4] >= 0.5).astype(float)
    return route


def load_route(path):
    """Load an operator-edited route without modifying it."""
    return _as_route(np.load(path), os.fspath(path))


def _longest_continuous_span(route, max_gap=MAX_TEACH_GAP):
    """Largest continuous portion of a taught trace for editor seeding."""
    gaps = np.linalg.norm(np.diff(route[:, :3], axis=0), axis=1)
    starts = np.r_[0, np.flatnonzero(gaps > max_gap) + 1]
    ends = np.r_[starts[1:], len(route)]
    start, end = max(zip(starts, ends), key=lambda span: span[1] - span[0])
    return route[start:end]


def load_taught_route(map_path, strict=True, edit_step=MAX_STEP):
    """Load a teach trajectory and thin it to safe-sized editable waypoints."""
    path = traj_path(os.fspath(map_path))
    route = _as_route(np.load(path), path)
    gaps = np.linalg.norm(np.diff(route[:, :3], axis=0), axis=1)
    if strict and len(gaps) and gaps.max() > MAX_TEACH_GAP:
        raise ValueError(
            f"Taught trajectory has a {gaps.max():.1f}m SLAM jump. It is not "
            "safe to replay; create and save an edited hybrid route instead.")
    if not strict:
        route = _longest_continuous_span(route)

    thinned = simplify_path(route[:, :4], max_step=edit_step)
    return np.column_stack([thinned, np.ones(len(thinned))])


def fill_yaws(route):
    """Use travel direction where a manual point deliberately has no yaw."""
    out = _as_route(route, "route")
    for i in range(len(out)):
        if not math.isnan(out[i, 3]):
            continue
        direction = None
        for j in range(i + 1, len(out)):
            delta = out[j, :2] - out[i, :2]
            if np.linalg.norm(delta) > 1e-6:
                direction = delta
                break
        if direction is None:
            for j in range(i - 1, -1, -1):
                delta = out[i, :2] - out[j, :2]
                if np.linalg.norm(delta) > 1e-6:
                    direction = delta
                    break
        out[i, 3] = (math.atan2(direction[1], direction[0])
                     if direction is not None else 0.0)
    return out


def _interpolate_yaw(start, end, fraction):
    """Interpolate across the shortest angular arc."""
    delta = (end - start + math.pi) % (2 * math.pi) - math.pi
    return start + fraction * delta


def densify_route(route, max_step=MAX_STEP):
    """Ensure every executable leg stays within the established step bound."""
    route = fill_yaws(route)
    out = [route[0].copy()]
    for end in route[1:]:
        start = out[-1]
        distance = float(np.linalg.norm(end[:3] - start[:3]))
        pieces = max(1, int(math.ceil(distance / max_step)))
        for i in range(1, pieces + 1):
            fraction = i / pieces
            point = start * (1.0 - fraction) + end * fraction
            point[:3] = start[:3] * (1.0 - fraction) + end[:3] * fraction
            point[3] = _interpolate_yaw(start[3], end[3], fraction)
            # A safety-inserted waypoint carries capture=0.
            point[4] = end[4] if i == pieces else 0.0
            out.append(point)
    return np.asarray(out, dtype=float)


def plan_safe_route(route, boxes, transit_z=TRANSIT_Z):
    """Insert obstacle-free detours while retaining the operator's anchors."""
    route = fill_yaws(route)
    for point in route:
        if segment_blocked(point[:3], point[:3], boxes) is not None:
            raise ValueError(
                "A route waypoint lies inside a mapped obstacle; move it "
                "in the editor rather than asking the controller to "
                "escape it.")

    planned = [route[0].copy()]
    here = route[0]
    for target in route[1:]:
        mids = _safe_leg_subdivided(here[:3], target[:3], boxes, transit_z)
        if mids is None:
            raise ValueError(
                f"No map-safe path from {tuple(here[:3])} "
                f"to {tuple(target[:3])}.")
        for mid in mids:
            planned.append(np.array([mid[0], mid[1], mid[2], target[3], 0.0]))
        planned.append(target.copy())
        here = target
    return densify_route(np.asarray(planned, dtype=float))


def compile_hybrid_route(map_path, route_path="", transit_z=TRANSIT_Z):
    """Make an obstacle-checked mission route in the saved-map ENU frame."""
    map_path = os.fspath(map_path)
    if route_path:
        raw = load_route(route_path)
        source = os.fspath(route_path)
    else:
        raw = load_taught_route(map_path, strict=True)
        source = traj_path(map_path)

    geo = geometry(np.load(map_path))
    route = plan_safe_route(raw, geo.boxes, transit_z=transit_z)
    return CompiledRoute(route, geo, source)


def route_to_px4(route, saved_from_live):
    """Saved-map ENU hybrid route -> PX4 NED, preserving capture flags."""
    route = fill_yaws(route)
    px4 = taught_to_px4(route[:, :4], saved_from_live)
    return np.column_stack([px4, route[:, 4]])
