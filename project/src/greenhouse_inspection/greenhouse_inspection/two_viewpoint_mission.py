"""Execute reviewed viewpoints for one or more crops in one PX4 flight."""

from copy import deepcopy
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from greenhouse_inspection.free_space import safe_route, segment_blocked
from greenhouse_inspection.gimbal_geometry import joint_pitch_to_camera
from greenhouse_inspection.hybrid_route import densify_route, route_to_px4
from greenhouse_inspection.viewpoint_execution import (
    actual_view_metrics,
    angle_error,
    insert_ground_launch_waypoint,
    pose_change,
    preflight_age_seconds,
    px4_pose_in_saved_map,
    saved_target_in_body,
    transform_change,
    validate_preflight_record,
)
from greenhouse_inspection.viewpoint_inspection import transform_from_tf
from greenhouse_inspection.viewpoint_route import route_length
from greenhouse_inspection.viewpoint_selection import (
    PerspectiveRequest,
    minimum_obstacle_clearance,
)
from greenhouse_inspection.viewpoint_validation import (
    collision_boxes_from_geometry,
)

from .static_greenhouse import rows, static_geometry
from .astar_route import plan_astar_route


MISSION_SCHEMA = "gps_greenhouse_multi_viewpoint_mission_result/v4"
SHOT_RESULT_SCHEMA = "gps_greenhouse_viewpoint_shot_result/v2"
HIGHLIGHT_VISIBLE_PIXEL_THRESHOLD = 5
# The inspection route flies above the canopy.
LIDAR_REACTIVE_MAX_BELOW_BODY_M = 0.75


def flight_level_lidar_points(
        points: np.ndarray,
        max_below_body_m: float = LIDAR_REACTIVE_MAX_BELOW_BODY_M) -> np.ndarray:
    """Keep LiDAR returns relevant to a vehicle travelling above the canopy."""
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must be an Nx3 array")
    if not math.isfinite(max_below_body_m) or max_below_body_m < 0.0:
        raise ValueError("max_below_body_m must be finite and non-negative")
    return array[array[:, 2] >= -float(max_below_body_m)]


def mission_elapsed_seconds(
        simulation_started_s: float | None,
        simulation_now_s: float | None,
        wall_started_s: float | None,
        wall_now_s: float) -> tuple[float, str]:
    """Return mission age, preferring PX4's simulation-clock timestamp."""
    if (simulation_started_s is not None and simulation_now_s is not None
            and math.isfinite(simulation_started_s)
            and math.isfinite(simulation_now_s)
            and simulation_now_s >= simulation_started_s):
        return simulation_now_s - simulation_started_s, "GAZEBO_SIMULATION_TIME"
    if wall_started_s is None:
        return 0.0, "NOT_STARTED"
    return max(0.0, wall_now_s - wall_started_s), "WALL_TIME_FALLBACK"


def build_abort_retrace_route(
        phase_routes: Sequence[np.ndarray], phase_index: int,
        waypoint_index: int, current_position: Sequence[float],
        current_yaw: float) -> np.ndarray:
    """Retrace already reviewed route segments to the airborne launch point."""
    if len(phase_routes) < 3:
        raise ValueError("at least three phase routes are required")
    routes = []
    for route in phase_routes:
        array = np.asarray(route, dtype=float)
        if (array.ndim != 2 or array.shape[1] < 4 or len(array) < 2
                or not np.isfinite(array[:, :3]).all()):
            raise ValueError(
                "phase routes must have finite XYZ and at least four columns")
        # PX4 accepts NaN yaw as "unspecified", and route_to_px4 can retain that value at the first point.
        routes.append(array[:, :4])
    phase = int(phase_index)
    return_phase = len(routes) - 1
    if not 0 <= phase <= return_phase:
        raise ValueError("phase_index is outside the mission phases")
    index = max(1, min(int(waypoint_index), len(routes[phase]) - 1))
    current = np.asarray(
        [*np.asarray(current_position, dtype=float), float(current_yaw)],
        dtype=float)
    if current.shape != (4,) or not np.isfinite(current).all():
        raise ValueError("current pose must contain finite xyz and yaw")

    if phase == 0:
        # index is the active target, so index-1 is the last reached point.
        steps = list(routes[0][max(1, index - 1):0:-1])
        if not steps:
            steps = [routes[0][1]]
    elif phase < return_phase:
        # Retrace the flown part of the active transition, then retrace every earlier completed transition and the first outbound.
        steps = list(routes[1][max(0, index - 1)::-1])
        if phase != 1:
            steps = list(routes[phase][max(0, index - 1)::-1])
            for previous_phase in range(phase - 1, 0, -1):
                steps.extend(routes[previous_phase][-2::-1])
        steps.extend(routes[0][-2:0:-1])
    else:
        # Already on the normal return corridor: continue toward the pad.
        steps = list(routes[return_phase][index:])
        if not steps:
            steps = [routes[return_phase][-1]]

    route = np.vstack((current, np.asarray(steps, dtype=float)))
    # Inspection yaw is irrelevant during an abort.
    route[:, 3] = float(current_yaw)
    deduplicated = [route[0]]
    for waypoint in route[1:]:
        if np.linalg.norm(waypoint[:3] - deduplicated[-1][:3]) > 1e-6:
            deduplicated.append(waypoint)
    if len(deduplicated) == 1:
        deduplicated.append(route[-1].copy())
    return np.asarray(deduplicated, dtype=float)


def _route_array(values: Sequence[Sequence[float]]) -> np.ndarray:
    route = np.asarray(values, dtype=float)
    if (route.ndim != 2 or route.shape[1] != 4 or len(route) < 2
            or not np.isfinite(route).all()):
        raise ValueError("route must be a finite Nx4 array with N >= 2")
    return route


def _deduplicate_route(route: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    """Drop adjacent duplicate positions while retaining the later yaw."""
    route = _route_array(route)
    kept = [route[0].copy()]
    for waypoint in route[1:]:
        if np.linalg.norm(waypoint[:3] - kept[-1][:3]) <= tolerance:
            kept[-1] = waypoint.copy()
        else:
            kept.append(waypoint.copy())
    return np.asarray(kept, dtype=float)


def _ground_launch_route(route: np.ndarray, ground_max_z: float = 0.5) -> bool:
    """Recognise the verified vertical ascent used by the GPS demo."""
    route = _route_array(route)
    return bool(
        len(route) >= 3
        and route[0, 2] <= float(ground_max_z)
        and np.linalg.norm(route[1, :2] - route[0, :2]) <= 1e-6
        and route[1, 2] > route[0, 2] + 0.5)


def _planned_connector(
        start: np.ndarray, goal: np.ndarray,
        collision_boxes: Iterable[Sequence[float]],
        route_planner: str) -> np.ndarray:
    """Plan and exactly validate one consecutive mission leg."""
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if start.shape != (4,) or goal.shape != (4,):
        raise ValueError("connector endpoints must be x/y/z/yaw poses")
    if not np.isfinite(start).all() or not np.isfinite(goal).all():
        raise ValueError("connector endpoints must be finite")
    if route_planner not in ("structured", "astar"):
        raise ValueError("route_planner must be structured or astar")
    boxes = tuple(tuple(float(value) for value in box)
                  for box in collision_boxes)

    if segment_blocked(start[:3], goal[:3], boxes) is None:
        route = np.vstack((start, goal))
    elif route_planner == "astar":
        route = plan_astar_route(start, goal, boxes).route
    else:
        planned = safe_route(
            [tuple(goal)], rows(), start=tuple(start[:3]), boxes=boxes)
        route = np.vstack((start, np.asarray(planned, dtype=float)))

    route = _deduplicate_route(route)
    if (np.linalg.norm(route[0, :3] - start[:3]) > 1e-9
            or np.linalg.norm(route[-1, :3] - goal[:3]) > 1e-9):
        raise RuntimeError("connector planner did not preserve its endpoints")
    for index, (begin, end) in enumerate(zip(route, route[1:])):
        if segment_blocked(begin[:3], end[:3], boxes) is not None:
            raise RuntimeError(
                "connector planner returned blocked segment %d" % index)
    return route


def build_multi_view_flight_routes(
        reviewed_routes: Sequence[np.ndarray],
        collision_boxes: Iterable[Sequence[float]],
        start_tolerance: float = 0.10,
        route_planner: str = "structured",
        return_order: bool = False):
    """Build a nearest-next, continuously planned multi-view flight."""
    routes = tuple(_route_array(route) for route in reviewed_routes)
    if not 2 <= len(routes) <= 8:
        raise ValueError(
            "a multi-view mission requires between two and eight routes")
    launch_reference = routes[0]
    if any(np.linalg.norm(launch_reference[0, :3] - route[0, :3])
           > float(start_tolerance) for route in routes[1:]):
        raise ValueError("preflights do not share the same launch position")
    if not all(_ground_launch_route(route) for route in routes):
        raise ValueError(
            "single-flight joining requires every route to contain the "
            "reviewed ground-to-air launch corridor")

    boxes = tuple(tuple(float(value) for value in box)
                  for box in collision_boxes)

    # The reviewed length includes any real detour required from launch.
    remaining = set(range(len(routes)))
    first_index = min(
        remaining, key=lambda index: (route_length(routes[index]), index))
    remaining.remove(first_index)
    order = [first_index]
    first = routes[first_index]
    phase_items = [("outbound_first", _deduplicate_route(first))]
    current = first[-1]

    ordinal_names = {2: "second", 3: "third", 4: "fourth"}
    for ordinal in range(2, len(routes) + 1):
        candidates = []
        for original_index in sorted(remaining):
            try:
                connector = _planned_connector(
                    current, routes[original_index][-1], boxes,
                    route_planner)
            except ValueError:
                continue
            candidates.append((
                route_length(connector), original_index, connector))
        if not candidates:
            raise ValueError(
                "no remaining viewpoint is reachable from the current view")
        _, selected_index, transition = min(
            candidates, key=lambda item: (item[0], item[1]))
        remaining.remove(selected_index)
        order.append(selected_index)
        current = routes[selected_index][-1]
        destination = ordinal_names.get(ordinal, "view_%02d" % ordinal)
        phase_items.append((
            "transition_to_%s" % destination,
            _deduplicate_route(transition)))

    # The final selected route supplies the reviewed airborne pad endpoint.
    airborne_pad = routes[order[-1]][1]
    return_route = _planned_connector(
        current, airborne_pad, boxes, route_planner)
    phase_items.append(("return_to_pad_airborne", return_route))
    phases = dict(phase_items)
    for name, route in phases.items():
        for index, (start, end) in enumerate(zip(route, route[1:])):
            obstacle = segment_blocked(start[:3], end[:3], boxes)
            if obstacle is not None:
                raise ValueError(
                    "%s creates blocked segment %d while joining preflights"
                    % (name, index))
    if return_order:
        return phases, tuple(order)
    return phases


def build_single_flight_routes(
        first_route: np.ndarray, second_route: np.ndarray,
        collision_boxes: Iterable[Sequence[float]],
        start_tolerance: float = 0.10,
        route_planner: str = "structured") -> dict[str, np.ndarray]:
    """Backward-compatible two-view wrapper used by existing callers/tests."""
    return build_multi_view_flight_routes(
        (first_route, second_route), collision_boxes, start_tolerance,
        route_planner)


def sampled_route_clearance(
        route: np.ndarray, boxes: Iterable[Sequence[float]],
        sample_spacing: float = 0.05) -> float:
    """Sample minimum extra clearance outside already-inflated obstacles."""
    route = _route_array(route)
    boxes = tuple(boxes)
    if sample_spacing <= 0.0 or not math.isfinite(sample_spacing):
        raise ValueError("sample_spacing must be positive and finite")
    minimum = math.inf
    for start, end in zip(route, route[1:]):
        distance = float(np.linalg.norm(end[:3] - start[:3]))
        count = max(1, int(math.ceil(distance / sample_spacing)))
        for fraction in np.linspace(0.0, 1.0, count + 1):
            point = start[:3] + fraction * (end[:3] - start[:3])
            minimum = min(minimum, minimum_obstacle_clearance(point, boxes))
    return float(minimum)


def selected_highlight_metrics(rgb_array) -> dict:
    """Measure the baked cyan hue without mistaking ordinary crop green."""
    image = np.asarray(rgb_array)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("highlight image must be an HxWx3 RGB array")
    red = image[..., 0].astype(int)
    green = image[..., 1].astype(int)
    blue = image[..., 2].astype(int)
    mask = (
        (green - red >= 15)
        & (blue - red >= 15)
        & (np.abs(green - blue) <= 80)
    )
    count = int(mask.sum())
    result = {
        "selected_plant_highlight_pixel_count": count,
        "selected_plant_highlight_image_fraction": float(count / mask.size),
        "selected_plant_highlight_visible": (
            count >= HIGHLIGHT_VISIBLE_PIXEL_THRESHOLD),
        "selected_plant_highlight_bbox_xyxy": None,
        "selected_plant_highlight_bbox_centre_xy_px": None,
        "selected_plant_highlight_centre_error_px": None,
        "selected_plant_highlight_centre_error_normalised": None,
        "selected_plant_highlight_edge_margins_px": None,
        "selected_plant_highlight_clipped_edges": [],
        "selected_plant_highlight_touches_image_edge": None,
        "selected_plant_highlight_fully_framed": None,
        "highlight_measurement_definition": (
            "cyan-hue pixels with green and blue >=15 above red and within "
            "80 intensity levels of one another"),
    }
    if count:
        y_values, x_values = np.nonzero(mask)
        x_min, x_max = int(x_values.min()), int(x_values.max())
        y_min, y_max = int(y_values.min()), int(y_values.max())
        result["selected_plant_highlight_bbox_xyxy"] = [
            x_min, y_min, x_max, y_max]
        bbox_centre_x = 0.5 * (x_min + x_max)
        bbox_centre_y = 0.5 * (y_min + y_max)
        image_centre_x = 0.5 * (mask.shape[1] - 1)
        image_centre_y = 0.5 * (mask.shape[0] - 1)
        centre_error = math.hypot(
            bbox_centre_x - image_centre_x,
            bbox_centre_y - image_centre_y,
        )
        image_half_diagonal = math.hypot(
            max(image_centre_x, 0.5), max(image_centre_y, 0.5))
        margins = {
            "left": x_min,
            "right": mask.shape[1] - 1 - x_max,
            "top": y_min,
            "bottom": mask.shape[0] - 1 - y_max,
        }
        clipped = [name for name, margin in margins.items() if margin == 0]
        visible = count >= HIGHLIGHT_VISIBLE_PIXEL_THRESHOLD
        result.update({
            "selected_plant_highlight_bbox_centre_xy_px": [
                bbox_centre_x, bbox_centre_y],
            "selected_plant_highlight_centre_error_px": centre_error,
            "selected_plant_highlight_centre_error_normalised": (
                centre_error / image_half_diagonal),
            "selected_plant_highlight_edge_margins_px": margins,
            "selected_plant_highlight_clipped_edges": clipped,
            "selected_plant_highlight_touches_image_edge": bool(clipped),
            "selected_plant_highlight_fully_framed": bool(
                visible and not clipped),
        })
    return result


def quaternion_to_rpy(quaternion) -> tuple[float, float, float]:
    """Convert PX4's w/x/y/z quaternion into roll, pitch and yaw radians."""
    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be non-zero")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sin_roll, cos_roll)
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(sin_yaw, cos_yaw)
    return roll, pitch, yaw


def point_to_polyline_distance(point, route) -> float:
    """Return shortest 3D distance from a point to a route polyline."""
    point = np.asarray(point, dtype=float)
    route = _route_array(route)[:, :3]
    minimum = math.inf
    for start, end in zip(route, route[1:]):
        segment = end - start
        denominator = float(segment @ segment)
        fraction = 0.0 if denominator <= 1e-12 else float(
            np.clip(((point - start) @ segment) / denominator, 0.0, 1.0))
        minimum = min(
            minimum,
            float(np.linalg.norm(point - (start + fraction * segment))),
        )
    return float(minimum)


def sample_statistics(values) -> dict:
    """Return a compact, JSON-safe description of finite scalar samples."""
    samples = np.asarray(tuple(values), dtype=float)
    samples = samples[np.isfinite(samples)]
    if not len(samples):
        return {
            "sample_count": 0,
            "mean_m": None,
            "median_m": None,
            "minimum_m": None,
            "maximum_m": None,
        }
    return {
        "sample_count": int(len(samples)),
        "mean_m": float(np.mean(samples)),
        "median_m": float(np.median(samples)),
        "minimum_m": float(np.min(samples)),
        "maximum_m": float(np.max(samples)),
    }


def load_reviewed_preflight(path: str) -> tuple[dict, np.ndarray]:
    """Load and cryptographically-equivalent-check one preflight route."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    route_path = record.get("route", {}).get("route_file")
    if not route_path:
        raise ValueError("preflight does not name its route file")
    route = validate_preflight_record(record, np.load(route_path))
    return record, route


def request_from_preflight(preflight: dict) -> PerspectiveRequest:
    request = preflight["request"]
    return PerspectiveRequest(
        math.radians(float(request["azimuth_deg"])),
        math.radians(float(request["elevation_deg"])),
        float(request["distance_m"]),
    )


def load_preflight_list(path: str) -> list[str]:
    """Load the ordered preflight paths written by the launcher."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    values = record.get("preflight_jsons")
    if not isinstance(values, list) or not values:
        raise ValueError("preflight list must contain preflight_jsons")
    paths = [str(value) for value in values]
    if any(not value for value in paths):
        raise ValueError("preflight list contains an empty path")
    return paths


def validate_mission_sources(
        preflights: Sequence[dict], maximum_targets: int = 4) -> list[dict]:
    """Validate one-crop multi-view or multi-crop two-sided shot groups."""
    groups = []
    group_by_key = {}
    for index, record in enumerate(preflights):
        source = record.get("auto_viewpoint", {})
        target_id = source.get("source_target_id")
        crop_id = source.get("source_crop_id")
        side = source.get("selected_side")
        if target_id is None or not crop_id or side not in (
                "negative_y", "positive_y"):
            raise ValueError(
                "every preflight must identify its crop and selected side")
        key = (int(target_id), str(crop_id))
        if key not in group_by_key:
            group = {
                "target_id": int(target_id),
                "crop_id": str(crop_id),
                "view_indices": [],
                "sides": [],
            }
            group_by_key[key] = group
            groups.append(group)
        group = group_by_key[key]
        group["view_indices"].append(index)
        group["sides"].append(str(side))

    if len(groups) > int(maximum_targets):
        raise ValueError(
            "multi-target mission exceeds the %d-crop safety limit"
            % int(maximum_targets))
    if len(groups) == 1:
        sides = groups[0]["sides"]
        if len(sides) == 2 and set(sides) == {"negative_y", "positive_y"}:
            return groups
        if (len(sides) == 4
                and sides.count("negative_y") == 2
                and sides.count("positive_y") == 2):
            pose_keys = [(
                record.get("auto_viewpoint", {}).get("selected_side"),
                record.get("auto_viewpoint", {}).get(
                    "selected_azimuth_deg"),
            ) for record in preflights]
            if len(set(pose_keys)) != 4:
                raise ValueError(
                    "four-view preflights contain duplicate poses")
            return groups
        raise ValueError(
            "a single crop requires two opposite-side or four distinct views")

    for group in groups:
        if (len(group["view_indices"]) != 2
                or set(group["sides"]) != {"negative_y", "positive_y"}):
            raise ValueError(
                "each crop in a multi-target mission requires two opposite-side views")
    return groups


def planned_mission_metrics(phases, boxes) -> dict:
    lengths = {name: route_length(route) for name, route in phases.items()}
    clearances = {
        name: sampled_route_clearance(route, boxes)
        for name, route in phases.items()
    }
    return {
        "route_length_m": lengths,
        "total_route_length_before_landing_m": float(sum(lengths.values())),
        "minimum_clearance_to_inflated_obstacles_m": float(
            min(clearances.values())),
        "phase_minimum_clearance_to_inflated_obstacles_m": clearances,
        "clearance_definition": (
            "vehicle-centre distance outside the planner's already-inflated "
            "crop/structure collision boxes; zero is the safety boundary"),
        "takeoffs_planned": 1,
        "landings_planned": 1,
        "captures_planned": max(0, len(phases) - 1),
    }


def write_mission_csv(result: dict, output_path: str) -> None:
    """Write one analysis-ready row per planned side shot."""
    fields = [
        "mission_id", "mission_status", "target_id", "crop_id",
        "shot_index", "side", "viewpoint_method", "route_planner",
        "shot_status", "image_path",
        "planner_time_ms", "attempted_candidates",
        "physically_feasible_candidates", "fully_framed_candidates",
        "planned_distance_m", "planned_pitch_deg", "planned_vehicle_yaw_deg",
        "predicted_image_area_fraction", "viewpoint_clearance_m",
        "capture_time_from_start_s", "actual_distance_m",
        "achieved_distance_error_m", "achieved_direction_error_deg",
        "optical_axis_error_deg", "actual_gimbal_yaw_deg",
        "actual_gimbal_pitch_deg", "image_width_px", "image_height_px",
        "image_file_bytes", "mission_duration_s", "actual_path_length_m",
        "minimum_clearance_to_inflated_obstacles_m", "landing_outcome",
        "selected_plant_highlight_pixel_count",
        "selected_plant_highlight_image_fraction",
        "selected_plant_highlight_visible",
        "selected_plant_highlight_touches_image_edge",
        "selected_plant_highlight_fully_framed",
        "selected_plant_highlight_centre_error_px",
        "selected_plant_highlight_centre_error_normalised",
        "selected_plant_highlight_edge_margins_px",
        "selected_plant_highlight_clipped_edges",
        "actual_vehicle_roll_deg", "actual_vehicle_pitch_deg",
        "actual_vehicle_yaw_deg", "actual_linear_speed_mps",
        "actual_angular_speed_rad_s", "post_settle_delay_s",
        "route_cross_track_error_m",
    ]
    mission = result.get("execution", {})
    rows = []
    for index, shot in enumerate(result.get("shots", []), start=1):
        planned = shot.get("planned", {})
        projection = planned.get("predicted_plant_projection", {})
        rows.append({
            "mission_id": result.get("mission_id"),
            "mission_status": result.get("status"),
            "target_id": shot.get(
                "target_id", planned.get(
                    "source_target_id", result.get("target_id"))),
            "crop_id": shot.get(
                "crop_id", planned.get(
                    "source_crop_id", result.get("crop_id"))),
            "shot_index": index,
            "side": shot.get("side"),
            "viewpoint_method": planned.get("mode"),
            "route_planner": planned.get("route_planner", "structured"),
            "shot_status": shot.get("status"),
            "image_path": shot.get("image_path"),
            "planner_time_ms": planned.get("planner_time_ms"),
            "attempted_candidates": planned.get("attempted_candidates"),
            "physically_feasible_candidates": planned.get(
                "physically_feasible_candidates"),
            "fully_framed_candidates": planned.get(
                "fully_framed_candidates"),
            "planned_distance_m": planned.get("selected_distance_m"),
            "planned_pitch_deg": planned.get("selected_pitch_deg"),
            "planned_vehicle_yaw_deg": planned.get(
                "selected_vehicle_yaw_deg"),
            "predicted_image_area_fraction": projection.get(
                "image_area_fraction"),
            "viewpoint_clearance_m": shot.get("viewpoint_clearance_m"),
            "capture_time_from_start_s": shot.get(
                "capture_time_from_start_s"),
            "actual_distance_m": shot.get("actual_distance_m"),
            "achieved_distance_error_m": shot.get(
                "achieved_distance_error_m"),
            "achieved_direction_error_deg": shot.get(
                "achieved_direction_error_deg"),
            "optical_axis_error_deg": shot.get("optical_axis_error_deg"),
            "actual_gimbal_yaw_deg": shot.get("actual_gimbal_yaw_deg"),
            "actual_gimbal_pitch_deg": shot.get("actual_gimbal_pitch_deg"),
            "image_width_px": shot.get("image_width_px"),
            "image_height_px": shot.get("image_height_px"),
            "image_file_bytes": shot.get("image_file_bytes"),
            "mission_duration_s": mission.get("mission_duration_s"),
            "actual_path_length_m": mission.get("actual_path_length_m"),
            "minimum_clearance_to_inflated_obstacles_m": mission.get(
                "minimum_clearance_to_inflated_obstacles_m"),
            "landing_outcome": mission.get("landing_outcome"),
            "selected_plant_highlight_pixel_count": shot.get(
                "selected_plant_highlight_pixel_count"),
            "selected_plant_highlight_image_fraction": shot.get(
                "selected_plant_highlight_image_fraction"),
            "selected_plant_highlight_visible": shot.get(
                "selected_plant_highlight_visible"),
            "selected_plant_highlight_touches_image_edge": shot.get(
                "selected_plant_highlight_touches_image_edge"),
            "selected_plant_highlight_fully_framed": shot.get(
                "selected_plant_highlight_fully_framed"),
            "selected_plant_highlight_centre_error_px": shot.get(
                "selected_plant_highlight_centre_error_px"),
            "selected_plant_highlight_centre_error_normalised": shot.get(
                "selected_plant_highlight_centre_error_normalised"),
            "selected_plant_highlight_edge_margins_px": json.dumps(
                shot.get("selected_plant_highlight_edge_margins_px")),
            "selected_plant_highlight_clipped_edges": json.dumps(
                shot.get("selected_plant_highlight_clipped_edges")),
            "actual_vehicle_roll_deg": shot.get("actual_vehicle_roll_deg"),
            "actual_vehicle_pitch_deg": shot.get("actual_vehicle_pitch_deg"),
            "actual_vehicle_yaw_deg": shot.get("actual_vehicle_yaw_deg"),
            "actual_linear_speed_mps": shot.get("actual_linear_speed_mps"),
            "actual_angular_speed_rad_s": shot.get(
                "actual_angular_speed_rad_s"),
            "post_settle_delay_s": shot.get("post_settle_delay_s"),
            "route_cross_track_error_m": shot.get(
                "route_cross_track_error_m"),
        })
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(args=None):
    """Run the two-shot mission node; flight remains opt-in."""
    import os

    from geometry_msgs.msg import PointStamped
    from rosgraph_msgs.msg import Clock
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from rclpy.time import Time
    from sensor_msgs.msg import Image, JointState, PointCloud2
    from sensor_msgs_py import point_cloud2 as pc2
    from std_msgs.msg import Bool
    from tf2_ros import Buffer, TransformListener
    from tf2_ros import (
        ConnectivityException,
        ExtrapolationException,
        LookupException,
    )

    from px4_msgs.msg import (
        OffboardControlMode,
        TrajectorySetpoint,
        VehicleCommand,
        VehicleOdometry,
        VehicleStatus,
    )

    from greenhouse_inspection.assisted_teleop import (
        clamp_to_obstacle,
        clamp_vertical,
    )
    from greenhouse_inspection.coverage_flight import (
        MAX_YAW_RATE,
        _body_flu_to_ned,
        _ned_to_body_flu,
        is_stuck,
        image_to_array,
        ramped_yaw,
        save_image,
        sidestep_scale,
    )
    from greenhouse_inspection.gimbal_controller import PITCH_JOINT, YAW_JOINT
    from greenhouse_inspection.path_follower import REACH_RADIUS

    tick_period = 0.1
    settle_samples_required = 15
    stuck_ticks_limit = 30
    sidestep_distance = 2.0

    class TwoViewpointMission(Node):
        def __init__(self):
            super().__init__("gps_two_viewpoint_mission")
            legacy_preflight_paths = [
                str(self.declare_parameter("first_preflight_json", "").value),
                str(self.declare_parameter("second_preflight_json", "").value),
                str(self.declare_parameter("third_preflight_json", "").value),
                str(self.declare_parameter("fourth_preflight_json", "").value),
            ]
            preflight_list_json = str(self.declare_parameter(
                "preflight_list_json", "").value)
            self.preflight_paths = (
                load_preflight_list(preflight_list_json)
                if preflight_list_json else
                [path for path in legacy_preflight_paths if path])
            self.capture_count = len(self.preflight_paths)
            self.return_phase_index = self.capture_count
            self.maximum_targets = int(self.declare_parameter(
                "maximum_targets", 4).value)
            default_root = Path(os.environ.get(
                "THESIS_DIR", str(Path.cwd())))
            self.result_path = str(self.declare_parameter(
                "result_json",
                str(default_root / "gps_two_viewpoint_result.json")).value)
            self.result_csv = str(self.declare_parameter(
                "result_csv",
                str(default_root / "gps_two_viewpoint_result.csv")).value)
            self.out_dir = str(self.declare_parameter(
                "out_dir", str(default_root / "gps_viewpoint_photos")).value)
            self.start_flight = bool(self.declare_parameter(
                "start_flight", False).value)
            self.use_lidar_safety = bool(self.declare_parameter(
                "use_lidar_safety", False).value)
            self.max_preflight_age = float(self.declare_parameter(
                "max_preflight_age_sec", 600.0).value)
            self.max_transform_translation = float(self.declare_parameter(
                "max_transform_translation_change", 0.25).value)
            self.max_transform_yaw = math.radians(float(self.declare_parameter(
                "max_transform_yaw_change_deg", 10.0).value))
            self.max_start_distance = float(self.declare_parameter(
                "max_start_distance", 0.75).value)
            self.max_start_yaw = math.radians(float(self.declare_parameter(
                "max_start_yaw_error_deg", 30.0).value))
            self.vehicle_settle_distance = float(self.declare_parameter(
                "vehicle_settle_distance", 0.20).value)
            self.vehicle_settle_yaw = math.radians(float(
                self.declare_parameter("vehicle_settle_yaw_deg", 5.0).value))
            self.vehicle_settle_speed = float(self.declare_parameter(
                "vehicle_settle_speed_mps", 0.15).value)
            self.vehicle_settle_angular_speed = float(self.declare_parameter(
                "vehicle_settle_angular_speed_rad_s", 0.15).value)
            self.post_settle_delay = float(self.declare_parameter(
                "post_settle_delay_sec", 0.75).value)
            self.capture_timeout = float(self.declare_parameter(
                "capture_timeout_sec", 30.0).value)
            self.landing_timeout = float(self.declare_parameter(
                "landing_timeout_sec", 45.0).value)
            self.mission_timeout = float(self.declare_parameter(
                "mission_timeout_sec", 240.0).value)
            self.safe_return_timeout = float(self.declare_parameter(
                "safe_return_timeout_sec", 60.0).value)
            self.runtime_clearance_abort_samples = int(
                self.declare_parameter(
                    "runtime_clearance_abort_samples", 5).value)
            self.sim_ground_disarm = bool(self.declare_parameter(
                "sim_ground_disarm", True).value)
            self.ground_launch_altitude = float(self.declare_parameter(
                "ground_launch_altitude_m", 1.2).value)
            self.ground_launch_speed = float(self.declare_parameter(
                "ground_launch_speed_mps", 0.45).value)
            self.ground_launch_max_saved_z = float(self.declare_parameter(
                "ground_launch_max_saved_z", 0.5).value)

            if not 2 <= self.capture_count <= 8:
                raise ValueError(
                    "between two and eight preflight JSON paths are required")
            loaded = [load_reviewed_preflight(path)
                      for path in self.preflight_paths]
            for record, _ in loaded:
                if (self.max_preflight_age > 0.0
                        and preflight_age_seconds(record)
                        > self.max_preflight_age):
                    raise ValueError("preflight is stale; generate a fresh set")

            configured_planners = {
                record.get("auto_viewpoint", {}).get(
                    "route_planner", "structured")
                for record, _ in loaded
            }
            if len(configured_planners) != 1:
                raise ValueError(
                    "all preflights in one mission must use the same route planner")
            self.route_planner = configured_planners.pop()

            self.boxes = collision_boxes_from_geometry(static_geometry())
            self.crop_boxes = self.boxes[:len(rows())]
            self.structure_boxes = self.boxes[len(rows()):]
            self.saved_phases, execution_order = build_multi_view_flight_routes(
                [item[1] for item in loaded],
                self.boxes,
                route_planner=self.route_planner,
                return_order=True,
            )
            loaded = [loaded[index] for index in execution_order]
            self.preflight_paths = [
                self.preflight_paths[index] for index in execution_order]
            self.preflights = [item[0] for item in loaded]
            self.saved_routes = [item[1] for item in loaded]
            sources = [record.get("auto_viewpoint", {})
                       for record in self.preflights]
            first_source = sources[0]
            self.target_groups = validate_mission_sources(
                self.preflights, self.maximum_targets)
            self.target_count = len(self.target_groups)

            self.phase_names = tuple(self.saved_phases)
            self.planned_metrics = planned_mission_metrics(
                self.saved_phases, self.boxes)
            self.targets_saved = [np.asarray(
                record["target_saved_map"], dtype=float)
                for record in self.preflights]
            self.requests = [request_from_preflight(record)
                             for record in self.preflights]
            self.recorded_transform = self.preflights[0][
                "pose_source"]["saved_from_live_xyzyaw"]
            other_transforms = [record["pose_source"][
                "saved_from_live_xyzyaw"] for record in self.preflights[1:]]
            if not all(np.allclose(
                    self.recorded_transform, transform, atol=1e-9)
                    for transform in other_transforms):
                raise ValueError("preflights were not generated from one pose snapshot")

            source = first_source
            self.mission_id = datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%S_%fZ")
            self.target_id = (
                source.get("source_target_id")
                if self.target_count == 1 else None)
            self.crop_id = (
                source.get("source_crop_id")
                if self.target_count == 1 else None)
            self.result = {
                "schema": MISSION_SCHEMA,
                "mission_id": self.mission_id,
                "status": "VALIDATED_NOT_STARTED",
                "detail": "",
                "timestamp_created_utc": datetime.now(timezone.utc).isoformat(),
                "target_id": self.target_id,
                "crop_id": self.crop_id,
                "target_count": self.target_count,
                "targets": deepcopy(self.target_groups),
                "flight_commanded": False,
                "localisation": "PX4 EKF using Gazebo simulated GPS and IMU",
                "slam_used": False,
                "safety_mode": (
                    "STATIC_MAP_PLUS_REACTIVE_LIDAR"
                    if self.use_lidar_safety else "STATIC_MAP_ONLY"),
                "simulation_ground_truth_highlight": {
                    "enabled": True,
                    "style": (
                        "translucent cyan duplicate of every selected plant mesh"),
                    "highlighted_target_count": self.target_count,
                    "per_shot_target_specific": self.target_count == 1,
                    "purpose": (
                        "verify selected-crop presence; multi-target masks may "
                        "contain another selected crop when both are co-visible"),
                    "not_natural_image_evidence": True,
                    "measured_post_capture_only": True,
                    "used_by_viewpoint_planner": False,
                    "used_by_flight_controller": False,
                },
                "preflight_jsons": self.preflight_paths,
                "viewpoint_execution_order_original_indices": [
                    int(index) for index in execution_order],
                "route_joining": {
                    "ordering": "nearest reachable next viewpoint",
                    "planner": self.route_planner,
                    "direct_segment_preferred_when_collision_free": True,
                    "intermediate_launch_return": False,
                    "single_final_return": True,
                },
                "view_count": self.capture_count,
                "planned": self.planned_metrics,
                "shots": [],
                "execution": {
                    "takeoff_count": 0,
                    "landing_request_count": 0,
                    "landing_outcome": "NOT_REQUESTED",
                    "mission_duration_s": None,
                    "mission_duration_wall_s": None,
                    "mission_duration_sim_s": None,
                    "mission_timeout_clock": (
                        "Gazebo /clock simulation time"),
                    "safe_abort_return_requested": False,
                    "safe_abort_reason": None,
                    "actual_path_length_m": 0.0,
                    "minimum_clearance_to_inflated_obstacles_m": None,
                    "minimum_clearance_to_inflated_canopy_m": None,
                    "minimum_clearance_to_inflated_structure_m": None,
                    "inflated_obstacle_violation_samples": 0,
                    "inflated_obstacle_violation_duration_s": 0.0,
                    "route_cross_track_error_mean_m": None,
                    "route_cross_track_error_max_m": None,
                    "route_cross_track_error_by_phase_m": {},
                    "ground_launch_horizontal_error_m": {},
                    "landing_horizontal_error_m": {},
                    "lidar_safety": {
                        "enabled": self.use_lidar_safety,
                        "clamped_setpoint_count": 0,
                        "clamp_event_count": 0,
                        "clamp_events": [],
                        "sidestep_count": 0,
                        "sidestep_events": [],
                        "hold_count": 0,
                        "hold_events": [],
                        "stale_data_hold_count": 0,
                    },
                    "trajectory_sample_count": 0,
                },
                "deferred_changes": [
                    "Expand one area cell into every intersecting crop target",
                    "Replace the provisional camera with a validated camera/zoom profile",
                ],
            }
            self.completed = False
            if not self.start_flight:
                self.result["status"] = "EXECUTION_DISABLED"
                self.result["detail"] = (
                    "start_flight=false; all preflights and the combined "
                    "single-flight route were validated without ROS flight publishers")
                self.write_result()
                self.get_logger().warn(
                    "START INTERLOCK: combined %d-view mission accepted, but "
                    "start_flight=false; no flight publisher was created."
                    % self.capture_count)
                self.completed = True
                return

            os.makedirs(self.out_dir, exist_ok=True)
            pub_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST, depth=1)
            sub_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST, depth=5)
            camera_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST, depth=5)
            self.offboard_pub = self.create_publisher(
                OffboardControlMode, "/fmu/in/offboard_control_mode", pub_qos)
            self.setpoint_pub = self.create_publisher(
                TrajectorySetpoint, "/fmu/in/trajectory_setpoint", pub_qos)
            self.command_pub = self.create_publisher(
                VehicleCommand, "/fmu/in/vehicle_command", pub_qos)
            self.gimbal_pub = self.create_publisher(
                PointStamped, "/gimbal/look_at", 10)
            self.create_subscription(
                VehicleOdometry, "/fmu/out/vehicle_odometry",
                self.odom_cb, sub_qos)
            self.create_subscription(
                Clock, "/clock", self.clock_cb, sub_qos)
            self.create_subscription(
                VehicleStatus, "/fmu/out/vehicle_status_v1",
                self.status_cb, sub_qos)
            self.lidar_subscription = None
            if self.use_lidar_safety:
                self.lidar_subscription = self.create_subscription(
                    PointCloud2, "/lidar/points", self.lidar_cb, sub_qos)
            self.create_subscription(
                Image, "/camera", self.image_cb, camera_qos)
            self.create_subscription(
                Bool, "/gimbal/settled", self.gimbal_cb, sub_qos)
            self.create_subscription(
                JointState, "/gimbal/joint_states", self.joints_cb, sub_qos)
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.tf_errors = (
                LookupException, ConnectivityException, ExtrapolationException)

            self.position = None
            self.velocity = None
            self.angular_velocity = None
            self.current_roll = 0.0
            self.current_pitch = 0.0
            self.current_yaw = 0.0
            self.armed = False
            self.ever_armed = False
            self.nav_state = None
            self.latest_points = np.zeros((0, 3), dtype=np.float32)
            self.lidar_time = None
            self.latest_image = None
            self.image_time = None
            self.actual_gimbal_yaw = None
            self.actual_gimbal_pitch = None
            self.gimbal_settled = False
            self.saved_from_live = None
            self.phase_routes_px4 = None
            self.phase_index = 0
            self.waypoints = None
            self.index = 0
            self.shot_index = 0
            self.holding_for_capture = False
            self.hold_started = None
            self.vehicle_good_samples = 0
            self.gimbal_target_sent = False
            self.settled_since = None
            self.capture_ready_since = None
            self.commanded_yaw = float("nan")
            self.tick = 0
            self.last_engage_tick = None
            self.stuck_ticks = 0
            self.prev_distance = float("inf")
            self.detour_target = None
            self.landing = False
            self.landing_started = None
            self.last_land_command = None
            self.landing_low_since = None
            self.last_disarm_command = None
            self.ground_takeoff = False
            self.takeoff_complete = True
            self.takeoff_origin = None
            self.mission_started = None
            self.mission_started_sim = None
            self.simulation_time_seconds = None
            self.returning_after_abort = False
            self.abort_reason = None
            self.safe_return_started = None
            self.static_violation_streak = 0
            self.trajectory = []
            self.actual_min_clearance = math.inf
            self.actual_min_canopy_clearance = math.inf
            self.actual_min_structure_clearance = math.inf
            self.violation_samples = 0
            self.violation_duration = 0.0
            self.last_odom_time = None
            self.cross_track_errors = []
            self.phase_cross_track_errors = {
                name: [] for name in self.phase_names}
            self.ground_launch_horizontal_errors = []
            self.landing_horizontal_errors = []
            self.lidar_clamped_count = 0
            self.current_lidar_clamp_event = None
            self.lidar_clamp_events = []
            self.lidar_sidestep_count = 0
            self.lidar_sidestep_events = []
            self.lidar_hold_count = 0
            self.lidar_hold_events = []
            self.lidar_stale_count = 0
            self.timer = self.create_timer(tick_period, self.timer_callback)
            safety_description = (
                "static map plus reactive LiDAR"
                if self.use_lidar_safety else "static-map-only safety")
            self.get_logger().warn(
                "START INTERLOCK RELEASED: %d views will run in ONE flight; "
                "waiting for fresh TF and matching start pose; safety=%s."
                % (self.capture_count, safety_description))

        def now_seconds(self):
            return self.get_clock().now().nanoseconds / 1e9

        def clock_cb(self, message):
            self.simulation_time_seconds = (
                float(message.clock.sec)
                + float(message.clock.nanosec) / 1e9)

        def odom_cb(self, message):
            self.position = np.asarray(message.position, dtype=float)
            self.velocity = np.asarray(message.velocity, dtype=float)
            self.angular_velocity = np.asarray(
                message.angular_velocity, dtype=float)
            self.current_roll, self.current_pitch, self.current_yaw = (
                quaternion_to_rpy(tuple(message.q)))
            if self.saved_from_live is None:
                return
            now = self.now_seconds()
            pose = px4_pose_in_saved_map(
                self.position, self.current_yaw, self.saved_from_live)
            point = np.asarray(pose[:3], dtype=float)
            clearance = minimum_obstacle_clearance(point, self.boxes)
            canopy_clearance = minimum_obstacle_clearance(
                point, self.crop_boxes)
            structure_clearance = minimum_obstacle_clearance(
                point, self.structure_boxes)
            airborne_route = (
                self.mission_started is not None
                and self.takeoff_complete and not self.landing)
            if airborne_route:
                self.actual_min_clearance = min(
                    self.actual_min_clearance, clearance)
                self.actual_min_canopy_clearance = min(
                    self.actual_min_canopy_clearance, canopy_clearance)
                self.actual_min_structure_clearance = min(
                    self.actual_min_structure_clearance, structure_clearance)
            if clearance <= 1e-9 and airborne_route:
                self.violation_samples += 1
                if airborne_route:
                    self.static_violation_streak += 1
                if self.last_odom_time is not None:
                    self.violation_duration += max(
                        0.0, now - self.last_odom_time)
            else:
                self.static_violation_streak = 0
            self.last_odom_time = now
            # Route tracking starts only after mission release.
            if self.mission_started is not None:
                launch = np.asarray(self.saved_routes[0][0, :3], dtype=float)
                horizontal_launch_error = float(np.linalg.norm(
                    point[:2] - launch[:2]))
                if self.landing:
                    self.landing_horizontal_errors.append(
                        horizontal_launch_error)
                elif not self.takeoff_complete:
                    self.ground_launch_horizontal_errors.append(
                        horizontal_launch_error)
                elif (not self.holding_for_capture
                        and not self.returning_after_abort
                        and self.phase_index < len(self.phase_names)):
                    phase_name = self.phase_names[self.phase_index]
                    route = self.saved_phases[phase_name]
                    error = point_to_polyline_distance(point, route)
                    self.cross_track_errors.append(error)
                    self.phase_cross_track_errors[phase_name].append(error)
            if (not self.trajectory
                    or np.linalg.norm(point - self.trajectory[-1]) >= 0.02):
                self.trajectory.append(point)

        def status_cb(self, message):
            self.armed = (
                message.arming_state == VehicleStatus.ARMING_STATE_ARMED)
            self.nav_state = message.nav_state
            was_ever_armed = self.ever_armed
            self.ever_armed = self.ever_armed or self.armed
            if self.ever_armed and not was_ever_armed:
                self.result["execution"]["takeoff_count"] = 1
            if self.landing and self.ever_armed and not self.armed:
                self.result["execution"]["landing_outcome"] = "DISARMED"
                images = sum(
                    shot.get("status") == "IMAGE_CAPTURED"
                    for shot in self.result["shots"])
                if self.abort_reason:
                    self.finalise(
                        "MISSION_ABORTED_SAFE_RETURN",
                        "%s; %d/%d images captured; vehicle returned to the "
                        "launch pad and disarmed" % (
                            self.abort_reason, images, self.capture_count),
                    )
                else:
                    status = (
                        "MISSION_COMPLETE" if images == self.capture_count
                        else "MISSION_COMPLETE_WITH_CAPTURE_FAILURE")
                    self.finalise(
                        status, "%d/%d images captured; vehicle disarmed" % (
                            images, self.capture_count))

        def lidar_cb(self, message):
            points = pc2.read_points(
                message, field_names=("x", "y", "z"), skip_nans=True)
            self.latest_points = np.column_stack(
                [points["x"], points["y"], points["z"]]).astype(np.float32)
            self.lidar_time = self.now_seconds()

        def image_cb(self, message):
            self.latest_image = message
            self.image_time = self.now_seconds()

        def gimbal_cb(self, message):
            self.gimbal_settled = bool(message.data)

        def joints_cb(self, message):
            positions = dict(zip(message.name, message.position))
            if YAW_JOINT in positions and PITCH_JOINT in positions:
                self.actual_gimbal_yaw = float(positions[YAW_JOINT])
                self.actual_gimbal_pitch = joint_pitch_to_camera(
                    positions[PITCH_JOINT])

        def lookup_saved_from_live(self):
            message = self.tf_buffer.lookup_transform(
                "saved_map", "map", Time())
            t = message.transform.translation
            q = message.transform.rotation
            return transform_from_tf(
                (t.x, t.y, t.z), (q.x, q.y, q.z, q.w))

        def try_accept_live_state(self):
            if (self.position is None
                    or self.simulation_time_seconds is None
                    or (self.use_lidar_safety and self.lidar_time is None)):
                return False
            if (self.use_lidar_safety
                    and self.now_seconds() - self.lidar_time > 1.0):
                return False
            try:
                transform = self.lookup_saved_from_live()
            except self.tf_errors:
                return False
            translation_change, yaw_change = transform_change(
                self.recorded_transform, transform)
            if (translation_change > self.max_transform_translation
                    or yaw_change > self.max_transform_yaw):
                self.abort_before_flight(
                    "map transform changed since preflight: %.3f m, %.2f deg"
                    % (translation_change, math.degrees(yaw_change)))
                return False
            current_saved = px4_pose_in_saved_map(
                self.position, self.current_yaw, transform)
            distance, yaw_error = pose_change(
                current_saved, self.saved_routes[0][0])
            if (distance > self.max_start_distance
                    or yaw_error > self.max_start_yaw):
                self.abort_before_flight(
                    "current pose no longer matches preflight start: %.3f m, %.2f deg"
                    % (distance, math.degrees(yaw_error)))
                return False

            outbound = insert_ground_launch_waypoint(
                self.saved_phases["outbound_first"],
                self.ground_launch_altitude)
            phase_routes = [outbound] + [
                self.saved_phases[name] for name in self.phase_names[1:]]
            dense_saved = []
            for phase_index, route in enumerate(phase_routes):
                marked = np.column_stack((
                    route, np.zeros(len(route), dtype=float)))
                dense = densify_route(marked)
                # Keep the explicit launch target intact so SITL leaves the pad.
                if phase_index == 0 and len(outbound) > len(
                        self.saved_phases["outbound_first"]):
                    tail = densify_route(marked[1:])
                    dense = np.vstack((marked[:2], tail[1:]))
                dense_saved.append(dense)
            self.saved_from_live = transform
            self.phase_routes_px4 = [
                route_to_px4(route, transform) for route in dense_saved]
            self.phase_index = 0
            self.waypoints = self.phase_routes_px4[0]
            self.index = 1
            self.ground_takeoff = len(outbound) > len(
                self.saved_phases["outbound_first"])
            self.takeoff_complete = not self.ground_takeoff
            self.takeoff_origin = self.position.copy()
            self.mission_started = self.now_seconds()
            self.mission_started_sim = self.simulation_time_seconds
            self.result["flight_commanded"] = True
            self.result["status"] = "IN_PROGRESS"
            self.write_result()
            self.get_logger().info(
                "EXECUTION_READY: one takeoff, %d captures, one landing; "
                "dense phase waypoints=%s." % (
                    self.capture_count,
                    [len(route) for route in self.phase_routes_px4]))
            return True

        def mission_elapsed(self):
            return mission_elapsed_seconds(
                self.mission_started_sim,
                self.simulation_time_seconds,
                self.mission_started,
                self.now_seconds(),
            )

        def abort_before_flight(self, reason):
            self.result["status"] = "EXECUTION_REFUSED"
            self.result["detail"] = reason
            self.write_result()
            self.get_logger().error("EXECUTION_REFUSED: %s" % reason)
            self.completed = True
            self.timer.cancel()

        def stamp(self):
            return int(self.get_clock().now().nanoseconds / 1000)

        def publish_offboard_mode(self):
            message = OffboardControlMode()
            message.timestamp = self.stamp()
            message.position = True
            self.offboard_pub.publish(message)

        def publish_offboard_velocity_mode(self):
            message = OffboardControlMode()
            message.timestamp = self.stamp()
            message.velocity = True
            self.offboard_pub.publish(message)

        def publish_setpoint(self, xyz, yaw):
            message = TrajectorySetpoint()
            message.timestamp = self.stamp()
            message.position = [float(value) for value in xyz]
            message.yaw = float(yaw)
            self.setpoint_pub.publish(message)

        def publish_velocity_setpoint(self, velocity_ned):
            message = TrajectorySetpoint()
            message.timestamp = self.stamp()
            message.position = [float("nan")] * 3
            message.velocity = [float(value) for value in velocity_ned]
            message.yaw = float("nan")
            message.yawspeed = 0.0
            self.setpoint_pub.publish(message)

        def publish_command(self, command, **parameters):
            message = VehicleCommand()
            message.timestamp = self.stamp()
            message.command = command
            message.param1 = float(parameters.get("param1", 0.0))
            message.param2 = float(parameters.get("param2", 0.0))
            message.target_system = 1
            message.target_component = 1
            message.source_system = 1
            message.source_component = 1
            message.from_external = True
            self.command_pub.publish(message)

        def engage_watchdog(self):
            ready = (self.armed and self.nav_state
                     == VehicleStatus.NAVIGATION_STATE_OFFBOARD)
            can_attempt = (self.last_engage_tick is None
                           or self.tick - self.last_engage_tick >= 10)
            if self.tick >= 20 and not ready and can_attempt:
                self.publish_command(
                    VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                    param1=1.0, param2=6.0)
                self.publish_command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                    param1=1.0)
                self.last_engage_tick = self.tick

        def publish_gimbal_target(self):
            target = saved_target_in_body(
                self.targets_saved[self.shot_index], self.saved_from_live,
                self.position, self.current_yaw)
            message = PointStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = "base_link"
            message.point.x, message.point.y, message.point.z = target
            self.gimbal_pub.publish(message)

        def find_sidestep(self):
            reactive_points = flight_level_lidar_points(self.latest_points)
            for dx, dy, dz in (
                    (0.0, sidestep_distance, 0.0),
                    (0.0, -sidestep_distance, 0.0),
                    (0.0, 0.0, sidestep_distance)):
                scale = sidestep_scale(
                    dx, dy, dz, reactive_points, -self.position[2])
                if scale >= 0.99:
                    n, e, d = _body_flu_to_ned(
                        dx, dy, dz, self.current_yaw)
                    return self.position + np.array([n, e, d])
            return None

        def current_phase_label(self):
            if self.landing:
                return "landing"
            if not self.takeoff_complete:
                return "ground_launch"
            if self.holding_for_capture:
                return "capture_%d" % (self.shot_index + 1)
            if self.returning_after_abort:
                return "safe_abort_return"
            if self.phase_index < len(self.phase_names):
                return self.phase_names[self.phase_index]
            return "unknown"

        def event_time(self):
            mission_time = (
                self.mission_elapsed()[0]
                if self.mission_started is not None else None)
            return {
                "mission_time_s": mission_time,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "phase": self.current_phase_label(),
            }

        def update_lidar_clamp_event(self, active, scale=None):
            """Group consecutive clamped control ticks into one event."""
            if active:
                if self.current_lidar_clamp_event is None:
                    self.current_lidar_clamp_event = {
                        "start": self.event_time(),
                        "end": None,
                        "duration_s": None,
                        "minimum_scale": float(scale),
                        "complete": False,
                    }
                else:
                    self.current_lidar_clamp_event["minimum_scale"] = min(
                        self.current_lidar_clamp_event["minimum_scale"],
                        float(scale),
                    )
                return
            if self.current_lidar_clamp_event is None:
                return
            event = self.current_lidar_clamp_event
            event["end"] = self.event_time()
            start_time = event["start"].get("mission_time_s")
            end_time = event["end"].get("mission_time_s")
            if start_time is not None and end_time is not None:
                event["duration_s"] = max(0.0, end_time - start_time)
            event["complete"] = True
            self.lidar_clamp_events.append(event)
            self.current_lidar_clamp_event = None

        def safe_setpoint(self, target):
            distance = float(np.linalg.norm(self.position - target))
            if not self.use_lidar_safety:
                self.update_lidar_clamp_event(False)
                self.detour_target = None
                return target, distance
            if self.detour_target is not None:
                self.update_lidar_clamp_event(False)
                if float(np.linalg.norm(
                        self.position - self.detour_target)) < REACH_RADIUS:
                    self.detour_target = None
                else:
                    return self.detour_target, distance
            delta = target - self.position
            dx, dy, dz = _ned_to_body_flu(
                delta[0], delta[1], delta[2], self.current_yaw)
            reactive_points = flight_level_lidar_points(self.latest_points)
            dx, dy, dz, scale = clamp_to_obstacle(
                dx, dy, dz, reactive_points)
            dz, vertical_scale = clamp_vertical(dz, -self.position[2])
            scale = min(scale, vertical_scale)
            self.update_lidar_clamp_event(scale < 0.999, scale)
            if scale < 0.999:
                self.lidar_clamped_count += 1
            self.stuck_ticks = (
                self.stuck_ticks + 1
                if is_stuck(scale, distance, self.prev_distance) else 0)
            self.prev_distance = distance
            if self.stuck_ticks > stuck_ticks_limit:
                self.detour_target = self.find_sidestep()
                self.stuck_ticks = 0
                if self.detour_target is None:
                    self.lidar_hold_count += 1
                    self.lidar_hold_events.append(self.event_time())
                    return self.position.copy(), distance
                self.lidar_sidestep_count += 1
                event = self.event_time()
                event["detour_target_ned_m"] = [
                    float(value) for value in self.detour_target]
                self.lidar_sidestep_events.append(event)
                return self.detour_target, distance
            n, e, d = _body_flu_to_ned(dx, dy, dz, self.current_yaw)
            return self.position + np.array([n, e, d]), distance

        def enter_capture_hold(self):
            self.holding_for_capture = True
            self.hold_started = self.now_seconds()
            self.vehicle_good_samples = 0
            self.gimbal_settled = False
            self.gimbal_target_sent = False
            self.settled_since = None
            self.capture_ready_since = None
            self.get_logger().info(
                "View %d/%d reached; settling vehicle/gimbal/image."
                % (self.shot_index + 1, self.capture_count))

        def capture_if_settled(self, target, yaw):
            distance = float(np.linalg.norm(self.position - target))
            yaw_error = angle_error(self.current_yaw, yaw)
            linear_speed = (
                float(np.linalg.norm(self.velocity))
                if self.velocity is not None else math.inf)
            angular_speed = (
                float(np.linalg.norm(self.angular_velocity))
                if self.angular_velocity is not None else math.inf)
            if (distance <= self.vehicle_settle_distance
                    and yaw_error <= self.vehicle_settle_yaw
                    and linear_speed <= self.vehicle_settle_speed
                    and angular_speed <= self.vehicle_settle_angular_speed):
                self.vehicle_good_samples += 1
            else:
                self.vehicle_good_samples = 0
                self.settled_since = None
                self.capture_ready_since = None
            vehicle_settled = (
                self.vehicle_good_samples >= settle_samples_required)
            if vehicle_settled and not self.gimbal_target_sent:
                self.gimbal_settled = False
                self.publish_gimbal_target()
                self.gimbal_target_sent = True
            basic_ready = (
                vehicle_settled and self.gimbal_target_sent
                and self.gimbal_settled
                and self.actual_gimbal_yaw is not None
                and self.actual_gimbal_pitch is not None)
            if basic_ready and self.capture_ready_since is None:
                self.capture_ready_since = self.now_seconds()
            if not basic_ready:
                self.capture_ready_since = None
            delay_complete = (
                self.capture_ready_since is not None
                and self.now_seconds() - self.capture_ready_since
                >= self.post_settle_delay)
            image_fresh = (
                self.image_time is not None
                and self.capture_ready_since is not None
                and self.image_time >= self.capture_ready_since)
            ready = basic_ready and delay_complete and image_fresh
            if ready:
                self.capture_image()
                return
            if self.now_seconds() - self.hold_started > self.capture_timeout:
                detail = (
                    "settle timeout: vehicle=%s, gimbal=%s, delay=%s, image=%s"
                    % (vehicle_settled, self.gimbal_settled,
                       delay_complete, image_fresh))
                self.record_shot("CAPTURE_FAILED", None, detail)
                self.advance_after_capture()

        def capture_image(self):
            source = self.preflights[self.shot_index].get(
                "auto_viewpoint", {})
            side = source.get(
                "selected_side", "view_%d" % (self.shot_index + 1))
            target_id = int(source.get("source_target_id", 0))
            name = datetime.now(timezone.utc).strftime(
                "target_%03d_%s_%%Y%%m%%dT%%H%%M%%S_%%fZ.jpg"
                % (target_id, side))
            image_path = os.path.join(self.out_dir, name)
            try:
                save_image(self.latest_image, image_path)
            except ValueError as error:
                self.record_shot("CAPTURE_FAILED", None, str(error))
                self.advance_after_capture()
                return
            self.record_shot("IMAGE_CAPTURED", image_path)
            self.get_logger().info(
                "Captured view %d/%d: %s" % (
                    self.shot_index + 1, self.capture_count, image_path))
            self.advance_after_capture()

        def record_shot(self, status, image_path, detail=""):
            preflight = self.preflights[self.shot_index]
            actual_pose = px4_pose_in_saved_map(
                self.position, self.current_yaw, self.saved_from_live)
            metrics = {}
            if (self.actual_gimbal_yaw is not None
                    and self.actual_gimbal_pitch is not None):
                metrics = actual_view_metrics(
                    actual_pose, self.actual_gimbal_yaw,
                    self.actual_gimbal_pitch,
                    self.targets_saved[self.shot_index],
                    self.requests[self.shot_index])
            planned = deepcopy(preflight.get("auto_viewpoint", {}))
            selected = preflight.get("selected", {})
            planned_route = self.saved_routes[self.shot_index]
            route_cross_track = point_to_polyline_distance(
                actual_pose[:3], planned_route)
            linear_speed = (
                float(np.linalg.norm(self.velocity))
                if self.velocity is not None else None)
            angular_speed = (
                float(np.linalg.norm(self.angular_velocity))
                if self.angular_velocity is not None else None)
            shot = {
                "schema": SHOT_RESULT_SCHEMA,
                "status": status,
                "detail": detail,
                "target_id": planned.get("source_target_id"),
                "crop_id": planned.get("source_crop_id"),
                "side": planned.get("selected_side"),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "capture_time_from_start_s": self.mission_elapsed()[0],
                "capture_time_from_start_wall_s": (
                    self.now_seconds() - self.mission_started),
                "planned": planned,
                "viewpoint_clearance_m": selected.get(
                    "mapped_obstacle_clearance_m"),
                "actual_vehicle_pose_map": list(actual_pose),
                "actual_vehicle_roll_deg": math.degrees(self.current_roll),
                "actual_vehicle_pitch_deg": math.degrees(self.current_pitch),
                "actual_vehicle_yaw_deg": math.degrees(self.current_yaw),
                "actual_linear_velocity_ned_mps": (
                    self.velocity.tolist() if self.velocity is not None
                    else None),
                "actual_angular_velocity_body_rad_s": (
                    self.angular_velocity.tolist()
                    if self.angular_velocity is not None else None),
                "actual_linear_speed_mps": linear_speed,
                "actual_angular_speed_rad_s": angular_speed,
                "vehicle_settle_speed_limit_mps": self.vehicle_settle_speed,
                "vehicle_settle_angular_speed_limit_rad_s": (
                    self.vehicle_settle_angular_speed),
                "post_settle_delay_s": self.post_settle_delay,
                "route_cross_track_error_m": route_cross_track,
                "actual_gimbal_yaw_deg": (
                    math.degrees(self.actual_gimbal_yaw)
                    if self.actual_gimbal_yaw is not None else None),
                "actual_gimbal_pitch_deg": (
                    math.degrees(self.actual_gimbal_pitch)
                    if self.actual_gimbal_pitch is not None else None),
                "gimbal_settled": bool(self.gimbal_settled),
                "image_path": image_path,
                "image_width_px": (
                    int(self.latest_image.width)
                    if image_path and self.latest_image is not None else None),
                "image_height_px": (
                    int(self.latest_image.height)
                    if image_path and self.latest_image is not None else None),
                "image_encoding": (
                    str(self.latest_image.encoding)
                    if image_path and self.latest_image is not None else None),
                "image_file_bytes": (
                    Path(image_path).stat().st_size if image_path else None),
                "target_visibility_verified_in_image": None,
            }
            shot.update(metrics)
            if image_path and self.latest_image is not None:
                shot.update(selected_highlight_metrics(
                    image_to_array(self.latest_image)))
                shot["highlight_measurement_scope"] = (
                    "target_specific" if self.target_count == 1 else
                    "all_selected_cyan_plants_in_frame")
                shot["target_visibility_verified_in_image"] = (
                    shot["selected_plant_highlight_visible"]
                    if self.target_count == 1 else None)
            self.result["shots"].append(shot)
            self.write_result()

        def advance_after_capture(self):
            self.holding_for_capture = False
            if self.shot_index + 1 < self.capture_count:
                self.shot_index += 1
                self.start_phase(self.shot_index)
                self.get_logger().info(
                    "View %d complete; remaining airborne for view %d."
                    % (self.shot_index, self.shot_index + 1))
            else:
                self.start_phase(self.return_phase_index)
                self.get_logger().info(
                    "Final view complete; returning once to the launch pad.")

        def start_phase(self, phase_index):
            self.update_lidar_clamp_event(False)
            self.phase_index = int(phase_index)
            self.waypoints = self.phase_routes_px4[self.phase_index]
            self.index = 1
            self.commanded_yaw = float("nan")
            self.stuck_ticks = 0
            self.prev_distance = float("inf")
            self.detour_target = None

        def begin_safe_return(self, reason):
            """Abort the inspection by retracing reviewed airborne corridors."""
            if self.returning_after_abort or self.landing:
                return
            if (self.position is None or self.phase_routes_px4 is None
                    or self.waypoints is None):
                self.request_land(
                    "%s; no retrace route was available" % reason)
                return
            abort_route = build_abort_retrace_route(
                self.phase_routes_px4,
                self.phase_index,
                self.index,
                self.position,
                self.current_yaw,
            )
            self.returning_after_abort = True
            self.abort_reason = str(reason)
            self.safe_return_started = self.now_seconds()
            self.holding_for_capture = False
            self.phase_index = self.return_phase_index
            self.waypoints = abort_route
            self.index = 1
            self.commanded_yaw = self.current_yaw
            self.stuck_ticks = 0
            self.prev_distance = float("inf")
            self.detour_target = None
            self.result["status"] = "ABORT_RETURNING_TO_PAD"
            self.result["detail"] = self.abort_reason
            execution = self.result["execution"]
            execution["safe_abort_return_requested"] = True
            execution["safe_abort_reason"] = self.abort_reason
            execution["safe_abort_route_waypoints"] = int(len(abort_route))
            self.write_result()
            self.get_logger().error(
                "SAFE_ABORT: %s; retracing %d reviewed waypoints to the pad "
                "before landing." % (self.abort_reason, len(abort_route)))

        def finish_return(self):
            self.update_lidar_clamp_event(False)
            self.landing = True
            self.landing_started = self.now_seconds()
            self.last_land_command = None
            self.landing_low_since = None
            self.last_disarm_command = None
            self.result["execution"]["landing_request_count"] = 1
            self.result["execution"]["landing_outcome"] = (
                "ABORT_RETURNED_TO_PAD_LANDING_REQUESTED"
                if self.returning_after_abort else "REQUESTED")
            self.send_land_command()
            self.write_result()
            self.get_logger().info(
                "Airborne pad waypoint reached; one PX4 landing requested.")

        def send_land_command(self):
            self.publish_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.last_land_command = self.now_seconds()

        def guarded_sim_disarm(self):
            """Stop Gazebo ground-contact bouncing, never an airborne disarm."""
            if (not self.sim_ground_disarm or self.position is None
                    or self.saved_from_live is None or not self.armed):
                self.landing_low_since = None
                return
            pose = px4_pose_in_saved_map(
                self.position, self.current_yaw, self.saved_from_live)
            launch = self.saved_routes[0][0]
            horizontal = math.hypot(pose[0] - launch[0], pose[1] - launch[1])
            height = abs(pose[2] - launch[2])
            speed = (
                float(np.linalg.norm(self.velocity))
                if self.velocity is not None else math.inf)
            safely_low = horizontal <= 0.60 and height <= 0.35 and speed <= 1.0
            if not safely_low:
                self.landing_low_since = None
                return
            now = self.now_seconds()
            if self.landing_low_since is None:
                self.landing_low_since = now
                return
            low_duration = now - self.landing_low_since
            if (low_duration >= 2.0
                    and (self.last_disarm_command is None
                         or now - self.last_disarm_command >= 1.0)):
                parameters = {"param1": 0.0}
                outcome = "NORMAL_DISARM_REQUESTED_NEAR_GROUND"
                if low_duration >= 6.0:
                    parameters["param2"] = 21196.0
                    outcome = "GUARDED_FORCE_DISARM_REQUESTED_NEAR_GROUND"
                self.publish_command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                    **parameters)
                self.last_disarm_command = now
                self.result["execution"]["landing_outcome"] = outcome

        def timer_callback(self):
            if self.completed:
                return
            if self.landing:
                elapsed = self.now_seconds() - self.landing_started
                if (self.last_land_command is None
                        or self.now_seconds() - self.last_land_command >= 2.0):
                    self.send_land_command()
                self.guarded_sim_disarm()
                if elapsed > self.landing_timeout:
                    # Preserve evidence but keep monitoring: exiting while the
                    # vehicle is armed would recreate the unsafe old behaviour.
                    if not self.result.get("landing_timeout_recorded"):
                        self.result["landing_timeout_recorded"] = True
                        self.result["status"] = "LANDING_DELAYED"
                        self.result["detail"] = (
                            "PX4 remained armed %.1f s after landing request; "
                            "landing commands and guarded ground disarm continue"
                            % self.landing_timeout)
                        self.write_result()
                return
            if self.waypoints is None:
                self.try_accept_live_state()
                return
            if (self.runtime_clearance_abort_samples > 0
                    and not self.returning_after_abort
                    and self.static_violation_streak
                    >= self.runtime_clearance_abort_samples):
                self.begin_safe_return(
                    "vehicle entered an inflated static-obstacle boundary for "
                    "%d consecutive odometry samples"
                    % self.static_violation_streak)
            mission_elapsed, timeout_clock = self.mission_elapsed()
            if (not self.returning_after_abort
                    and mission_elapsed > self.mission_timeout):
                self.begin_safe_return(
                    "mission timeout after %.1f simulated seconds (%s)"
                    % (mission_elapsed, timeout_clock))
                return
            if (self.returning_after_abort
                    and self.safe_return_started is not None
                    and self.now_seconds() - self.safe_return_started
                    > self.safe_return_timeout):
                # Continuing to publish a setpoint while LiDAR is repeatedly deflecting the aircraft is not a safe recovery action.
                self.result["execution"]["safe_return_timeout_sec"] = float(
                    self.safe_return_timeout)
                self.result["execution"]["safe_return_timeout_action"] = (
                    "OFFBOARD_LOSS_FAILSAFE_RTL")
                self.finalise(
                    "MISSION_ABORTED_RETURN_TIMEOUT",
                    "%s; reviewed safe-return route did not complete within "
                    "%.1f wall seconds, so Offboard publishing was stopped "
                    "and PX4 failsafe was allowed to return/land"
                    % (self.abort_reason, self.safe_return_timeout))
                return
            if (self.use_lidar_safety
                    and (self.lidar_time is None
                         or self.now_seconds() - self.lidar_time > 1.0)):
                self.lidar_stale_count += 1
                self.get_logger().error(
                    "Live LiDAR is stale; holding and withholding progress.",
                    throttle_duration_sec=2.0)
                self.publish_offboard_mode()
                self.publish_setpoint(self.position, self.current_yaw)
                self.engage_watchdog()
                self.tick += 1
                return
            if self.ground_takeoff and not self.takeoff_complete:
                ascent = self.takeoff_origin[2] - self.position[2]
                if ascent >= self.ground_launch_altitude - 0.10:
                    self.takeoff_complete = True
                    self.index = 1
                    self.get_logger().info(
                        "GROUND_LAUNCH_COMPLETE: following reviewed route.")
                else:
                    upward = self.ground_launch_speed
                    if self.use_lidar_safety:
                        _, _, upward, _ = clamp_to_obstacle(
                            0.0, 0.0, upward, self.latest_points)
                        upward, _ = clamp_vertical(
                            upward, -self.position[2])
                    self.publish_offboard_velocity_mode()
                    self.publish_velocity_setpoint((0.0, 0.0, -upward))
                    self.engage_watchdog()
                    self.tick += 1
                    return
            self.publish_offboard_mode()
            waypoint = self.waypoints[self.index]
            target = waypoint[:3]
            yaw = waypoint[3]
            self.commanded_yaw = ramped_yaw(
                self.commanded_yaw, yaw, max_rate=MAX_YAW_RATE)
            setpoint, distance = self.safe_setpoint(target)
            self.publish_setpoint(setpoint, self.commanded_yaw)
            if self.holding_for_capture:
                self.capture_if_settled(target, yaw)
            elif distance < REACH_RADIUS:
                if self.index < len(self.waypoints) - 1:
                    self.index += 1
                elif self.phase_index < self.capture_count:
                    self.enter_capture_hold()
                else:
                    self.finish_return()
            self.engage_watchdog()
            self.tick += 1

        def update_execution_metrics(self):
            execution = self.result["execution"]
            mission_started = getattr(self, "mission_started", None)
            trajectory = getattr(self, "trajectory", [])
            actual_min_clearance = getattr(
                self, "actual_min_clearance", math.inf)
            if mission_started is not None:
                wall_duration = float(self.now_seconds() - mission_started)
                sim_duration, clock = self.mission_elapsed()
                execution["mission_duration_s"] = float(sim_duration)
                execution["mission_duration_wall_s"] = wall_duration
                execution["mission_duration_sim_s"] = float(sim_duration)
                execution["mission_elapsed_clock"] = clock
            execution["trajectory_sample_count"] = len(trajectory)
            if len(trajectory) >= 2:
                execution["actual_path_length_m"] = float(sum(
                    np.linalg.norm(end - start)
                    for start, end in zip(trajectory, trajectory[1:])))
            if math.isfinite(actual_min_clearance):
                execution["minimum_clearance_to_inflated_obstacles_m"] = float(
                    actual_min_clearance)
            if math.isfinite(getattr(
                    self, "actual_min_canopy_clearance", math.inf)):
                execution["minimum_clearance_to_inflated_canopy_m"] = float(
                    self.actual_min_canopy_clearance)
            if math.isfinite(getattr(
                    self, "actual_min_structure_clearance", math.inf)):
                execution["minimum_clearance_to_inflated_structure_m"] = float(
                    self.actual_min_structure_clearance)
            execution["inflated_obstacle_violation_samples"] = int(
                getattr(self, "violation_samples", 0))
            execution["inflated_obstacle_violation_duration_s"] = float(
                getattr(self, "violation_duration", 0.0))
            cross_track = getattr(self, "cross_track_errors", [])
            if cross_track:
                execution["route_cross_track_error_mean_m"] = float(
                    np.mean(cross_track))
                execution["route_cross_track_error_max_m"] = float(
                    np.max(cross_track))
                execution["cruise_route_cross_track_error_mean_m"] = float(
                    np.mean(cross_track))
                execution["cruise_route_cross_track_error_max_m"] = float(
                    np.max(cross_track))
            execution["route_cross_track_error_by_phase_m"] = {
                name: sample_statistics(values)
                for name, values in getattr(
                    self, "phase_cross_track_errors", {}).items()
            }
            execution["ground_launch_horizontal_error_m"] = sample_statistics(
                getattr(self, "ground_launch_horizontal_errors", []))
            execution["landing_horizontal_error_m"] = sample_statistics(
                getattr(self, "landing_horizontal_errors", []))
            clamp_events = deepcopy(getattr(
                self, "lidar_clamp_events", []))
            active_event = getattr(self, "current_lidar_clamp_event", None)
            if active_event is not None:
                snapshot = deepcopy(active_event)
                snapshot["end"] = self.event_time()
                start_time = snapshot["start"].get("mission_time_s")
                end_time = snapshot["end"].get("mission_time_s")
                if start_time is not None and end_time is not None:
                    snapshot["duration_s"] = max(
                        0.0, end_time - start_time)
                clamp_events.append(snapshot)
            execution["lidar_safety"] = {
                "enabled": bool(getattr(self, "use_lidar_safety", False)),
                "status": (
                    "ACTIVE" if getattr(self, "use_lidar_safety", False)
                    else "DISABLED_STATIC_MAP_ONLY"),
                "clamped_setpoint_count": int(getattr(
                    self, "lidar_clamped_count", 0)),
                "clamp_event_count": len(clamp_events),
                "clamp_events": clamp_events,
                "sidestep_count": int(getattr(
                    self, "lidar_sidestep_count", 0)),
                "sidestep_events": deepcopy(getattr(
                    self, "lidar_sidestep_events", [])),
                "hold_count": int(getattr(self, "lidar_hold_count", 0)),
                "hold_events": deepcopy(getattr(
                    self, "lidar_hold_events", [])),
                "stale_data_hold_count": int(getattr(
                    self, "lidar_stale_count", 0)),
            }
            execution["images_captured"] = sum(
                shot.get("status") == "IMAGE_CAPTURED"
                for shot in self.result["shots"])
            execution["shots_attempted"] = len(self.result["shots"])

        def write_result(self):
            self.update_execution_metrics()
            path = Path(self.result_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self.result, indent=2) + "\n", encoding="utf-8")
            write_mission_csv(self.result, self.result_csv)

        def finalise(self, status, detail):
            if self.completed:
                return
            self.result["status"] = status
            self.result["detail"] = detail
            self.result["timestamp_finished_utc"] = (
                datetime.now(timezone.utc).isoformat())
            self.write_result()
            self.completed = True
            self.timer.cancel()

        def request_land(self, reason):
            if not self.start_flight or not hasattr(self, "command_pub"):
                return
            if not self.landing:
                self.landing = True
                self.landing_started = self.now_seconds()
                self.last_land_command = None
                self.landing_low_since = None
                self.last_disarm_command = None
                self.result["execution"]["landing_request_count"] = 1
                self.result["execution"]["landing_outcome"] = "ABORT_REQUESTED"
            self.send_land_command()
            self.result["status"] = "ABORTED_LANDING_REQUESTED"
            self.result["detail"] = reason
            self.write_result()
            self.get_logger().warn("Landing requested: %s" % reason)

    rclpy.init(args=args)
    node = TwoViewpointMission()
    try:
        while rclpy.ok() and not node.completed:
            rclpy.spin_once(node, timeout_sec=0.25)
    except KeyboardInterrupt:
        node.request_land("Ctrl-C")
        for _ in range(20):
            rclpy.spin_once(node, timeout_sec=0.1)
    except ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
