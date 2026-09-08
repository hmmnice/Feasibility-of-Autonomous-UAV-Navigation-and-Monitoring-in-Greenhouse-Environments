"""Automatic camera-pose selection for a mapped greenhouse crop."""

from dataclasses import dataclass
import math
import time

import numpy as np

from greenhouse_inspection.free_space import TRANSIT_Z, segment_blocked
from greenhouse_inspection.side_shot_inspection import SideShotPlan
from greenhouse_inspection.viewpoint_geometry import (
    CameraConfig,
    CandidateSampling,
    generate_candidates,
    wrap_pi,
)
from greenhouse_inspection.viewpoint_inspection import (
    InspectionPreflight,
    plan_inspection_preflight,
)
from greenhouse_inspection.viewpoint_selection import (
    minimum_obstacle_clearance,
    PerspectiveRequest,
    SelectionConfig,
)
from greenhouse_inspection.viewpoint_validation import (
    ValidationConfig,
    associate_target_box,
    collision_boxes_from_geometry,
    flight_bounds_from_geometry,
    optical_boxes_from_geometry,
    validate_candidate,
)

from .static_greenhouse import CANOPY_TOP, TARGET_FACE_OFFSET


CAMERA_HFOV_DEG = 55.3
CAMERA_VFOV_DEG = 42.9
ROI_HALF_WIDTH_X = 0.55
ROI_HALF_DEPTH_Y = 0.30
# Model the full crop height so lower leaves are included in the projection.
ROI_BOTTOM_Z = 0.0
ROI_TOP_Z = CANOPY_TOP
FOV_MARGIN = 0.92
DESIRED_IMAGE_FRACTION = 0.36
SCORE_WEIGHTS = {
    "overflow": 100.0,
    "centre_error": 1.5,
    "target_size_error": 1.0,
    "route_length_m": 0.01,
    "azimuth_offset_deg": 0.01,
    "occluded_fraction": 3.0,
    "edge_margin_shortfall": 2.0,
    "clearance_shortfall_m": 1.5,
}
DESIRED_NORMALISED_EDGE_MARGIN = 0.08
DESIRED_VIEWPOINT_CLEARANCE_M = 0.35

SIDE_OPTIONS = (
    ("positive_y", 90.0, 1.0),
    ("negative_y", 270.0, -1.0),
)
AZIMUTH_OFFSETS_DEG = (-15.0, 0.0, 15.0)
AIM_HEIGHTS_M = (1.8, 2.3, 2.8, 3.3)
# These stand-offs keep the full crop in view without approaching the canopy.
MIN_EXECUTION_HORIZONTAL_STANDOFF_M = 2.2
HORIZONTAL_STANDOFFS_M = (2.2, 2.6)
CAMERA_HEIGHTS_M = (2.2, 3.2, 4.0, 4.8, 5.4)


@dataclass(frozen=True)
class ImageProjection:
    """Predicted angular plant bounds in the candidate camera image."""

    u_min_deg: float
    u_max_deg: float
    v_min_deg: float
    v_max_deg: float
    horizontal_fill: float
    vertical_fill: float
    image_fraction: float
    centre_error: float
    minimum_edge_margin: float
    overflow: float
    fully_inside_margin: bool


@dataclass(frozen=True)
class AutoViewCandidate:
    """One feasible flight/viewpoint candidate and its image-space score."""

    side: str
    azimuth_deg: float
    azimuth_offset_deg: float
    aim_height_m: float
    horizontal_standoff_m: float
    camera_height_m: float
    pitch_deg: float
    distance_m: float
    target: tuple
    vehicle_yaw_rad: float
    preflight: InspectionPreflight
    projection: ImageProjection
    predicted_visible_point_fraction: float
    viewpoint_clearance_m: float
    score_components: dict
    score: float


@dataclass(frozen=True)
class AutoViewResult:
    """Best automatic view plus candidate statistics for reporting."""

    selected: AutoViewCandidate
    attempted_candidates: int
    physically_feasible_candidates: int
    fully_framed_candidates: int
    routed_candidates: int
    route_feasible_candidates: int
    candidate_evaluation_time_ms: float
    route_search_time_ms: float
    planner_time_ms: float


@dataclass(frozen=True)
class _GeometricCandidate:
    """Fast first-stage result before expensive route generation."""

    side: str
    azimuth_deg: float
    azimuth_offset_deg: float
    aim_height_m: float
    horizontal_standoff_m: float
    camera_height_m: float
    pitch_deg: float
    distance_m: float
    target: tuple
    vehicle_yaw_rad: float
    projection: ImageProjection
    predicted_visible_point_fraction: float
    viewpoint_clearance_m: float
    score_components_without_route: dict
    score_without_route: float


def plant_roi_corners(plant_x, row_y):
    """Return the eight corners of the finite plant inspection region."""
    return tuple(
        (float(x_value), float(y_value), float(z_value))
        for x_value in (
            plant_x - ROI_HALF_WIDTH_X,
            plant_x + ROI_HALF_WIDTH_X,
        )
        for y_value in (
            row_y - ROI_HALF_DEPTH_Y,
            row_y + ROI_HALF_DEPTH_Y,
        )
        for z_value in (ROI_BOTTOM_Z, ROI_TOP_Z)
    )


def plant_roi_sample_points(plant_x, face_y):
    """Return a 3x3 inspection grid across the selected plant face."""
    return tuple(
        (float(x_value), float(face_y), float(z_value))
        for z_value in np.linspace(ROI_BOTTOM_Z, ROI_TOP_Z, 3)
        for x_value in np.linspace(
            plant_x - ROI_HALF_WIDTH_X,
            plant_x + ROI_HALF_WIDTH_X,
            3,
        )
    )


def visible_point_fraction(camera_position, points, optical_boxes, target_box):
    """Estimate target-region visibility using multiple geometric rays."""
    if isinstance(target_box, (int, np.integer)):
        boxes = tuple(
            box for index, box in enumerate(optical_boxes)
            if index != int(target_box))
    else:
        boxes = tuple(
            box for box in optical_boxes
            if target_box is None or tuple(box) != tuple(target_box))
    visible = sum(
        segment_blocked(camera_position, point, boxes) is None
        for point in points
    )
    return float(visible / len(points)) if points else 0.0


def project_plant_region(camera_position, aim_point, roi_corners):
    """Project the plant corners into angular pinhole-camera coordinates."""
    camera = np.asarray(camera_position, dtype=float)
    aim = np.asarray(aim_point, dtype=float)
    forward = aim - camera
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1e-9:
        raise ValueError("camera position and aim point must differ")
    forward /= forward_norm
    world_up = np.asarray((0.0, 0.0, 1.0))
    right = np.cross(forward, world_up)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-9:
        raise ValueError("vertical camera optical axes are unsupported")
    right /= right_norm
    up = np.cross(right, forward)

    horizontal_angles = []
    vertical_angles = []
    for corner in roi_corners:
        relative = np.asarray(corner, dtype=float) - camera
        depth = float(relative @ forward)
        if depth <= 1e-9:
            raise ValueError("plant ROI extends behind the candidate camera")
        horizontal_angles.append(math.degrees(math.atan2(
            float(relative @ right), depth)))
        vertical_angles.append(math.degrees(math.atan2(
            float(relative @ up), depth)))

    u_min, u_max = min(horizontal_angles), max(horizontal_angles)
    v_min, v_max = min(vertical_angles), max(vertical_angles)
    horizontal_fill = (u_max - u_min) / CAMERA_HFOV_DEG
    vertical_fill = (v_max - v_min) / CAMERA_VFOV_DEG
    image_fraction = horizontal_fill * vertical_fill
    u_centre = (u_min + u_max) / 2.0
    v_centre = (v_min + v_max) / 2.0
    centre_error = math.hypot(
        u_centre / (CAMERA_HFOV_DEG / 2.0),
        v_centre / (CAMERA_VFOV_DEG / 2.0),
    )
    horizontal_limit = FOV_MARGIN * CAMERA_HFOV_DEG / 2.0
    vertical_limit = FOV_MARGIN * CAMERA_VFOV_DEG / 2.0
    horizontal_overflow = max(
        0.0,
        max(abs(u_min), abs(u_max)) - horizontal_limit,
    ) / (CAMERA_HFOV_DEG / 2.0)
    vertical_overflow = max(
        0.0,
        max(abs(v_min), abs(v_max)) - vertical_limit,
    ) / (CAMERA_VFOV_DEG / 2.0)
    overflow = horizontal_overflow + vertical_overflow
    minimum_edge_margin = min(
        horizontal_limit - max(abs(u_min), abs(u_max)),
        vertical_limit - max(abs(v_min), abs(v_max)),
    ) / max(CAMERA_HFOV_DEG / 2.0, CAMERA_VFOV_DEG / 2.0)
    return ImageProjection(
        u_min, u_max, v_min, v_max,
        horizontal_fill, vertical_fill, image_fraction,
        centre_error, minimum_edge_margin, overflow, overflow <= 1e-12,
    )


def candidate_score_components(
        projection, route_length, azimuth_offset,
        predicted_visible_fraction=1.0,
        viewpoint_clearance_m=DESIRED_VIEWPOINT_CLEARANCE_M):
    """Return transparent weighted terms; physical validity is separate."""
    terms = {
        "overflow": SCORE_WEIGHTS["overflow"] * projection.overflow,
        "centre_error": (
            SCORE_WEIGHTS["centre_error"] * projection.centre_error),
        "target_size_error": SCORE_WEIGHTS["target_size_error"] * abs(
            projection.image_fraction - DESIRED_IMAGE_FRACTION),
        "route_length": SCORE_WEIGHTS["route_length_m"] * float(
            route_length),
        "azimuth_offset": SCORE_WEIGHTS["azimuth_offset_deg"] * abs(
            float(azimuth_offset)),
        "occluded_fraction": SCORE_WEIGHTS["occluded_fraction"] * (
            1.0 - float(predicted_visible_fraction)),
        "edge_margin_shortfall": SCORE_WEIGHTS[
            "edge_margin_shortfall"] * max(
                0.0,
                DESIRED_NORMALISED_EDGE_MARGIN
                - projection.minimum_edge_margin),
        "clearance_shortfall": SCORE_WEIGHTS[
            "clearance_shortfall_m"] * max(
                0.0,
                DESIRED_VIEWPOINT_CLEARANCE_M
                - float(viewpoint_clearance_m)),
    }
    return terms


def candidate_score(
        projection, route_length, azimuth_offset,
        predicted_visible_fraction=1.0,
        viewpoint_clearance_m=DESIRED_VIEWPOINT_CLEARANCE_M):
    """Lower is better; physical validity is handled before this score."""
    return float(sum(candidate_score_components(
        projection, route_length, azimuth_offset,
        predicted_visible_fraction, viewpoint_clearance_m).values()))


def plan_best_view(
        map_geometry, start_pose, plant_x, row_y,
        transit_z=TRANSIT_Z, side_options=SIDE_OPTIONS,
        azimuth_offsets_deg=AZIMUTH_OFFSETS_DEG):
    """Search the requested plant faces for the best safe, framed view."""
    planning_started = time.perf_counter()
    camera_config = CameraConfig(
        minimum_standoff=1.0,
        maximum_standoff=4.0,
        desired_standoff=2.5,
    )
    selection_config = SelectionConfig(
        maximum_direction_error=math.radians(0.5),
        maximum_distance_error=0.02,
    )
    collision_boxes = collision_boxes_from_geometry(map_geometry)
    optical_boxes = optical_boxes_from_geometry(map_geometry)
    validation_config = ValidationConfig(
        bounds=flight_bounds_from_geometry(map_geometry),
        camera=camera_config,
    )
    roi_corners = plant_roi_corners(plant_x, row_y)
    attempted = 0
    geometric_candidates = []
    for side, base_azimuth, face_sign in side_options:
        face_y = row_y + face_sign * TARGET_FACE_OFFSET
        roi_sample_points = plant_roi_sample_points(plant_x, face_y)
        for azimuth_offset in azimuth_offsets_deg:
            azimuth = (base_azimuth + azimuth_offset) % 360.0
            vehicle_yaw = wrap_pi(math.radians(azimuth) + math.pi)
            for aim_height in AIM_HEIGHTS_M:
                target = (float(plant_x), float(face_y), float(aim_height))
                target_box = associate_target_box(
                    target,
                    optical_boxes[:len(map_geometry.rows)],
                )
                for horizontal_standoff in HORIZONTAL_STANDOFFS_M:
                    for camera_height in CAMERA_HEIGHTS_M:
                        attempted += 1
                        vertical_offset = camera_height - aim_height
                        if vertical_offset < -0.1:
                            continue
                        distance = math.hypot(
                            horizontal_standoff, vertical_offset)
                        if not 1.0 <= distance <= 4.0:
                            continue
                        pitch = math.degrees(math.atan2(
                            vertical_offset, horizontal_standoff))
                        request = PerspectiveRequest(
                            math.radians(azimuth),
                            math.radians(pitch),
                            distance,
                        )
                        sampling = CandidateSampling(
                            azimuths_deg=(azimuth,),
                            elevations_deg=(pitch,),
                            distances=(distance,),
                        )
                        candidate = generate_candidates(
                            target,
                            camera=camera_config,
                            sampling=sampling,
                            vehicle_yaw_map=vehicle_yaw,
                        )[0]
                        validation = validate_candidate(
                            candidate,
                            validation_config,
                            collision_boxes,
                            optical_boxes,
                            target_box,
                        )
                        if not validation.valid:
                            continue
                        try:
                            projection = project_plant_region(
                                candidate.camera_position,
                                target,
                                roi_corners,
                            )
                        except ValueError:
                            continue
                        predicted_visibility = visible_point_fraction(
                            candidate.camera_position,
                            roi_sample_points,
                            optical_boxes,
                            target_box,
                        )
                        viewpoint_clearance = minimum_obstacle_clearance(
                            validation.vehicle_position,
                            collision_boxes,
                        )
                        score_components = candidate_score_components(
                            projection,
                            0.0,
                            azimuth_offset,
                            predicted_visibility,
                            viewpoint_clearance,
                        )
                        score = candidate_score(
                            projection,
                            0.0,
                            azimuth_offset,
                            predicted_visibility,
                            viewpoint_clearance,
                        )
                        geometric_candidates.append(_GeometricCandidate(
                            side=side,
                            azimuth_deg=azimuth,
                            azimuth_offset_deg=azimuth_offset,
                            aim_height_m=aim_height,
                            horizontal_standoff_m=horizontal_standoff,
                            camera_height_m=camera_height,
                            pitch_deg=pitch,
                            distance_m=distance,
                            target=target,
                            vehicle_yaw_rad=vehicle_yaw,
                            projection=projection,
                            predicted_visible_point_fraction=(
                                predicted_visibility),
                            viewpoint_clearance_m=viewpoint_clearance,
                            score_components_without_route=score_components,
                            score_without_route=score,
                        ))
    candidate_evaluation_finished = time.perf_counter()
    if not geometric_candidates:
        raise ValueError(
            "no physically feasible automatic viewpoint was found")
    fully_framed = [
        candidate for candidate in geometric_candidates
        if candidate.projection.fully_inside_margin
    ]
    search_groups = [fully_framed]
    if not fully_framed:
        search_groups = [geometric_candidates]
    else:
        search_groups.append([
            candidate for candidate in geometric_candidates
            if not candidate.projection.fully_inside_margin
        ])

    selected = None
    routed_candidates = 0
    route_feasible_candidates = 0
    route_search_started = time.perf_counter()
    for group in search_groups:
        for candidate in sorted(
                group, key=lambda item: item.score_without_route):
            # Stop when the remaining candidates cannot improve the best score.
            if (selected is not None
                    and candidate.score_without_route >= selected.score):
                break
            request = PerspectiveRequest(
                math.radians(candidate.azimuth_deg),
                math.radians(candidate.pitch_deg),
                candidate.distance_m,
            )
            routed_candidates += 1
            preflight = plan_inspection_preflight(
                map_geometry,
                start_pose,
                candidate.target,
                request,
                camera=camera_config,
                sampling=CandidateSampling(
                    azimuths_deg=(candidate.azimuth_deg,),
                    elevations_deg=(candidate.pitch_deg,),
                    distances=(candidate.distance_m,),
                ),
                selection_config=selection_config,
                vehicle_yaw_map=candidate.vehicle_yaw_rad,
                transit_z=float(transit_z),
            )
            if not preflight.ready:
                continue
            route_feasible_candidates += 1
            final_score = candidate_score(
                candidate.projection,
                preflight.route.length,
                candidate.azimuth_offset_deg,
                candidate.predicted_visible_point_fraction,
                candidate.viewpoint_clearance_m,
            )
            score_components = candidate_score_components(
                candidate.projection,
                preflight.route.length,
                candidate.azimuth_offset_deg,
                candidate.predicted_visible_point_fraction,
                candidate.viewpoint_clearance_m,
            )
            complete = AutoViewCandidate(
                side=candidate.side,
                azimuth_deg=candidate.azimuth_deg,
                azimuth_offset_deg=candidate.azimuth_offset_deg,
                aim_height_m=candidate.aim_height_m,
                horizontal_standoff_m=candidate.horizontal_standoff_m,
                camera_height_m=candidate.camera_height_m,
                pitch_deg=candidate.pitch_deg,
                distance_m=candidate.distance_m,
                target=candidate.target,
                vehicle_yaw_rad=candidate.vehicle_yaw_rad,
                preflight=preflight,
                projection=candidate.projection,
                predicted_visible_point_fraction=(
                    candidate.predicted_visible_point_fraction),
                viewpoint_clearance_m=candidate.viewpoint_clearance_m,
                score_components=score_components,
                score=final_score,
            )
            if selected is None or complete.score < selected.score:
                selected = complete
        if selected is not None:
            break
    if selected is None:
        raise ValueError(
            "image-valid viewpoints exist, but none has a safe route")
    planning_finished = time.perf_counter()
    return AutoViewResult(
        selected=selected,
        attempted_candidates=attempted,
        physically_feasible_candidates=len(geometric_candidates),
        fully_framed_candidates=len(fully_framed),
        routed_candidates=routed_candidates,
        route_feasible_candidates=route_feasible_candidates,
        candidate_evaluation_time_ms=(
            1000.0 * (candidate_evaluation_finished - planning_started)),
        route_search_time_ms=(
            1000.0 * (planning_finished - route_search_started)),
        planner_time_ms=1000.0 * (planning_finished - planning_started),
    )


def plan_both_sides(
        map_geometry, start_pose, plant_x, row_y,
        transit_z=TRANSIT_Z):
    """Select one independently optimised, safely routed view per row face."""
    return tuple(
        plan_best_view(
            map_geometry,
            start_pose,
            plant_x,
            row_y,
            transit_z,
            side_options=(side_option,),
        )
        for side_option in SIDE_OPTIONS
    )


def plan_requested_sides(
        map_geometry, start_pose, plant_x, row_y,
        transit_z=TRANSIT_Z, side_policy="both", views_per_side=1):
    """Plan both faces or one explicitly requested crop-row face."""
    if side_policy == "both":
        options = SIDE_OPTIONS
    else:
        options = tuple(
            option for option in SIDE_OPTIONS if option[0] == side_policy)
        if not options:
            raise ValueError(
                "side policy must be both, positive_y or negative_y")
    views_per_side = int(views_per_side)
    if views_per_side not in (1, 2):
        raise ValueError("views_per_side must be 1 or 2")
    if views_per_side == 1:
        offset_groups = (AZIMUTH_OFFSETS_DEG,)
    else:
        # Use one oblique observation from each direction.
        offset_groups = ((-15.0,), (15.0,))
    return tuple(
        plan_best_view(
            map_geometry,
            start_pose,
            plant_x,
            row_y,
            transit_z,
            side_options=(option,),
            azimuth_offsets_deg=offsets,
        )
        for option in options
        for offsets in offset_groups
    )


def as_side_shot(auto_result, name="auto_best"):
    """Adapt the selected view to the existing audited artifact writer."""
    selected = auto_result.selected
    return SideShotPlan(
        name=str(name),
        azimuth_deg=selected.azimuth_deg,
        vehicle_yaw_deg=math.degrees(selected.vehicle_yaw_rad),
        preflight=selected.preflight,
    )
