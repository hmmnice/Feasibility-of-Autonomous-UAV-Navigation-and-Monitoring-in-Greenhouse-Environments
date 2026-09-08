"""non-arming preflight for one targeted crop inspection."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .free_space import TRANSIT_Z
from .map_geometry import geometry
from .relocalize import decompose, yaw_transform
from .viewpoint_geometry import (
    CameraConfig,
    CandidateSampling,
    generate_candidates,
)
from .viewpoint_route import RoutePlan, RouteStatus, plan_route_to_viewpoint
from .viewpoint_selection import (
    NoAcceptableViewpointError,
    NoValidViewpointError,
    PerspectiveRequest,
    SelectionConfig,
    SelectionResult,
    select_viewpoint,
)
from .viewpoint_validation import (
    FlightBounds,
    ValidationConfig,
    ValidationResult,
    associate_target_box,
    collision_boxes_from_geometry,
    flight_bounds_from_geometry,
    optical_boxes_from_geometry,
    validate_candidates,
)


FLIGHT_COMMAND_CAPABLE = False
Pose4 = tuple[float, float, float, float]


class PreflightStatus(str, Enum):
    """Stable outcomes suitable for logs, tests and dissertation evidence."""

    PREFLIGHT_READY = "PREFLIGHT_READY"
    NO_ACCEPTABLE_VIEWPOINT = "NO_ACCEPTABLE_VIEWPOINT"
    NO_SAFE_ROUTE = "NO_SAFE_ROUTE"


@dataclass(frozen=True)
class InspectionPreflight:
    """Complete Phase 6a decision chain for one current vehicle pose."""

    status: PreflightStatus
    start_pose_saved: Pose4
    target: tuple[float, float, float]
    request: PerspectiveRequest
    validations: tuple[ValidationResult, ...]
    selection: Optional[SelectionResult]
    route: Optional[RoutePlan]
    detail: str = ""

    @property
    def ready(self) -> bool:
        return self.status is PreflightStatus.PREFLIGHT_READY


def _finite_pose(values: Sequence[float], name: str) -> Pose4:
    if len(values) != 4:
        raise ValueError(f"{name} must contain x, y, z and yaw")
    pose = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in pose):
        raise ValueError(f"{name} must contain finite values")
    return pose


def pose_in_saved_map(
        live_pose: Sequence[float], saved_from_live: np.ndarray) -> Pose4:
    """Transform a level live-map pose into the saved-map frame."""
    live_pose = _finite_pose(live_pose, "live_pose")
    transform = np.asarray(saved_from_live, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("saved_from_live must be a finite 4x4 transform")
    composed = transform @ yaw_transform(*live_pose)
    return tuple(float(value) for value in decompose(composed))


def quaternion_yaw(x: float, y: float, z: float, w: float) -> float:
    """Return ENU yaw from a finite ROS quaternion."""
    values = tuple(float(value) for value in (x, y, z, w))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("quaternion must contain finite values")
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        raise ValueError("quaternion must not be zero")
    x, y, z, w = (value / norm for value in values)
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def transform_from_tf(translation, rotation) -> np.ndarray:
    """Build the relocalizer's saved-from-live 4-DOF transform."""
    position = tuple(float(value) for value in translation)
    if (len(position) != 3
            or not all(math.isfinite(value) for value in position)):
        raise ValueError("translation must contain three finite values")
    yaw = quaternion_yaw(*rotation)
    return yaw_transform(*position, yaw)


def plan_inspection_preflight(
        map_geometry, start_pose_saved: Sequence[float],
        target: Sequence[float],
        request: PerspectiveRequest, camera: CameraConfig = CameraConfig(),
        sampling: CandidateSampling = CandidateSampling(),
        selection_config: SelectionConfig = SelectionConfig(),
        vehicle_yaw_map: float = 0.0,
        bounds: Optional[FlightBounds] = None,
        target_terminal_depth: float = 0.25,
        transit_z: float = TRANSIT_Z) -> InspectionPreflight:
    """Run Phases 2--5 from a measured/relocalised start pose."""
    start = _finite_pose(start_pose_saved, "start_pose_saved")
    target = tuple(float(value) for value in target)
    if len(target) != 3 or not all(math.isfinite(value) for value in target):
        raise ValueError("target must contain three finite values")

    collision_boxes = collision_boxes_from_geometry(map_geometry)
    optical_boxes = optical_boxes_from_geometry(map_geometry)
    target_box = associate_target_box(
        target, optical_boxes[:len(map_geometry.rows)])
    validation_config = ValidationConfig(
        bounds=bounds or flight_bounds_from_geometry(map_geometry),
        camera=camera,
        target_terminal_depth=target_terminal_depth,
    )
    candidates = generate_candidates(
        target, camera=camera, sampling=sampling,
        vehicle_yaw_map=vehicle_yaw_map)
    validations = tuple(validate_candidates(
        candidates, validation_config, collision_boxes, optical_boxes,
        target_box))

    try:
        selection = select_viewpoint(
            validations, request, collision_boxes, selection_config)
    except (NoValidViewpointError, NoAcceptableViewpointError) as error:
        return InspectionPreflight(
            PreflightStatus.NO_ACCEPTABLE_VIEWPOINT,
            start, target, request, validations, None, None, str(error))

    route = plan_route_to_viewpoint(
        selection, start[:3], collision_boxes, map_geometry.rows,
        start_yaw=start[3], transit_z=transit_z)
    if route.status is RouteStatus.NO_SAFE_ROUTE:
        return InspectionPreflight(
            PreflightStatus.NO_SAFE_ROUTE,
            start, target, request, validations, selection, route,
            route.detail)
    return InspectionPreflight(
        PreflightStatus.PREFLIGHT_READY,
        start, target, request, validations, selection, route)


def preflight_record(
        preflight: InspectionPreflight, map_path: str,
        live_pose: Sequence[float], saved_from_live: np.ndarray,
        run_id: Optional[str] = None, timestamp: Optional[str] = None,
        route_path: Optional[str] = None) -> dict:
    """Create explicit planned-vs-actual metadata before any flight exists."""
    live_pose = _finite_pose(live_pose, "live_pose")
    transform_pose = tuple(float(value) for value in decompose(
        np.asarray(saved_from_live, dtype=float)))
    now = timestamp or datetime.now(timezone.utc).isoformat()
    identifier = run_id or datetime.now(timezone.utc).strftime(
        "inspection_%Y%m%dT%H%M%S_%fZ")
    selection = preflight.selection
    route = preflight.route
    selected = None
    if selection is not None:
        score = selection.selected
        candidate = score.candidate
        selected = {
            "candidate_id": candidate.index,
            "camera_pose_map": {
                "position": list(candidate.camera_position),
                "yaw_deg": math.degrees(candidate.camera_yaw_map),
                "pitch_deg": math.degrees(candidate.camera_pitch),
            },
            "vehicle_pose_map": list(score.vehicle_position) + [
                math.degrees(candidate.vehicle_yaw_map)],
            "gimbal_yaw_deg": math.degrees(candidate.gimbal_yaw),
            "gimbal_pitch_deg": math.degrees(candidate.gimbal_pitch),
            "direction_error_deg": math.degrees(score.angular_error),
            "distance_error_m": score.distance_error,
            "mapped_obstacle_clearance_m": score.obstacle_clearance,
        }
    route_data = None
    if route is not None:
        route_data = {
            "status": route.status.value,
            "waypoints_saved_map": [list(point) for point in route.waypoints],
            "waypoint_count_including_start": len(route.waypoints),
            "inserted_waypoints": route.inserted_waypoints,
            "length_m": route.length,
            "all_segments_collision_free": route.valid,
            "route_file": route_path,
            "detail": route.detail,
        }
    return {
        "schema": "greenhouse_targeted_inspection_preflight/v1",
        "phase": "6a",
        "run_id": identifier,
        "timestamp_utc": now,
        "status": preflight.status.value,
        "detail": preflight.detail,
        "dry_run": True,
        "flight_command_capable": FLIGHT_COMMAND_CAPABLE,
        "flight_commanded": False,
        "map_path": str(map_path),
        "pose_source": {
            "odometry_topic": "/odometry",
            "live_frame": "map",
            "planning_frame": "saved_map",
            "saved_from_live_xyzyaw": list(transform_pose),
            "vehicle_pose_live_xyzyaw": list(live_pose),
            "vehicle_pose_saved_xyzyaw": list(preflight.start_pose_saved),
        },
        "target_saved_map": list(preflight.target),
        "request": {
            "azimuth_deg": math.degrees(preflight.request.azimuth),
            "elevation_deg": math.degrees(preflight.request.elevation),
            "distance_m": preflight.request.distance,
        },
        "candidate_count": len(preflight.validations),
        "valid_candidate_count": sum(
            result.valid for result in preflight.validations),
        "selected": selected,
        "route": route_data,
        "execution": {
            "actual_vehicle_pose_map": None,
            "actual_gimbal_yaw_deg": None,
            "actual_gimbal_pitch_deg": None,
            "gimbal_settled": False,
            "image_path": None,
            "achieved_direction_error_deg": None,
            "achieved_distance_error_m": None,
            "target_visibility_verified_in_image": None,
        },
    }


def main(args=None):
    """Wait for relocalisation + odometry, save a preflight, and exit."""
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener
    from tf2_ros import (
        ConnectivityException,
        ExtrapolationException,
        LookupException,
    )

    class ViewpointInspectionPreflight(Node):
        def __init__(self):
            super().__init__("viewpoint_inspection_preflight")
            self.map_path = str(self.declare_parameter(
                "map_path", "/root/masters-thesis/corridor_map3.npy").value)
            self.output_json = str(self.declare_parameter(
                "output_json",
                "/root/masters-thesis/phase6_inspection_preflight.json").value)
            self.output_route = str(self.declare_parameter(
                "output_route",
                "/root/masters-thesis/phase6_inspection_route.npy").value)
            target = self.declare_parameter(
                "target", [5.0, 2.10, 3.50]).value
            self.target = tuple(float(value) for value in target)
            self.request = PerspectiveRequest(
                math.radians(float(self.declare_parameter(
                    "requested_azimuth_deg", 90.0).value)),
                math.radians(float(self.declare_parameter(
                    "requested_elevation_deg", 35.0).value)),
                float(self.declare_parameter(
                    "requested_distance", 2.0).value),
            )
            self.vehicle_yaw = math.radians(float(self.declare_parameter(
                "vehicle_yaw_deg", 0.0).value))
            self.transit_z = float(self.declare_parameter(
                "transit_z", TRANSIT_Z).value)
            self.timeout = float(self.declare_parameter(
                "timeout_sec", 120.0).value)
            self.minimum_standoff = float(self.declare_parameter(
                "minimum_standoff", 1.0).value)
            self.maximum_standoff = float(self.declare_parameter(
                "maximum_standoff", 4.0).value)
            self.direction_tolerance = math.radians(float(
                self.declare_parameter(
                    "maximum_direction_error_deg", 30.0).value))
            self.distance_tolerance = float(self.declare_parameter(
                "maximum_distance_error", 0.5).value)

            cloud = np.load(self.map_path)
            if cloud.ndim != 2 or cloud.shape[1] != 3:
                raise ValueError(
                    f"expected an Nx3 saved map, got {cloud.shape}")
            self.map_geometry = geometry(cloud)
            self.live_pose = None
            self.completed = False
            self.started = self.get_clock().now()
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
            )
            self.create_subscription(Odometry, "/odometry", self.odom_cb, qos)
            self.timer = self.create_timer(0.25, self.try_preflight)
            self.get_logger().warn(
                "PHASE 6A DRY RUN: waiting for saved_map -> map and current "
                "/odometry. This node has no flight-command publishers and "
                "cannot arm or move the vehicle.")

        def odom_cb(self, message):
            position = message.pose.pose.position
            orientation = message.pose.pose.orientation
            self.live_pose = (
                float(position.x), float(position.y), float(position.z),
                quaternion_yaw(
                    orientation.x, orientation.y,
                    orientation.z, orientation.w),
            )

        def try_preflight(self):
            if self.completed or self.live_pose is None:
                self.check_timeout()
                return
            try:
                message = self.tf_buffer.lookup_transform(
                    "saved_map", "map", Time())
            except (
                    LookupException,
                    ConnectivityException,
                    ExtrapolationException):
                self.check_timeout()
                return
            translation = message.transform.translation
            rotation = message.transform.rotation
            saved_from_live = transform_from_tf(
                (translation.x, translation.y, translation.z),
                (rotation.x, rotation.y, rotation.z, rotation.w))
            start_saved = pose_in_saved_map(self.live_pose, saved_from_live)
            camera = CameraConfig(
                minimum_standoff=self.minimum_standoff,
                maximum_standoff=self.maximum_standoff,
                desired_standoff=self.request.distance,
            )
            sampling = CandidateSampling(distances=(self.request.distance,))
            preflight = plan_inspection_preflight(
                self.map_geometry, start_saved, self.target, self.request,
                camera=camera, sampling=sampling,
                selection_config=SelectionConfig(
                    self.direction_tolerance, self.distance_tolerance),
                vehicle_yaw_map=self.vehicle_yaw,
                transit_z=self.transit_z,
            )
            route_path = None
            if preflight.ready:
                route_path = self.output_route
                np.save(route_path, np.asarray(
                    preflight.route.waypoints, dtype=np.float64))
            record = preflight_record(
                preflight, self.map_path, self.live_pose, saved_from_live,
                route_path=route_path)
            Path(self.output_json).write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8")
            self.completed = True
            self.timer.cancel()
            if preflight.ready:
                self.get_logger().info(
                    "PREFLIGHT_READY: %d points, %.3f m, saved %s; no flight "
                    "was commanded."
                    % (len(preflight.route.waypoints), preflight.route.length,
                       self.output_json))
            else:
                self.get_logger().error(
                    f"{preflight.status.value}: {preflight.detail}; saved "
                    f"{self.output_json}; no flight was commanded.")

        def check_timeout(self):
            elapsed = (self.get_clock().now() - self.started).nanoseconds / 1e9
            if elapsed >= self.timeout:
                self.completed = True
                self.timer.cancel()
                self.get_logger().error(
                    "PRECHECK TIMEOUT: no accepted saved_map -> map transform "
                    "and current /odometry became available. No flight was "
                    "commanded.")

    rclpy.init(args=args)
    node = ViewpointInspectionPreflight()
    try:
        while rclpy.ok() and not node.completed:
            rclpy.spin_once(node, timeout_sec=0.25)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
