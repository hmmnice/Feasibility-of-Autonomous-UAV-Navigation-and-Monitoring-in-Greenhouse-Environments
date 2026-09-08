"""execution of one accepted targeted-inspection preflight."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .gimbal_geometry import joint_pitch_to_camera
from .hybrid_route import densify_route, route_to_px4
from .relocalize import decompose, transform_points
from .viewpoint_geometry import CAMERA_ORIGIN_IN_BASE
from .viewpoint_inspection import (
    pose_in_saved_map,
    transform_from_tf,
)
from .viewpoint_selection import PerspectiveRequest


PREFLIGHT_SCHEMA = "greenhouse_targeted_inspection_preflight/v1"
RESULT_SCHEMA = "greenhouse_targeted_inspection_result/v1"


def angle_error(first: float, second: float) -> float:
    """Absolute shortest angular difference in radians."""
    return abs(math.atan2(
        math.sin(float(first) - float(second)),
        math.cos(float(first) - float(second))))


def validate_preflight_record(record: dict, route: np.ndarray) -> np.ndarray:
    """Reject any record/route that is not the reviewed Phase 6a output."""
    required = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "PREFLIGHT_READY",
        "dry_run": True,
        "flight_command_capable": False,
        "flight_commanded": False,
    }
    for key, expected in required.items():
        if record.get(key) != expected:
            raise ValueError(
                f"preflight {key!r} must be {expected!r}, got "
                f"{record.get(key)!r}")
    route_record = record.get("route")
    if not isinstance(route_record, dict):
        raise ValueError("preflight has no route record")
    if (route_record.get("status") != "ROUTE_VALID"
            or route_record.get("all_segments_collision_free") is not True):
        raise ValueError("preflight route was not accepted as collision-free")
    route = np.asarray(route, dtype=float)
    if (route.ndim != 2 or route.shape[1] != 4 or len(route) < 2
            or not np.isfinite(route).all()):
        raise ValueError("execution route must be a finite Nx4 array, N >= 2")
    recorded = np.asarray(
        route_record.get("waypoints_saved_map", ()), dtype=float)
    if recorded.shape != route.shape or not np.allclose(
            recorded, route, atol=1e-9, rtol=0.0):
        raise ValueError(
            "route file does not exactly match the preflight JSON")
    if not isinstance(record.get("selected"), dict):
        raise ValueError("preflight has no selected viewpoint")
    target = record.get("target_saved_map")
    if not isinstance(target, list) or len(target) != 3:
        raise ValueError("preflight target must contain saved-map x, y and z")
    return route


def insert_ground_launch_waypoint(
        route: np.ndarray, launch_altitude: float) -> np.ndarray:
    """Keep a verified initial vertical ascent large enough for takeoff."""
    route = np.asarray(route, dtype=float)
    if (route.ndim != 2 or route.shape[1] != 4 or len(route) < 2
            or not np.isfinite(route).all()):
        raise ValueError(
            "route must be a finite Nx4 array with at least 2 rows")
    launch_altitude = float(launch_altitude)
    if not math.isfinite(launch_altitude) or launch_altitude <= 0.0:
        raise ValueError("launch altitude must be positive and finite")
    first, second = route[:2]
    vertical_gain = float(second[2] - first[2])
    horizontal = float(np.linalg.norm(second[:2] - first[:2]))
    if horizontal > 1e-6 or vertical_gain <= launch_altitude:
        return route.copy()
    launch = first.copy()
    launch[2] += launch_altitude
    launch[3] = second[3]
    return np.vstack((first, launch, route[1:]))


def densify_execution_route(
        route: np.ndarray,
        preserve_initial_launch: bool = False) -> np.ndarray:
    """Densify a route without splitting an explicit pad takeoff leg."""
    route = np.asarray(route, dtype=float)
    if (route.ndim != 2 or route.shape[1] != 5 or len(route) < 2
            or not np.isfinite(route).all()):
        raise ValueError("execution route must be a finite Nx5 array, N >= 2")
    if not preserve_initial_launch:
        return densify_route(route)
    if len(route) < 3:
        return route.copy()
    tail = densify_route(route[1:])
    return np.vstack((route[:2], tail[1:]))


def preflight_age_seconds(record: dict, now=None) -> float:
    """Age of an ISO-8601 preflight record in wall-clock seconds."""
    try:
        timestamp = datetime.fromisoformat(record["timestamp_utc"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "preflight timestamp_utc is missing or invalid") from error
    if timestamp.tzinfo is None:
        raise ValueError("preflight timestamp_utc must include a timezone")
    now = now or datetime.now(timezone.utc)
    return max(0.0, (now - timestamp).total_seconds())


def transform_change(
        recorded_xyzyaw: Sequence[float], current_saved_from_live: np.ndarray
        ) -> tuple[float, float]:
    """Translation/yaw change since Phase 6a recorded relocalisation."""
    recorded = np.asarray(recorded_xyzyaw, dtype=float)
    if recorded.shape != (4,) or not np.isfinite(recorded).all():
        raise ValueError("recorded transform must contain finite x,y,z,yaw")
    current = np.asarray(decompose(current_saved_from_live), dtype=float)
    return (
        float(np.linalg.norm(current[:3] - recorded[:3])),
        angle_error(current[3], recorded[3]),
    )


def pose_change(
        first: Sequence[float], second: Sequence[float]
        ) -> tuple[float, float]:
    """3D translation/yaw separation between two x,y,z,yaw poses."""
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if (first.shape != (4,) or second.shape != (4,)
            or not np.isfinite(first).all() or not np.isfinite(second).all()):
        raise ValueError("poses must contain finite x,y,z,yaw")
    return (
        float(np.linalg.norm(first[:3] - second[:3])),
        angle_error(first[3], second[3]),
    )


def px4_pose_in_saved_map(
        position_ned: Sequence[float], yaw_ned: float,
        saved_from_live: np.ndarray) -> tuple[float, float, float, float]:
    """PX4 NED pose -> live ENU -> saved-map ENU pose."""
    position = np.asarray(position_ned, dtype=float)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("PX4 position must contain finite north/east/down")
    live_pose = (
        float(position[1]), float(position[0]), float(-position[2]),
        math.pi / 2.0 - float(yaw_ned),
    )
    return pose_in_saved_map(live_pose, saved_from_live)


def saved_target_in_body(
        target_saved: Sequence[float], saved_from_live: np.ndarray,
        position_ned: Sequence[float], yaw_ned: float
        ) -> tuple[float, float, float]:
    """Express a saved-map target as a body-FLU point for the gimbal."""
    target = np.asarray(target_saved, dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("target must contain finite saved-map x,y,z")
    live = transform_points(
        target.reshape(1, 3), np.linalg.inv(saved_from_live))[0]
    target_ned = np.array([live[1], live[0], -live[2]], dtype=float)
    delta = target_ned - np.asarray(position_ned, dtype=float)
    # Kept local to avoid importing the ROS flight node. PX4 body is FRD;
    # the gimbal contract is ROS base_link FLU.
    forward = math.cos(yaw_ned) * delta[0] + math.sin(yaw_ned) * delta[1]
    right = -math.sin(yaw_ned) * delta[0] + math.cos(yaw_ned) * delta[1]
    return float(forward), float(-right), float(-delta[2])


def actual_view_metrics(
        vehicle_pose_saved: Sequence[float], gimbal_yaw: float,
        gimbal_pitch: float, target_saved: Sequence[float],
        request: PerspectiveRequest) -> dict:
    """Calculate achieved position and optical-axis errors after settling."""
    vehicle = np.asarray(vehicle_pose_saved, dtype=float)
    target = np.asarray(target_saved, dtype=float)
    if vehicle.shape != (4,) or target.shape != (3,):
        raise ValueError("vehicle pose/target have invalid dimensions")
    yaw = vehicle[3]
    ox, oy, oz = CAMERA_ORIGIN_IN_BASE
    camera = np.array([
        vehicle[0] + math.cos(yaw) * ox - math.sin(yaw) * oy,
        vehicle[1] + math.sin(yaw) * ox + math.cos(yaw) * oy,
        vehicle[2] + oz,
    ])
    target_to_camera = camera - target
    distance = float(np.linalg.norm(target_to_camera))
    if distance <= 1e-9:
        raise ValueError("actual camera and target positions coincide")
    achieved_direction = target_to_camera / distance
    direction_error = math.acos(float(np.clip(
        np.dot(achieved_direction, request.direction), -1.0, 1.0)))

    camera_yaw = yaw + float(gimbal_yaw)
    pitch = float(gimbal_pitch)
    optical = np.array([
        math.cos(pitch) * math.cos(camera_yaw),
        math.cos(pitch) * math.sin(camera_yaw),
        -math.sin(pitch),
    ])
    camera_to_target = (target - camera) / distance
    optical_error = math.acos(float(np.clip(
        np.dot(optical, camera_to_target), -1.0, 1.0)))
    return {
        "actual_camera_position_map": camera.tolist(),
        "actual_distance_m": distance,
        "achieved_distance_error_m": abs(distance - request.distance),
        "achieved_direction_error_deg": math.degrees(direction_error),
        "optical_axis_error_deg": math.degrees(optical_error),
    }


def execution_result_record(
        preflight: dict, status: str, flight_commanded: bool,
        actual_vehicle_pose=None, actual_gimbal_yaw=None,
        actual_gimbal_pitch=None, gimbal_settled=False,
        image_path=None, metrics=None,
        detail: str = "", safety_mode=None) -> dict:
    """Create a result without overwriting the reviewed preflight file."""
    record = deepcopy(preflight)
    record["schema"] = RESULT_SCHEMA
    record["phase"] = "6b"
    record["status"] = str(status)
    record["detail"] = str(detail)
    record["timestamp_result_utc"] = datetime.now(timezone.utc).isoformat()
    record["dry_run"] = not bool(flight_commanded)
    record["flight_command_capable"] = True
    record["flight_commanded"] = bool(flight_commanded)
    if safety_mode is not None:
        record["safety_mode"] = str(safety_mode)
    execution = record.setdefault("execution", {})
    execution.update({
        "actual_vehicle_pose_map": (
            list(actual_vehicle_pose)
            if actual_vehicle_pose is not None else None),
        "actual_gimbal_yaw_deg": (
            math.degrees(actual_gimbal_yaw)
            if actual_gimbal_yaw is not None else None),
        "actual_gimbal_pitch_deg": (
            math.degrees(actual_gimbal_pitch)
            if actual_gimbal_pitch is not None else None),
        "gimbal_settled": bool(gimbal_settled),
        "image_path": str(image_path) if image_path is not None else None,
        "target_visibility_verified_in_image": None,
    })
    if metrics:
        execution.update(metrics)
    return record


def main(args=None):
    import os

    from geometry_msgs.msg import PointStamped
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

    from .assisted_teleop import (
        clamp_to_obstacle,
        clamp_vertical,
        yaw_from_quat,
    )
    from .coverage_flight import (
        MAX_YAW_RATE,
        _body_flu_to_ned,
        _ned_to_body_flu,
        is_stuck,
        ramped_yaw,
        save_image,
        sidestep_scale,
    )
    from .gimbal_controller import PITCH_JOINT, YAW_JOINT
    from .path_follower import REACH_RADIUS

    tick_period = 0.1
    settle_samples_required = 15
    stuck_ticks_limit = 30
    sidestep_distance = 2.0

    class ViewpointInspectionExecutor(Node):
        def __init__(self):
            super().__init__("viewpoint_inspection_executor")
            self.preflight_path = str(self.declare_parameter(
                "preflight_json",
                "/root/masters-thesis/phase6_inspection_preflight.json").value)
            self.result_path = str(self.declare_parameter(
                "result_json",
                "/root/masters-thesis/phase6_inspection_result.json").value)
            self.out_dir = str(self.declare_parameter(
                "out_dir", "/root/masters-thesis/phase6_photos").value)
            self.start_flight = bool(self.declare_parameter(
                "start_flight", False).value)
            self.use_lidar_safety = bool(self.declare_parameter(
                "use_lidar_safety", True).value)
            self.return_after_capture = bool(self.declare_parameter(
                "return_after_capture", True).value)
            self.land_after_return = bool(self.declare_parameter(
                "land_after_return", False).value)
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
                self.declare_parameter(
                    "vehicle_settle_yaw_deg", 5.0).value))
            self.capture_timeout = float(self.declare_parameter(
                "capture_timeout_sec", 30.0).value)
            self.ground_launch_altitude = float(self.declare_parameter(
                "ground_launch_altitude_m", 1.2).value)
            self.ground_launch_speed = float(self.declare_parameter(
                "ground_launch_speed_mps", 0.45).value)
            self.ground_launch_max_saved_z = float(self.declare_parameter(
                "ground_launch_max_saved_z", 0.5).value)

            self.preflight = json.loads(Path(
                self.preflight_path).read_text(encoding="utf-8"))
            route_path = self.preflight.get("route", {}).get("route_file")
            if not route_path:
                raise ValueError("preflight does not name its route file")
            self.saved_route = validate_preflight_record(
                self.preflight, np.load(route_path))
            if (self.max_preflight_age > 0.0
                    and preflight_age_seconds(self.preflight)
                    > self.max_preflight_age):
                raise ValueError(
                    "preflight is stale; generate a fresh Phase 6a record")
            self.target_saved = np.asarray(
                self.preflight["target_saved_map"], dtype=float)
            request = self.preflight["request"]
            self.request = PerspectiveRequest(
                math.radians(float(request["azimuth_deg"])),
                math.radians(float(request["elevation_deg"])),
                float(request["distance_m"]),
            )
            self.recorded_transform = self.preflight[
                "pose_source"]["saved_from_live_xyzyaw"]

            self.completed = False
            if not self.start_flight:
                result = execution_result_record(
                    self.preflight, "EXECUTION_DISABLED", False,
                    detail="start_flight=false; artifact validation only",
                    safety_mode=(
                        "STATIC_MAP_PLUS_REACTIVE_LIDAR"
                        if self.use_lidar_safety else "STATIC_MAP_ONLY"))
                Path(self.result_path).write_text(
                    json.dumps(result, indent=2) + "\n", encoding="utf-8")
                self.get_logger().warn(
                    "START INTERLOCK: preflight/route accepted, but "
                    "start_flight=false. No ROS flight publishers were "
                    "created; no movement or capture was commanded.")
                self.completed = True
                return

            os.makedirs(self.out_dir, exist_ok=True)
            pub_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
            sub_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
            )
            # ros_gz_bridge advertises /camera as RELIABLE in this stack.
            camera_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
            )
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
                # The active PX4 DDS client publishes VehicleStatus as v1.
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
            self.current_yaw = 0.0
            self.armed = False
            self.nav_state = None
            self.latest_points = np.zeros((0, 3), dtype=np.float32)
            self.lidar_time = None
            self.latest_image = None
            self.image_time = None
            self.actual_gimbal_yaw = None
            self.actual_gimbal_pitch = None
            self.gimbal_settled = False
            self.saved_from_live = None
            self.waypoints = None
            self.index = 0
            self.returning = False
            self.holding_for_capture = False
            self.mission_done_holding = False
            self.hold_started = None
            self.vehicle_good_samples = 0
            self.gimbal_target_sent = False
            self.commanded_yaw = float("nan")
            self.tick = 0
            self.last_engage_tick = None
            self.stuck_ticks = 0
            self.prev_distance = float("inf")
            self.detour_target = None
            self.landing = False
            self.ever_armed = False
            self.ground_takeoff = False
            self.takeoff_complete = True
            self.takeoff_origin = None
            self.timer = self.create_timer(tick_period, self.timer_callback)
            safety_description = (
                "static map plus reactive LiDAR"
                if self.use_lidar_safety else "static-map-only safety")
            self.get_logger().warn(
                "START INTERLOCK RELEASED: start_flight=true. Waiting for "
                "fresh TF, matching start pose and PX4 state before "
                "publishing Offboard commands; safety=%s."
                % safety_description)

        def now_seconds(self):
            return self.get_clock().now().nanoseconds / 1e9

        def odom_cb(self, message):
            self.position = np.asarray(message.position, dtype=float)
            self.current_yaw = yaw_from_quat(tuple(message.q))

        def status_cb(self, message):
            self.armed = (
                message.arming_state == VehicleStatus.ARMING_STATE_ARMED)
            self.nav_state = message.nav_state
            self.ever_armed = self.ever_armed or self.armed
            if self.landing and self.ever_armed and not self.armed:
                self.completed = True
                self.timer.cancel()

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
                    "relocalisation changed since preflight: %.3f m, %.2f deg"
                    % (translation_change, math.degrees(yaw_change)))
                return False
            current_saved = px4_pose_in_saved_map(
                self.position, self.current_yaw, transform)
            distance, yaw_error = pose_change(
                current_saved, self.saved_route[0])
            if (distance > self.max_start_distance
                    or yaw_error > self.max_start_yaw):
                self.abort_before_flight(
                    "current pose no longer matches preflight start: "
                    "%.3f m, %.2f deg" % (
                        distance, math.degrees(yaw_error)))
                return False
            route_for_execution = self.saved_route
            inserted_ground_launch = False
            if current_saved[2] <= self.ground_launch_max_saved_z:
                route_for_execution = insert_ground_launch_waypoint(
                    self.saved_route, self.ground_launch_altitude)
                inserted_ground_launch = (
                    len(route_for_execution) > len(self.saved_route))
                if inserted_ground_launch:
                    self.get_logger().info(
                        "GROUND_LAUNCH: using %.2f m initial vertical "
                        "waypoint before densifying the verified ascent."
                        % self.ground_launch_altitude)
            execution_route = np.column_stack([
                route_for_execution,
                np.zeros(len(route_for_execution), dtype=float),
            ])
            execution_route[-1, 4] = 1.0
            dense = densify_execution_route(
                execution_route, inserted_ground_launch)
            self.saved_from_live = transform
            self.waypoints = route_to_px4(dense, transform)
            self.index = 1
            self.ground_takeoff = bool(inserted_ground_launch)
            self.takeoff_complete = not self.ground_takeoff
            self.takeoff_origin = self.position.copy()
            self.get_logger().info(
                "EXECUTION_READY: transform/start gates passed; "
                "%d dense waypoints." % len(self.waypoints))
            return True

        def abort_before_flight(self, reason):
            result = execution_result_record(
                self.preflight, "EXECUTION_REFUSED", False, detail=reason,
                safety_mode=(
                    "STATIC_MAP_PLUS_REACTIVE_LIDAR"
                    if self.use_lidar_safety else "STATIC_MAP_ONLY"))
            Path(self.result_path).write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8")
            self.get_logger().error(f"EXECUTION_REFUSED: {reason}")
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
                self.target_saved, self.saved_from_live,
                self.position, self.current_yaw)
            message = PointStamped()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = "base_link"
            message.point.x, message.point.y, message.point.z = target
            self.gimbal_pub.publish(message)

        def find_sidestep(self):
            for dx, dy, dz in (
                    (0.0, sidestep_distance, 0.0),
                    (0.0, -sidestep_distance, 0.0),
                    (0.0, 0.0, sidestep_distance)):
                scale = sidestep_scale(
                    dx, dy, dz, self.latest_points, -self.position[2])
                if scale >= 0.99:
                    n, e, d = _body_flu_to_ned(
                        dx, dy, dz, self.current_yaw)
                    return self.position + np.array([n, e, d])
            return None

        def safe_setpoint(self, target):
            distance = float(np.linalg.norm(self.position - target))
            if not self.use_lidar_safety:
                self.detour_target = None
                return target, distance
            if self.detour_target is not None:
                if float(np.linalg.norm(
                        self.position - self.detour_target)) < REACH_RADIUS:
                    self.detour_target = None
                else:
                    return self.detour_target, distance
            delta = target - self.position
            dx, dy, dz = _ned_to_body_flu(
                delta[0], delta[1], delta[2], self.current_yaw)
            dx, dy, dz, scale = clamp_to_obstacle(
                dx, dy, dz, self.latest_points)
            dz, vertical_scale = clamp_vertical(dz, -self.position[2])
            scale = min(scale, vertical_scale)
            self.stuck_ticks = (
                self.stuck_ticks + 1
                if is_stuck(scale, distance, self.prev_distance) else 0)
            self.prev_distance = distance
            if self.stuck_ticks > stuck_ticks_limit:
                self.detour_target = self.find_sidestep()
                self.stuck_ticks = 0
                if self.detour_target is None:
                    return self.position.copy(), distance
                return self.detour_target, distance
            n, e, d = _body_flu_to_ned(dx, dy, dz, self.current_yaw)
            return self.position + np.array([n, e, d]), distance

        def enter_capture_hold(self):
            self.holding_for_capture = True
            self.hold_started = self.now_seconds()
            self.vehicle_good_samples = 0
            self.gimbal_settled = False
            self.gimbal_target_sent = False
            self.get_logger().info(
                "Inspection pose reached; aiming gimbal and waiting for "
                "vehicle/gimbal/image settle gates.")

        def capture_if_settled(self, target, yaw):
            distance = float(np.linalg.norm(self.position - target))
            yaw_error = angle_error(self.current_yaw, yaw)
            if (distance <= self.vehicle_settle_distance
                    and yaw_error <= self.vehicle_settle_yaw):
                self.vehicle_good_samples += 1
            else:
                self.vehicle_good_samples = 0
            vehicle_settled = (
                self.vehicle_good_samples >= settle_samples_required)
            if vehicle_settled and not self.gimbal_target_sent:
                self.gimbal_settled = False
                self.publish_gimbal_target()
                self.gimbal_target_sent = True
            image_fresh = (
                self.image_time is not None
                and self.image_time >= self.hold_started)
            ready = (
                vehicle_settled
                and self.gimbal_target_sent
                and self.gimbal_settled
                and self.actual_gimbal_yaw is not None
                and self.actual_gimbal_pitch is not None
                and image_fresh)
            if ready:
                self.capture_image()
                return
            if self.now_seconds() - self.hold_started > self.capture_timeout:
                detail = (
                    "settle timeout: vehicle_settled=%s, "
                    "gimbal_target_sent=%s, gimbal_settled=%s, "
                    "gimbal_joints_received=%s, image_received=%s, "
                    "fresh_image=%s" % (
                        vehicle_settled,
                        self.gimbal_target_sent,
                        self.gimbal_settled,
                        self.actual_gimbal_yaw is not None
                        and self.actual_gimbal_pitch is not None,
                        self.latest_image is not None,
                        image_fresh))
                self.write_result(
                    "CAPTURE_FAILED", None, detail)
                self.begin_return()

        def capture_image(self):
            name = datetime.now(timezone.utc).strftime(
                "inspection_%Y%m%dT%H%M%S_%fZ.jpg")
            image_path = os.path.join(self.out_dir, name)
            try:
                save_image(self.latest_image, image_path)
            except ValueError as error:
                self.write_result("CAPTURE_FAILED", None, str(error))
                self.begin_return()
                return
            self.write_result("IMAGE_CAPTURED", image_path)
            self.get_logger().info(
                f"Captured settled target image {image_path}")
            self.begin_return()

        def write_result(self, status, image_path, detail=""):
            actual_pose = px4_pose_in_saved_map(
                self.position, self.current_yaw, self.saved_from_live)
            metrics = None
            if (self.actual_gimbal_yaw is not None
                    and self.actual_gimbal_pitch is not None):
                metrics = actual_view_metrics(
                    actual_pose, self.actual_gimbal_yaw,
                    self.actual_gimbal_pitch, self.target_saved, self.request)
            result = execution_result_record(
                self.preflight, status, True,
                actual_vehicle_pose=actual_pose,
                actual_gimbal_yaw=self.actual_gimbal_yaw,
                actual_gimbal_pitch=self.actual_gimbal_pitch,
                gimbal_settled=self.gimbal_settled,
                image_path=image_path, metrics=metrics, detail=detail,
                safety_mode=(
                    "STATIC_MAP_PLUS_REACTIVE_LIDAR"
                    if self.use_lidar_safety else "STATIC_MAP_ONLY"))
            Path(self.result_path).write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8")

        def begin_return(self):
            self.holding_for_capture = False
            if not self.return_after_capture:
                self.mission_done_holding = True
                return
            self.returning = True
            self.waypoints = self.waypoints[::-1].copy()
            self.index = 1
            self.commanded_yaw = float("nan")
            self.get_logger().info("Returning along the verified route.")

        def finish_return(self):
            self.returning = False
            if self.land_after_return:
                self.publish_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.get_logger().info("Returned to start; landing requested.")
                self.landing = True
            else:
                self.get_logger().info(
                    "Returned to preflight start; holding. Ctrl-C requests "
                    "landing only if this start is a safe pad.")
                self.mission_done_holding = True

        def timer_callback(self):
            if self.completed or self.landing:
                return
            if self.waypoints is None:
                self.try_accept_live_state()
                return
            if (self.use_lidar_safety
                    and (self.lidar_time is None
                         or self.now_seconds() - self.lidar_time > 1.0)):
                self.get_logger().error(
                    "Live LiDAR is stale; holding and withholding route "
                    "progress.", throttle_duration_sec=2.0)
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
                        "GROUND_LAUNCH_COMPLETE: switching to the verified "
                        "position route.")
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
            if self.mission_done_holding:
                self.publish_offboard_mode()
                self.publish_setpoint(
                    self.waypoints[-1, :3], self.waypoints[-1, 3])
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
                elif self.returning:
                    self.finish_return()
                else:
                    self.enter_capture_hold()
            self.engage_watchdog()
            self.tick += 1

        def request_land(self, reason):
            if self.start_flight and hasattr(self, "command_pub"):
                self.landing = True
                self.publish_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.get_logger().warn(f"Landing requested: {reason}")

    rclpy.init(args=args)
    node = ViewpointInspectionExecutor()
    try:
        while rclpy.ok() and not node.completed:
            rclpy.spin_once(node, timeout_sec=0.25)
    except KeyboardInterrupt:
        node.request_land("Ctrl-C")
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.1)
    except ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
