"""Click-driven, non-arming preflight for two crop side photographs."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from .free_space import TRANSIT_Z
from .relocalize import transform_points
from .viewpoint_geometry import CameraConfig, CandidateSampling, wrap_pi
from .viewpoint_inspection import (
    InspectionPreflight,
    plan_inspection_preflight,
    pose_in_saved_map,
    preflight_record,
    quaternion_yaw,
    transform_from_tf,
)
from .viewpoint_selection import PerspectiveRequest, SelectionConfig


SIDE_SPECS = (
    ("side_pos_y", 90.0),
    ("side_neg_y", 270.0),
)


@dataclass(frozen=True)
class SideShotPlan:
    """One named, independently validated side-shot preflight."""

    name: str
    azimuth_deg: float
    vehicle_yaw_deg: float
    preflight: InspectionPreflight


def side_shot_requests(
        distance: float = 2.0, pitch_deg: float = 62.1,
        side_specs=SIDE_SPECS):
    """Return named requests and body yaw for zero relative gimbal yaw."""
    distance = float(distance)
    pitch_deg = float(pitch_deg)
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("side-shot distance must be positive and finite")
    if not math.isfinite(pitch_deg) or not -10.0 <= pitch_deg <= 100.0:
        raise ValueError(
            "side-shot pitch is outside the simulated gimbal limits")
    requests = []
    for name, azimuth_deg in side_specs:
        azimuth_deg = float(azimuth_deg) % 360.0
        request = PerspectiveRequest(
            math.radians(azimuth_deg), math.radians(pitch_deg), distance)
        vehicle_yaw = wrap_pi(math.radians(azimuth_deg) + math.pi)
        requests.append((str(name), azimuth_deg, vehicle_yaw, request))
    return tuple(requests)


def plan_side_shots(
        map_geometry, start_pose_saved: Sequence[float],
        target: Sequence[float], distance: float = 2.0,
        pitch_deg: float = 62.1, transit_z: float = TRANSIT_Z,
        minimum_standoff: float = 1.0,
        maximum_standoff: float = 4.0,
        side_targets: dict | None = None) -> tuple[SideShotPlan, ...]:
    """Plan both exact cross-row shots from the measured saved-map start."""
    camera = CameraConfig(
        minimum_standoff=float(minimum_standoff),
        maximum_standoff=float(maximum_standoff),
        desired_standoff=float(distance),
    )
    plans = []
    for name, azimuth_deg, vehicle_yaw, request in side_shot_requests(
            distance, pitch_deg):
        # A selected catalogue face is an operator-request identifier, not a universal optical aim point.
        shot_target = tuple(float(value) for value in (
            side_targets.get(name, target) if side_targets else target))
        sampling = CandidateSampling(
            azimuths_deg=(azimuth_deg,),
            elevations_deg=(float(pitch_deg),),
            distances=(float(distance),),
        )
        preflight = plan_inspection_preflight(
            map_geometry, start_pose_saved, shot_target, request,
            camera=camera,
            sampling=sampling,
            selection_config=SelectionConfig(
                maximum_direction_error=math.radians(0.5),
                maximum_distance_error=0.02),
            vehicle_yaw_map=vehicle_yaw,
            transit_z=float(transit_z),
        )
        plans.append(SideShotPlan(
            name=name,
            azimuth_deg=azimuth_deg,
            vehicle_yaw_deg=math.degrees(vehicle_yaw),
            preflight=preflight,
        ))
    return tuple(plans)


def write_side_shot_artifacts(
        plans: Sequence[SideShotPlan], output_dir: str, map_path: str,
        live_pose: Sequence[float], saved_from_live: np.ndarray,
        target: Sequence[float]) -> dict:
    """Write Phase 6a-compatible artifacts and a click-level manifest."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    entries = []
    for plan in plans:
        route_path = directory / f"{plan.name}_route.npy"
        preflight_path = directory / f"{plan.name}_preflight.json"
        route_name = None
        if plan.preflight.ready:
            np.save(route_path, np.asarray(
                plan.preflight.route.waypoints, dtype=np.float64))
            route_name = str(route_path)
        record = preflight_record(
            plan.preflight, map_path, live_pose, saved_from_live,
            run_id=f"side_shot_{plan.name}", timestamp=timestamp,
            route_path=route_name)
        record["side_shot"] = {
            "name": plan.name,
            "requested_gimbal_yaw_deg": 0.0,
            "requested_gimbal_pitch_deg": math.degrees(
                plan.preflight.request.elevation),
            "vehicle_yaw_deg": plan.vehicle_yaw_deg,
        }
        preflight_path.write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8")
        entries.append({
            "name": plan.name,
            "status": plan.preflight.status.value,
            "detail": plan.preflight.detail,
            "preflight_json": str(preflight_path),
            "route_file": route_name,
            "vehicle_yaw_deg": plan.vehicle_yaw_deg,
            "gimbal_yaw_deg": 0.0,
            "gimbal_pitch_deg": math.degrees(
                plan.preflight.request.elevation),
        })
    manifest = {
        "schema": "greenhouse_click_side_shots/v1",
        "timestamp_utc": timestamp,
        "target_saved_map": [float(value) for value in target],
        "dry_run": True,
        "flight_command_capable": False,
        "flight_commanded": False,
        "shots": entries,
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(args=None):
    """Listen for RViz clicks and write two reviewed flight artifacts."""
    from geometry_msgs.msg import Point, PointStamped
    from nav_msgs.msg import Odometry
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
    from tf2_ros import Buffer, TransformListener
    from tf2_ros import (
        ConnectivityException,
        ExtrapolationException,
        LookupException,
    )
    from visualization_msgs.msg import Marker, MarkerArray

    from .map_geometry import geometry

    class ClickSideShotPlanner(Node):
        def __init__(self):
            super().__init__("click_side_shot_planner")
            self.map_path = str(self.declare_parameter(
                "map_path", "/root/masters-thesis/corridor_map3.npy").value)
            self.output_dir = str(self.declare_parameter(
                "output_dir", "/root/masters-thesis/side_shot_plan").value)
            self.distance = float(self.declare_parameter(
                "distance", 2.0).value)
            self.pitch_deg = float(self.declare_parameter(
                "pitch_deg", 62.1).value)
            self.transit_z = float(self.declare_parameter(
                "transit_z", TRANSIT_Z).value)
            self.click_coordinates_are_saved_map = bool(
                self.declare_parameter(
                    "click_coordinates_are_saved_map", True).value)
            self.marker_frame = str(self.declare_parameter(
                "marker_frame", "map").value)

            cloud = np.load(self.map_path)
            if cloud.ndim != 2 or cloud.shape[1] != 3:
                raise ValueError(
                    f"expected an Nx3 saved map, got {cloud.shape}")
            self.map_geometry = geometry(cloud)
            self.live_pose = None
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.tf_errors = (
                LookupException, ConnectivityException,
                ExtrapolationException)
            sub_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST, depth=5)
            marker_qos = QoSProfile(
                depth=1, reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL)
            self.create_subscription(
                Odometry, "/odometry", self.odom_cb, sub_qos)
            self.create_subscription(
                PointStamped, "/clicked_point", self.click_cb, 10)
            self.marker_pub = self.create_publisher(
                MarkerArray, "/side_shot_plan", marker_qos)
            self.get_logger().warn(
                "CLICK SIDE-SHOT DRY RUN: clicks plan +y/-y views at %.1f "
                "deg pitch and %.2f m stand-off. This node cannot arm or "
                "publish PX4 commands." % (self.pitch_deg, self.distance))

        def odom_cb(self, message):
            position = message.pose.pose.position
            orientation = message.pose.pose.orientation
            self.live_pose = (
                float(position.x), float(position.y), float(position.z),
                quaternion_yaw(
                    orientation.x, orientation.y,
                    orientation.z, orientation.w),
            )

        def current_transform(self):
            message = self.tf_buffer.lookup_transform(
                "saved_map", "map", Time())
            translation = message.transform.translation
            rotation = message.transform.rotation
            return transform_from_tf(
                (translation.x, translation.y, translation.z),
                (rotation.x, rotation.y, rotation.z, rotation.w))

        def click_cb(self, message):
            if self.live_pose is None:
                self.get_logger().error(
                    "Click ignored: no current /odometry sample yet")
                return
            try:
                saved_from_live = self.current_transform()
            except self.tf_errors:
                self.get_logger().error(
                    "Click ignored: no accepted saved_map -> map transform")
                return
            clicked = np.array([
                message.point.x, message.point.y, message.point.z],
                dtype=float)
            if not np.isfinite(clicked).all():
                self.get_logger().error("Click ignored: target is not finite")
                return
            target = (
                tuple(float(value) for value in clicked)
                if self.click_coordinates_are_saved_map else
                tuple(float(value) for value in transform_points(
                    clicked.reshape(1, 3), saved_from_live)[0]))
            start_saved = pose_in_saved_map(self.live_pose, saved_from_live)
            plans = plan_side_shots(
                self.map_geometry, start_saved, target,
                distance=self.distance, pitch_deg=self.pitch_deg,
                transit_z=self.transit_z)
            manifest = write_side_shot_artifacts(
                plans, self.output_dir, self.map_path, self.live_pose,
                saved_from_live, target)
            self.publish_plan(target, plans)
            ready = sum(
                plan.preflight.ready for plan in plans)
            self.get_logger().info(
                "Clicked target saved-map (%.3f, %.3f, %.3f): %d/2 side "
                "shots ready. Review %s/manifest.json; no flight commanded."
                % (target + (ready, self.output_dir)))
            for entry in manifest["shots"]:
                log = (
                    self.get_logger().info
                    if entry["status"] == "PREFLIGHT_READY"
                    else self.get_logger().error)
                log("%s: %s %s" % (
                    entry["name"], entry["status"], entry["detail"]))

        def publish_plan(self, target, plans):
            markers = MarkerArray()
            clear = Marker()
            clear.header.frame_id = self.marker_frame
            clear.action = Marker.DELETEALL
            markers.markers.append(clear)
            markers.markers.append(self.point_marker(
                "side_target", 0, target, Marker.SPHERE,
                (0.25, 0.25, 0.25), (0.9, 0.1, 0.35, 1.0)))
            for index, plan in enumerate(plans):
                preflight = plan.preflight
                if preflight.selection is None:
                    continue
                candidate = preflight.selection.selected.candidate
                colour = (
                    (0.05, 0.75, 0.25, 1.0) if preflight.ready
                    else (0.9, 0.15, 0.1, 1.0))
                markers.markers.append(self.point_marker(
                    "side_camera", index, candidate.camera_position,
                    Marker.CUBE, (0.25, 0.25, 0.25), colour))
                ray = self.base_marker("side_ray", index, Marker.ARROW)
                ray.points = [self.point(candidate.camera_position),
                              self.point(target)]
                ray.scale.x, ray.scale.y, ray.scale.z = 0.05, 0.10, 0.12
                ray.color.r, ray.color.g, ray.color.b, ray.color.a = colour
                markers.markers.append(ray)
                if preflight.route is not None and preflight.route.valid:
                    route = self.base_marker(
                        "side_route", index, Marker.LINE_STRIP)
                    route.points = [self.point(w[:3])
                                    for w in preflight.route.waypoints]
                    route.scale.x = 0.06
                    route.color.r, route.color.g = 1.0, 0.55
                    route.color.a = 0.9
                    markers.markers.append(route)
            self.marker_pub.publish(markers)

        def base_marker(self, namespace, marker_id, marker_type):
            marker = Marker()
            marker.header.frame_id = self.marker_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns, marker.id = namespace, marker_id
            marker.type, marker.action = marker_type, Marker.ADD
            marker.pose.orientation.w = 1.0
            return marker

        def point_marker(self, namespace, marker_id, values, marker_type,
                         scale, colour):
            marker = self.base_marker(namespace, marker_id, marker_type)
            marker.pose.position = self.point(values)
            marker.scale.x, marker.scale.y, marker.scale.z = scale
            (marker.color.r, marker.color.g,
             marker.color.b, marker.color.a) = colour
            return marker

        @staticmethod
        def point(values):
            point = Point()
            point.x, point.y, point.z = (float(value) for value in values)
            return point

    rclpy.init(args=args)
    node = ClickSideShotPlanner()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
