"""Build reviewable side-shot routes from PX4's GPS-backed pose."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from greenhouse_inspection.free_space import TRANSIT_Z
from greenhouse_inspection.png_target_preflight import (
    PngSelection,
    build_preflight_set,
)
from greenhouse_inspection.viewpoint_execution import px4_pose_in_saved_map
from greenhouse_inspection.viewpoint_validation import (
    associate_target_box,
    optical_boxes_from_geometry,
)

from .auto_viewpoint import (
    plant_roi_corners,
    plant_roi_sample_points,
    project_plant_region,
    visible_point_fraction,
)
from .static_greenhouse import (
    SELECTION_SCHEMA,
    crop_catalogue,
    rows,
    static_geometry,
)


@dataclass(frozen=True)
class GpsSelection:
    """Validated intent loaded from the separate static-map UI."""

    map_path: str
    distance: float
    pitch_deg: float
    targets: tuple


def px4_quaternion_yaw(quaternion):
    """Return PX4 NED yaw from a ``(w, x, y, z)`` quaternion."""
    values = tuple(float(value) for value in quaternion)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("PX4 quaternion must contain four finite values")
    w_value, x_value, y_value, z_value = values
    return math.atan2(
        2.0 * (w_value * z_value + x_value * y_value),
        1.0 - 2.0 * (y_value * y_value + z_value * z_value),
    )


def load_selection(record, target_id=0):
    """Validate a non-arming static-map selection."""
    required = {
        "schema": SELECTION_SCHEMA,
        "mode": "offline_surveyed_map_only",
        "dry_run": True,
        "flight_command_capable": False,
        "flight_commanded": False,
        "executable_by_viewpoint_execution": False,
    }
    for key, expected in required.items():
        if record.get(key) != expected:
            raise ValueError(
                "selection %r must be %r, got %r" % (
                    key, expected, record.get(key)))
    map_path = record.get("map_path")
    if not isinstance(map_path, str) or not map_path:
        raise ValueError("selection map_path is missing")
    distance = float(record.get("distance_m", float("nan")))
    pitch_deg = float(record.get("pitch_deg", float("nan")))
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("distance_m must be positive and finite")
    if not math.isfinite(pitch_deg):
        raise ValueError("pitch_deg must be finite")
    targets = []
    entries = record.get("targets")
    if not isinstance(entries, list) or not entries:
        raise ValueError("selection contains no targets")
    for entry in entries:
        identifier = (
            entry.get("target_id") if isinstance(entry, dict) else None)
        values = (
            entry.get("target_greenhouse_map")
            if isinstance(entry, dict) else None)
        point = np.asarray(values, dtype=float)
        if not isinstance(identifier, int) or identifier <= 0:
            raise ValueError("target IDs must be positive integers")
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("target coordinates must contain finite x/y/z")
        if target_id in (0, identifier):
            targets.append((
                identifier, tuple(float(value) for value in point)))
    if target_id and not targets:
        raise ValueError("selection does not contain target_id %d" % target_id)
    return GpsSelection(map_path, distance, pitch_deg, tuple(targets))


def px4_ned_to_greenhouse_pose(position_ned, yaw_ned):
    """Convert PX4 NED pose into the aligned greenhouse ENU frame."""
    return px4_pose_in_saved_map(position_ned, yaw_ned, np.eye(4))


def annotate_gps_preflights(result, output_dir):
    """Replace inherited SLAM wording with the actual GPS pose provenance."""
    geometry = static_geometry()
    optical_boxes = optical_boxes_from_geometry(geometry)
    catalogue = {item["target_id"]: item for item in crop_catalogue()}
    for target in result["targets"]:
        target_id = int(target["target_id"])
        source = catalogue[target_id]
        plant_x = float(source["target_greenhouse_map"][0])
        row_y = float(rows()[source["row_index"]])
        for shot in target["shots"]:
            path = Path(shot["preflight_json"])
            record = json.loads(path.read_text(encoding="utf-8"))
            previous = record["pose_source"]
            record["pose_source"] = {
                "odometry_topic": "/fmu/out/vehicle_odometry",
                "source_frame": "PX4 NED",
                "live_frame": "map",
                "planning_frame": "saved_map",
                "localisation_source": (
                    "PX4 EKF using Gazebo simulated GPS and IMU"),
                "map_alignment": (
                    "identity: Gazebo world ENU equals surveyed map ENU"),
                "saved_from_live_xyzyaw": [0.0, 0.0, 0.0, 0.0],
                "vehicle_pose_live_xyzyaw": previous[
                    "vehicle_pose_live_xyzyaw"],
                "vehicle_pose_saved_xyzyaw": previous[
                    "vehicle_pose_saved_xyzyaw"],
            }
            record["planner_map"] = {
                "type": "surveyed_static_greenhouse",
                "source_schema": "gps_greenhouse_static_map/v1",
                "slam_used": False,
                "gps_used": True,
            }
            if "auto_viewpoint" not in record:
                request = record["request"]
                selected = record["selected"]
                camera_position = selected["camera_pose_map"]["position"]
                side = (
                    "positive_y"
                    if 0.0 < float(request["azimuth_deg"]) < 180.0
                    else "negative_y")
                projection = project_plant_region(
                    camera_position,
                    record["target_saved_map"],
                    plant_roi_corners(plant_x, row_y),
                )
                target_box = associate_target_box(
                    record["target_saved_map"],
                    optical_boxes[:len(rows())],
                )
                visible_fraction = visible_point_fraction(
                    camera_position,
                    plant_roi_sample_points(
                        plant_x, record["target_saved_map"][1]),
                    optical_boxes,
                    target_box,
                )
                record["auto_viewpoint"] = {
                    "mode": "fixed_distance_pitch_baseline",
                    "source_target_id": target_id,
                    "source_crop_id": source["crop_id"],
                    "selected_side": side,
                    "selected_azimuth_deg": request["azimuth_deg"],
                    "selected_azimuth_offset_deg": 0.0,
                    "selected_vehicle_yaw_deg": selected[
                        "vehicle_pose_map"][3],
                    "selected_pitch_deg": request["elevation_deg"],
                    "selected_distance_m": request["distance_m"],
                    "selected_horizontal_standoff_m": (
                        float(request["distance_m"])
                        * math.cos(math.radians(
                            float(request["elevation_deg"])))),
                    "selected_camera_height_m": camera_position[2],
                    "selected_aim_height_m": record[
                        "target_saved_map"][2],
                    "selected_aim_point": record["target_saved_map"],
                    "attempted_candidates": 1,
                    "physically_feasible_candidates": 1,
                    "fully_framed_candidates": int(
                        projection.fully_inside_margin),
                    "routed_candidates": 1,
                    "route_feasible_candidates": 1,
                    "candidate_evaluation_time_ms": None,
                    "route_search_time_ms": None,
                    "planner_time_ms": None,
                    "predicted_visible_sample_fraction": visible_fraction,
                    "predicted_visibility_sample_count": 9,
                    "predicted_plant_projection": {
                        "u_bounds_deg": [
                            projection.u_min_deg, projection.u_max_deg],
                        "v_bounds_deg": [
                            projection.v_min_deg, projection.v_max_deg],
                        "horizontal_frame_fraction": (
                            projection.horizontal_fill),
                        "vertical_frame_fraction": projection.vertical_fill,
                        "image_area_fraction": projection.image_fraction,
                        "normalised_centre_error": projection.centre_error,
                        "minimum_normalised_edge_margin": (
                            projection.minimum_edge_margin),
                        "overflow": projection.overflow,
                        "fully_inside_92_percent_fov": (
                            projection.fully_inside_margin),
                    },
                    "score": None,
                    "score_definition": (
                        "fixed baseline; no candidate ranking performed"),
                }
            path.write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8")
    manifest_path = Path(output_dir) / "live_preflight_manifest.json"
    result["pose_source"] = (
        "PX4 EKF using Gazebo simulated GPS and IMU")
    result["slam_used"] = False
    result["gps_used"] = True
    manifest_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def build_gps_preflight_set(
        selection, output_dir, position_ned, yaw_ned,
        transit_z=TRANSIT_Z):
    """Plan from one measured PX4 state without any SLAM input."""
    live_pose = px4_ned_to_greenhouse_pose(position_ned, yaw_ned)
    inherited_selection = PngSelection(
        selection.map_path,
        selection.distance,
        selection.pitch_deg,
        selection.targets,
    )
    # The operator may select either catalogue face, but a two-sided fixed baseline must aim at the physical face.
    catalogue = {item["target_id"]: item for item in crop_catalogue()}
    side_targets_by_id = {}
    for identifier, _ in selection.targets:
        source = catalogue[int(identifier)]
        plant_x = float(source["target_greenhouse_map"][0])
        row_y = float(rows()[source["row_index"]])
        aim_z = float(source["target_greenhouse_map"][2])
        side_targets_by_id[int(identifier)] = {
            "side_neg_y": (plant_x, row_y - 0.30, aim_z),
            "side_pos_y": (plant_x, row_y + 0.30, aim_z),
        }
    result = build_preflight_set(
        inherited_selection,
        static_geometry(),
        output_dir,
        live_pose,
        np.eye(4),
        float(transit_z),
        side_targets_by_id=side_targets_by_id,
    )
    result["schema"] = "gps_greenhouse_live_preflight_set/v1"
    result["source_schema"] = SELECTION_SCHEMA
    return annotate_gps_preflights(result, output_dir)


def main(args=None):
    """Wait for one PX4 pose, write fresh routes, and exit without arming."""
    import rclpy
    from px4_msgs.msg import VehicleOdometry
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )

    class GpsTargetPreflight(Node):
        def __init__(self):
            super().__init__("gps_target_preflight")
            selection_path = str(self.declare_parameter(
                "selection_json",
                "gps_viewpoint_plan/"
                "selected_targets.json").value)
            self.output_dir = str(self.declare_parameter(
                "output_dir",
                "gps_viewpoint_preflight").value)
            target_id = int(self.declare_parameter("target_id", 0).value)
            self.transit_z = float(self.declare_parameter(
                "transit_z", TRANSIT_Z).value)
            self.timeout_sec = float(self.declare_parameter(
                "timeout_sec", 60.0).value)
            record = json.loads(
                Path(selection_path).read_text(encoding="utf-8"))
            self.selection = load_selection(record, target_id)
            self.completed = False
            self.started = self.get_clock().now()
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
            )
            self.create_subscription(
                VehicleOdometry,
                "/fmu/out/vehicle_odometry",
                self.odometry_callback,
                qos,
            )
            self.timer = self.create_timer(0.25, self.check_timeout)
            self.get_logger().warn(
                "GPS PREFLIGHT ONLY: waiting for PX4 vehicle odometry. "
                "No SLAM topics, PX4 command publishers or flight commands "
                "are used by this node.")

        def odometry_callback(self, message):
            if self.completed:
                return
            position = np.asarray(message.position, dtype=float)
            yaw_ned = px4_quaternion_yaw(tuple(message.q))
            result = build_gps_preflight_set(
                self.selection,
                self.output_dir,
                position,
                yaw_ned,
                self.transit_z,
            )
            ready = sum(
                target["ready_shots"] for target in result["targets"])
            self.get_logger().info(
                "GPS_PREFLIGHT_READY: %d target(s), %d side shot(s) ready. "
                "Review %s/live_preflight_manifest.json; no flight commanded."
                % (len(result["targets"]), ready, self.output_dir))
            self.completed = True
            self.timer.cancel()

        def check_timeout(self):
            elapsed = (
                self.get_clock().now() - self.started).nanoseconds / 1e9
            if elapsed > self.timeout_sec:
                self.get_logger().error(
                    "Timed out waiting for /fmu/out/vehicle_odometry")
                self.completed = True
                self.timer.cancel()

    rclpy.init(args=args)
    node = GpsTargetPreflight()
    try:
        while rclpy.ok() and not node.completed:
            rclpy.spin_once(node, timeout_sec=0.25)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
