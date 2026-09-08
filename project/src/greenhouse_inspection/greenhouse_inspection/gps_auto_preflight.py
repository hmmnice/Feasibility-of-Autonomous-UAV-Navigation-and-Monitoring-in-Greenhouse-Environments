"""ROS adapter for automatic GPS-backed crop viewpoint optimisation."""

import json
import math
import os
from pathlib import Path

import numpy as np

from greenhouse_inspection.free_space import TRANSIT_Z
from greenhouse_inspection.side_shot_inspection import write_side_shot_artifacts
from greenhouse_inspection.viewpoint_route import route_length
from greenhouse_inspection.viewpoint_validation import (
    collision_boxes_from_geometry,
)

from .astar_route import plan_astar_route
from .auto_viewpoint import (
    ROI_BOTTOM_Z,
    ROI_TOP_Z,
    SCORE_WEIGHTS,
    as_side_shot,
    plan_requested_sides,
)
from .gps_preflight import (
    annotate_gps_preflights,
    load_selection,
    px4_ned_to_greenhouse_pose,
    px4_quaternion_yaw,
)
from .static_greenhouse import rows, static_geometry


def selected_target_entry(record, target_id=0):
    """Return exactly one validated catalogue entry from operator intent."""
    load_selection(record, target_id)
    entries = [
        entry for entry in record["targets"]
        if target_id in (0, entry["target_id"])
    ]
    if len(entries) != 1:
        raise ValueError(
            "automatic execution requires exactly one selected crop")
    entry = entries[0]
    row_index = int(entry.get("row_index", -1))
    plant_index = int(entry.get("plant_index", -1))
    if not 0 <= row_index < len(rows()) or plant_index < 0:
        raise ValueError("selection has invalid row/plant metadata")
    return entry


def auto_view_metadata(auto_result, source_entry):
    """Return the auditable inputs, search counts and chosen image geometry."""
    selected = auto_result.selected
    projection = selected.projection
    return {
        "mode": "automatic_plant_roi_search",
        "source_target_id": source_entry["target_id"],
        "source_crop_id": source_entry["crop_id"],
        "selected_side": selected.side,
        "selected_azimuth_deg": selected.azimuth_deg,
        "selected_azimuth_offset_deg": selected.azimuth_offset_deg,
        "selected_vehicle_yaw_deg": math.degrees(
            selected.vehicle_yaw_rad),
        "selected_pitch_deg": selected.pitch_deg,
        "selected_distance_m": selected.distance_m,
        "selected_horizontal_standoff_m": (
            selected.horizontal_standoff_m),
        "selected_camera_height_m": selected.camera_height_m,
        "selected_aim_height_m": selected.aim_height_m,
        "selected_aim_point": list(selected.target),
        "attempted_candidates": auto_result.attempted_candidates,
        "physically_feasible_candidates": (
            auto_result.physically_feasible_candidates),
        "fully_framed_candidates": auto_result.fully_framed_candidates,
        "routed_candidates": auto_result.routed_candidates,
        "route_feasible_candidates": auto_result.route_feasible_candidates,
        "candidate_evaluation_time_ms": (
            auto_result.candidate_evaluation_time_ms),
        "route_search_time_ms": auto_result.route_search_time_ms,
        "planner_time_ms": auto_result.planner_time_ms,
        "predicted_plant_projection": {
            "u_bounds_deg": [projection.u_min_deg, projection.u_max_deg],
            "v_bounds_deg": [projection.v_min_deg, projection.v_max_deg],
            "horizontal_frame_fraction": projection.horizontal_fill,
            "vertical_frame_fraction": projection.vertical_fill,
            "image_area_fraction": projection.image_fraction,
            "normalised_centre_error": projection.centre_error,
            "minimum_normalised_edge_margin": (
                projection.minimum_edge_margin),
            "overflow": projection.overflow,
            "fully_inside_92_percent_fov": (
                projection.fully_inside_margin),
        },
        "predicted_visible_sample_fraction": (
            selected.predicted_visible_point_fraction),
        "predicted_visibility_sample_count": 9,
        "plant_roi_vertical_bounds_m": [ROI_BOTTOM_Z, ROI_TOP_Z],
        "planning_inputs": {
            "predefined_crop_geometry": True,
            "rendered_image_feedback": False,
            "cyan_highlight_feedback": False,
        },
        "selected_viewpoint_clearance_m": selected.viewpoint_clearance_m,
        "score": selected.score,
        "score_components": selected.score_components,
        "score_weights": SCORE_WEIGHTS,
        "candidate_route_feasibility_planner": "structured",
        "score_definition": (
            "FOV overflow, centring error, target-size error, route length "
            "and oblique-azimuth penalty; lower is better"),
    }


def build_auto_preflight(
        selection_record, output_dir, position_ned, yaw_ned,
        target_id=0, transit_z=TRANSIT_Z, side_policy=None,
        route_planner="structured", views_per_side=1):
    """Optimise and write one executor-compatible automatic viewpoint."""
    source_entry = selected_target_entry(selection_record, target_id)
    live_pose = px4_ned_to_greenhouse_pose(position_ned, yaw_ned)
    plant_x = float(source_entry["target_greenhouse_map"][0])
    row_y = rows()[int(source_entry["row_index"])]
    requested_side_policy = (
        str(side_policy) if side_policy else
        str(selection_record.get("side_policy", "both")))
    if route_planner not in ("structured", "astar"):
        raise ValueError("route planner must be structured or astar")
    auto_results = plan_requested_sides(
        static_geometry(),
        live_pose,
        plant_x,
        row_y,
        float(transit_z),
        requested_side_policy,
        views_per_side,
    )
    target_dir = Path(output_dir) / (
        "target_%03d" % source_entry["target_id"])
    def result_name(auto_result):
        suffix = ""
        if int(views_per_side) == 2:
            suffix = "_left" if auto_result.selected.azimuth_offset_deg < 0 else "_right"
        return "auto_%s%s" % (auto_result.selected.side, suffix)

    shots = tuple(
        as_side_shot(auto_result, name=result_name(auto_result))
        for auto_result in auto_results)
    identity = np.eye(4)
    manifest = write_side_shot_artifacts(
        shots,
        str(target_dir),
        selection_record["map_path"],
        live_pose,
        identity,
        (plant_x, row_y, 2.0),
    )
    metadata_by_name = {
        result_name(auto_result): auto_view_metadata(
            auto_result, source_entry)
        for auto_result in auto_results
    }
    shot_records = manifest["shots"]
    result = {
        "schema": "gps_greenhouse_auto_preflight_set/v1",
        "source_schema": selection_record["schema"],
        "map_path": selection_record["map_path"],
        "live_pose_map": list(live_pose),
        "start_pose_saved_map": list(live_pose),
        "dry_run": True,
        "flight_command_capable": False,
        "flight_commanded": False,
        "selection_mode": selection_record.get("selection_mode", "plant"),
        "side_policy": requested_side_policy,
        "views_per_side": int(views_per_side),
        "route_planner": route_planner,
        "target_count": 1,
        "targets": [{
            "target_id": source_entry["target_id"],
            "source_target_greenhouse_map": source_entry[
                "target_greenhouse_map"],
            "target_saved_map": [plant_x, row_y, 2.0],
            "operator_cell": source_entry.get("operator_cell"),
            "directory": str(target_dir),
            "ready_shots": sum(
                shot_record["status"] == "PREFLIGHT_READY"
                for shot_record in shot_records),
            "shots": shot_records,
        }],
    }
    annotate_gps_preflights(result, output_dir)
    for shot_record in shot_records:
        metadata = metadata_by_name[shot_record["name"]]
        preflight_path = Path(shot_record["preflight_json"])
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        metadata["route_planner"] = route_planner
        if route_planner == "astar":
            start_pose = np.asarray(
                preflight["pose_source"]["vehicle_pose_saved_xyzyaw"],
                dtype=float)
            goal_pose = np.asarray(
                preflight["selected"]["vehicle_pose_map"], dtype=float)
            goal_pose[3] = math.radians(goal_pose[3])
            astar = plan_astar_route(
                start_pose,
                goal_pose,
                collision_boxes_from_geometry(static_geometry()),
            )
            route_path = Path(preflight["route"]["route_file"])
            np.save(route_path, astar.route)
            preflight["route"].update({
                "waypoints_saved_map": astar.route.tolist(),
                "waypoint_count_including_start": len(astar.route),
                "inserted_waypoints": max(0, len(astar.route) - 2),
                "length_m": route_length(astar.route),
                "all_segments_collision_free": True,
                "detail": "2.5D occupancy-grid A*; exact box revalidation",
            })
            metadata["execution_route_planning_time_ms"] = (
                astar.planning_time_ms)
            metadata["planner_time_ms"] += astar.planning_time_ms
            metadata["astar"] = {
                "grid_resolution_m": astar.grid_resolution_m,
                "expanded_nodes": astar.expanded_nodes,
                "raw_grid_nodes": astar.raw_grid_nodes,
                "simplified_grid_nodes": astar.simplified_grid_nodes,
            }
        preflight["auto_viewpoint"] = metadata
        preflight_path.write_text(
            json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    result["auto_viewpoints"] = [
        metadata_by_name[shot_record["name"]]
        for shot_record in shot_records
    ]
    manifest_path = Path(output_dir) / "live_preflight_manifest.json"
    manifest_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def build_auto_preflight_batch(
        selection_record, output_dir, position_ned, yaw_ned,
        target_id=0, transit_z=TRANSIT_Z, side_policy=None,
        route_planner="structured", views_per_side=1):
    """Plan every selected crop from one consistent PX4 pose snapshot."""
    load_selection(selection_record, target_id)
    target_ids = [
        int(entry["target_id"])
        for entry in selection_record.get("targets", [])
        if target_id in (0, int(entry["target_id"]))
    ]
    if not target_ids:
        raise ValueError("selection contains no requested crop targets")
    results = [
        build_auto_preflight(
            selection_record,
            output_dir,
            position_ned,
            yaw_ned,
            identifier,
            transit_z,
            side_policy,
            route_planner,
            views_per_side,
        )
        for identifier in target_ids
    ]
    aggregate = {
        "schema": "gps_greenhouse_auto_preflight_set/v2",
        "source_schema": selection_record["schema"],
        "map_path": selection_record["map_path"],
        "live_pose_map": results[0]["live_pose_map"],
        "start_pose_saved_map": results[0]["start_pose_saved_map"],
        "dry_run": True,
        "flight_command_capable": False,
        "flight_commanded": False,
        "selection_mode": selection_record.get("selection_mode", "plant"),
        "side_policy": (
            str(side_policy) if side_policy else
            str(selection_record.get("side_policy", "both"))),
        "views_per_side": int(views_per_side),
        "route_planner": route_planner,
        "target_count": len(results),
        "target_order": target_ids,
        "targets": [target for result in results
                    for target in result["targets"]],
        "auto_viewpoints": [viewpoint for result in results
                            for viewpoint in result["auto_viewpoints"]],
    }
    manifest_path = Path(output_dir) / "live_preflight_manifest.json"
    manifest_path.write_text(
        json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    return aggregate


def main(args=None):
    """Wait for PX4 odometry, optimise once, write artifacts, and exit."""
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

    class GpsAutoPreflight(Node):
        def __init__(self):
            super().__init__("gps_auto_viewpoint_preflight")
            default_root = Path(os.environ.get(
                "THESIS_DIR", str(Path.cwd())))
            selection_path = str(self.declare_parameter(
                "selection_json",
                str(default_root / "gps_viewpoint_plan/selected_targets.json")
                ).value)
            self.output_dir = str(self.declare_parameter(
                "output_dir",
                str(default_root / "gps_auto_viewpoint_preflight")).value)
            self.target_id = int(self.declare_parameter(
                "target_id", 0).value)
            self.transit_z = float(self.declare_parameter(
                "transit_z", TRANSIT_Z).value)
            self.side_policy = str(self.declare_parameter(
                "side_policy", "").value)
            self.route_planner = str(self.declare_parameter(
                "route_planner", "structured").value)
            self.views_per_side = int(self.declare_parameter(
                "views_per_side", 1).value)
            self.timeout_sec = float(self.declare_parameter(
                "timeout_sec", 90.0).value)
            self.selection_record = json.loads(
                Path(selection_path).read_text(encoding="utf-8"))
            load_selection(self.selection_record, self.target_id)
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
                "AUTO GPS PREFLIGHT ONLY: sampling distance, pitch, yaw, "
                "aim height and both plant sides. This node cannot arm or "
                "publish PX4 flight commands.")

        def odometry_callback(self, message):
            if self.completed:
                return
            self.completed = True
            self.timer.cancel()
            try:
                result = build_auto_preflight_batch(
                    self.selection_record,
                    self.output_dir,
                    np.asarray(message.position, dtype=float),
                    px4_quaternion_yaw(tuple(message.q)),
                    self.target_id,
                    self.transit_z,
                    self.side_policy or None,
                    self.route_planner,
                    self.views_per_side,
                )
            except Exception as error:
                self.get_logger().error(
                    "Automatic viewpoint planning failed: %s" % error)
                raise
            for metadata in result["auto_viewpoints"]:
                self.get_logger().info(
                    "AUTO_SIDE_READY: side=%s, distance=%.2f m, "
                    "pitch=%.1f deg, azimuth=%.1f deg, camera_z=%.2f m; "
                    "%d/%d candidates physically feasible and %d fully "
                    "framed." % (
                        metadata["selected_side"],
                        metadata["selected_distance_m"],
                        metadata["selected_pitch_deg"],
                        metadata["selected_azimuth_deg"],
                        metadata["selected_camera_height_m"],
                        metadata["physically_feasible_candidates"],
                        metadata["attempted_candidates"],
                        metadata["fully_framed_candidates"],
                    ))
            self.get_logger().info(
                "AUTO_PREFLIGHT_READY: %d crop target(s), %d independently "
                "optimised shot(s) are ready; no flight was commanded."
                % (result["target_count"], len(result["auto_viewpoints"])))

        def check_timeout(self):
            elapsed = (
                self.get_clock().now() - self.started).nanoseconds / 1e9
            if elapsed > self.timeout_sec:
                self.get_logger().error(
                    "Timed out waiting for /fmu/out/vehicle_odometry")
                self.completed = True
                self.timer.cancel()

    rclpy.init(args=args)
    node = GpsAutoPreflight()
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
