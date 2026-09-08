"""Convert offline PNG clicks into fresh, live preflights."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

from .free_space import TRANSIT_Z
from .side_shot_inspection import (
    plan_side_shots,
    write_side_shot_artifacts,
)


SELECTION_SCHEMA = "greenhouse_png_target_selection/v1"
PREFLIGHT_SET_SCHEMA = "greenhouse_png_live_preflight_set/v1"


@dataclass(frozen=True)
class PngSelection:
    """Validated operator intent loaded from the offline selector."""

    map_path: str
    distance: float
    pitch_deg: float
    targets: tuple[tuple[int, tuple[float, float, float]], ...]


def load_png_selection(
        record: dict, target_id: int = 0) -> PngSelection:
    """Validate a non-executable PNG selection and return its target intent."""
    required = {
        "schema": SELECTION_SCHEMA,
        "mode": "offline_saved_map_only",
        "flight_command_capable": False,
        "flight_commanded": False,
        "executable_by_viewpoint_execution": False,
    }
    for key, expected in required.items():
        if record.get(key) != expected:
            raise ValueError(
                f"selection {key!r} must be {expected!r}, got "
                f"{record.get(key)!r}")
    map_path = record.get("map_path")
    if not isinstance(map_path, str) or not map_path:
        raise ValueError("selection map_path is missing")
    distance = float(record.get("distance_m", float("nan")))
    pitch = float(record.get("pitch_deg", float("nan")))
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("selection distance_m must be positive and finite")
    if not math.isfinite(pitch):
        raise ValueError("selection pitch_deg must be finite")
    entries = record.get("targets")
    if not isinstance(entries, list) or not entries:
        raise ValueError("selection contains no targets")
    if record.get("target_count") != len(entries):
        raise ValueError("selection target_count does not match targets")
    targets = []
    seen = set()
    for entry in entries:
        identifier = (
            entry.get("target_id") if isinstance(entry, dict) else None)
        values = (
            entry.get("target_saved_map")
            if isinstance(entry, dict) else None)
        if (not isinstance(identifier, int) or identifier <= 0
                or identifier in seen):
            raise ValueError(
                "selection target IDs must be unique positive ints")
        target = np.asarray(values, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError(
                f"target {identifier} must contain finite saved-map x,y,z")
        seen.add(identifier)
        if target_id in (0, identifier):
            targets.append((
                identifier, tuple(float(value) for value in target)))
    if target_id > 0 and not targets:
        raise ValueError(f"selection does not contain target_id {target_id}")
    return PngSelection(map_path, distance, pitch, tuple(targets))


def build_preflight_set(
        selection: PngSelection, map_geometry, output_dir: str,
        live_pose, saved_from_live: np.ndarray,
        transit_z: float = TRANSIT_Z,
        side_targets_by_id: dict | None = None) -> dict:
    """Replan selected targets and write live Phase 6a artifacts."""
    from .viewpoint_inspection import pose_in_saved_map

    start_saved = pose_in_saved_map(live_pose, saved_from_live)
    entries = []
    for identifier, target in selection.targets:
        target_dir = Path(output_dir) / f"target_{identifier:03d}"
        plans = plan_side_shots(
            map_geometry, start_saved, target,
            distance=selection.distance,
            pitch_deg=selection.pitch_deg,
            transit_z=float(transit_z),
            side_targets=(side_targets_by_id or {}).get(identifier))
        manifest = write_side_shot_artifacts(
            plans, str(target_dir), selection.map_path, live_pose,
            saved_from_live, target)
        entries.append({
            "target_id": identifier,
            "target_saved_map": list(target),
            "directory": str(target_dir),
            "ready_shots": sum(
                shot["status"] == "PREFLIGHT_READY"
                for shot in manifest["shots"]),
            "shots": manifest["shots"],
        })
    result = {
        "schema": PREFLIGHT_SET_SCHEMA,
        "source_schema": SELECTION_SCHEMA,
        "map_path": selection.map_path,
        "live_pose_map": [float(value) for value in live_pose],
        "start_pose_saved_map": [float(value) for value in start_saved],
        "dry_run": True,
        "flight_command_capable": False,
        "flight_commanded": False,
        "target_count": len(entries),
        "targets": entries,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "live_preflight_manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(args=None):
    """Wait for live state, write fresh plans once, and exit."""
    from nav_msgs.msg import Odometry
    import rclpy
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener
    from tf2_ros import (
        ConnectivityException,
        ExtrapolationException,
        LookupException,
    )

    from .map_geometry import geometry
    from .viewpoint_inspection import quaternion_yaw, transform_from_tf

    class PngTargetPreflight(Node):
        def __init__(self):
            super().__init__("png_target_preflight")
            selection_path = str(self.declare_parameter(
                "selection_json",
                "/root/masters-thesis/png_target_plan/"
                "selected_targets.json").value)
            self.output_dir = str(self.declare_parameter(
                "output_dir",
                "/root/masters-thesis/png_live_preflight").value)
            target_id = int(self.declare_parameter("target_id", 0).value)
            self.transit_z = float(self.declare_parameter(
                "transit_z", TRANSIT_Z).value)
            self.timeout = float(self.declare_parameter(
                "timeout_sec", 120.0).value)
            record = json.loads(
                Path(selection_path).read_text(encoding="utf-8"))
            self.selection = load_png_selection(record, target_id)
            cloud = np.load(self.selection.map_path)
            if (cloud.ndim != 2 or cloud.shape[1] != 3
                    or not np.isfinite(cloud).all()):
                raise ValueError("selected map must be a finite Nx3 cloud")
            self.map_geometry = geometry(cloud)
            self.live_pose: Optional[tuple[float, float, float, float]] = None
            self.completed = False
            self.started = self.get_clock().now()
            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.tf_errors = (
                LookupException, ConnectivityException,
                ExtrapolationException)
            qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST, depth=5)
            self.create_subscription(
                Odometry, "/odometry", self.odom_cb, qos)
            self.timer = self.create_timer(0.25, self.try_plan)
            self.get_logger().warn(
                "PNG LIVE PREFLIGHT: waiting for FAST-LIO /odometry and "
                "accepted saved_map -> map TF. This node has no PX4 "
                "publishers and cannot move the drone.")

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

        def try_plan(self):
            elapsed = (
                self.get_clock().now() - self.started).nanoseconds / 1e9
            if elapsed > self.timeout:
                self.get_logger().error(
                    "Timed out waiting for live odometry/relocalisation")
                self.completed = True
                self.timer.cancel()
                return
            if self.live_pose is None:
                return
            try:
                saved_from_live = self.current_transform()
            except self.tf_errors:
                return
            result = build_preflight_set(
                self.selection, self.map_geometry, self.output_dir,
                self.live_pose, saved_from_live, self.transit_z)
            ready = sum(
                entry["ready_shots"] for entry in result["targets"])
            self.get_logger().info(
                "PREFLIGHT_SET_READY: %d targets, %d side shots ready. "
                "Review %s/live_preflight_manifest.json. No flight "
                "commanded." % (
                    len(result["targets"]), ready, self.output_dir))
            self.completed = True
            self.timer.cancel()

    rclpy.init(args=args)
    node = PngTargetPreflight()
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
