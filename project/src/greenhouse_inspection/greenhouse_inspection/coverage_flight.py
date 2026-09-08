"""Fly greenhouse photo routes and save position-tagged images."""

import math
import os

import numpy as np

from greenhouse_inspection.assisted_teleop import (
    CONE_HALF_ANGLE_DEG,
    MAX_CHECK_RANGE,
    clamp_to_obstacle,
    clamp_vertical,
    yaw_from_quat,
)
from greenhouse_inspection.coverage_grid import generate_coverage_grid
from greenhouse_inspection.free_space import obstacles, safe_leg
from greenhouse_inspection.map_geometry import geometry
from greenhouse_inspection.row_coverage import (
    corridor_route,
    imaging_altitude,
    inspection_route,
    over_row_route,
    route_length,
    row_pass_legs,
    venlo_rows,
)

HOLD_SECONDS = 2.0   # Per-waypoint capture hold.
REACH_RADIUS = 0.3   # Waypoint arrival radius.
TICK_PERIOD = 0.1    # Control-loop period.

# Limit yaw changes at row-end reversals to reduce SLAM drift.
MAX_YAW_RATE = math.radians(20.0)


def ramped_yaw(current, target, dt=TICK_PERIOD, max_rate=MAX_YAW_RATE):
    """Move toward target yaw by the shortest rate-limited step."""
    if math.isnan(target):
        return target
    if math.isnan(current):
        return target
    diff = (target - current + math.pi) % (2.0 * math.pi) - math.pi
    step = max_rate * dt
    if abs(diff) <= step:
        return target
    return current + (step if diff > 0.0 else -step)

# Brake with LiDAR and try a verified sidestep when progress stops.
STUCK_TICKS = 30        # About three seconds at 10 Hz.
SIDESTEP_DISTANCE = 2.0  # One sidestep attempt in metres.

# Use the launch position as the dock and fly home above the canopy.
DOCK_TRANSIT_ALT = 4.5
DOCK_LAND_RADIUS = 0.5    # Begin descent inside this horizontal radius.
# Limit the final descent rate for a controlled pad approach.
MAX_LAND_DESCENT_RATE = 0.5  # m/s
RTB_BATTERY_PCT = 25.0    # Battery level that starts the return.
RESUME_BATTERY_PCT = 95.0  # Battery level that resumes the mission.
RTB_HYSTERESIS = 5.0      # Resume margin above the return threshold.
# Test-only simulated full-charge duration.
BATTERY_FREEZE_SECONDS = 480.0


def resume_threshold(trigger, default=RESUME_BATTERY_PCT,
                     hysteresis=RTB_HYSTERESIS):
    """Return a resume level safely above the return trigger."""
    return min(100.0, max(default, trigger + hysteresis))


def battery_needs_return(percentage, threshold=RTB_BATTERY_PCT):
    """Return whether a valid percentage is below the return threshold."""
    if percentage is None or percentage <= 1.0 or percentage > 100.0:
        return False
    return percentage < threshold


def sidestep_scale(dx_b, dy_b, dz_b, points_flu, current_altitude):
    """Check horizontal LiDAR clearance or vertical landing-altitude clearance."""
    if dz_b != 0.0:
        _, scale = clamp_vertical(dz_b, current_altitude + dz_b)
    else:
        _, _, _, scale = clamp_to_obstacle(dx_b, dy_b, dz_b, points_flu)
    return scale


def is_stuck(scale, distance, prev_distance, reach_radius=REACH_RADIUS):
    """Return whether obstacle braking has stopped route progress."""
    return scale < 1.0 and distance > reach_radius and distance > prev_distance - 0.02


def _ned_to_body_flu(n, e, d, yaw):
    """Convert a NED displacement to body-FLU coordinates."""
    x_frd = math.cos(yaw) * n + math.sin(yaw) * e
    y_frd = -math.sin(yaw) * n + math.cos(yaw) * e
    return x_frd, -y_frd, -d


def _body_flu_to_ned(vx, vy, vz, yaw):
    """Convert a body-FLU displacement to NED coordinates."""
    x_frd, y_frd, z_frd = vx, -vy, -vz
    n = math.cos(yaw) * x_frd - math.sin(yaw) * y_frd
    e = math.sin(yaw) * x_frd + math.cos(yaw) * y_frd
    return n, e, z_frd


def nearest_in_cone(dx_b, dy_b, dz_b, points_flu):
    """Return the nearest LiDAR point in the obstacle-braking cone."""
    speed = math.sqrt(dx_b * dx_b + dy_b * dy_b + dz_b * dz_b)
    if speed < 1e-3 or points_flu.shape[0] == 0:
        return None
    direction = np.array([dx_b, dy_b, dz_b]) / speed
    dists = np.linalg.norm(points_flu, axis=1)
    in_range = dists < MAX_CHECK_RANGE
    if not np.any(in_range):
        return None
    pts, dists = points_flu[in_range], dists[in_range]
    cos_angle = (pts @ direction) / dists
    in_cone = cos_angle > math.cos(math.radians(CONE_HALF_ANGLE_DEG))
    if not np.any(in_cone):
        return None
    i = np.argmin(dists[in_cone])
    return tuple(pts[in_cone][i]), float(dists[in_cone][i])


def image_to_array(msg):
    """sensor_msgs/Image (rgb8, no row padding) -> (H, W, 3) uint8 array."""
    if msg.encoding not in ("rgb8", "R8G8B8"):
        raise ValueError(f"expected rgb8, got encoding {msg.encoding!r}")
    expected = msg.height * msg.width * 3
    if len(msg.data) != expected:
        raise ValueError(f"image data is {len(msg.data)} bytes, expected {expected}")
    arr = np.frombuffer(msg.data, dtype=np.uint8)
    return arr.reshape((msg.height, msg.width, 3))


def save_image(msg, path):
    from PIL import Image
    Image.fromarray(image_to_array(msg)).save(path)


def map_to_px4_waypoints(grid):
    """coverage_grid points are in the map frame."""
    return [(y, x, -alt) for (x, y, alt) in grid]


def route_to_px4_waypoints(route):
    """Same remap for a row_coverage route, which also carries a heading."""
    return [(y, x, -alt, math.pi / 2 - yaw) for (x, y, alt, yaw) in route]


def main(args=None):
    import rclpy
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from sensor_msgs.msg import BatteryState, Image, PointCloud2
    from std_msgs.msg import Bool
    from sensor_msgs_py import point_cloud2 as pc2

    from px4_msgs.msg import (
        OffboardControlMode,
        TrajectorySetpoint,
        VehicleCommand,
        VehicleOdometry,
        VehicleStatus,
    )

    class CoverageFlight(Node):
        def __init__(self):
            super().__init__("coverage_flight")

            map_path = self.declare_parameter(
                "map_path", "/root/masters-thesis/teach_map.npy").value
            altitude = self.declare_parameter("altitude", 1.2).value
            out_dir = self.declare_parameter(
                "out_dir", "/root/masters-thesis/coverage_photos").value

            # Positive spacing enables continuous survey capture.
            self.capture_spacing = float(
                self.declare_parameter("capture_spacing", 0.0).value)
            self.last_capture_pos = None
            self.shot_count = 0
            # ``None`` captures every leg; row routes mark imaging legs.
            self.capture_legs = None

            # Default return path geometry for the simulated greenhouse.
            self.rtb_rows = venlo_rows()
            # Map-derived routes replace this with sensed obstacle boxes.
            self.rtb_boxes = None

            route_name = self.declare_parameter("route", "").value
            if route_name == "overrow":
                route = over_row_route()
                self.waypoints = route_to_px4_waypoints(route)
                self.capture_legs = row_pass_legs(route)
                n_rows = len({round(p[1], 2) for p, shooting
                             in zip(route, self.capture_legs) if shooting})
                self.get_logger().info(
                    f"Over-row imaging route: {len(self.waypoints)} waypoints "
                    f"({sum(self.capture_legs)} shooting), "
                    f"{route_length(route):.0f} m above {n_rows} rows at "
                    f"{max(p[2] for p in route):.2f}m.")
            elif route_name == "venlo":
                route = inspection_route(altitude=altitude)
                self.waypoints = route_to_px4_waypoints(route)
                self.get_logger().info(
                    f"Venlo aisle route: {len(self.waypoints)} waypoints over "
                    f"{len({round(p[1], 3) for p in route})} passes, "
                    f"{route_length(route):.0f} m at {altitude:.1f}m. "
                    "Aisles blocked by columns are skipped.")
            elif route_name == "mappedrow":
                # Build this route from the recorded map geometry.
                geo = geometry(np.load(map_path))
                extents = {round(y, 3): ext
                           for y, ext in zip(geo.rows, geo.extents)}
                alt = imaging_altitude(canopy_top=geo.canopy_top)
                self.rtb_rows = geo.rows
                self.rtb_boxes = geo.boxes

                # Select one sorted row, or fly every row by default.
                row_only = int(self.declare_parameter("row_only", -1).value)
                # Exclude one sorted row without changing other pass directions.
                skip_row = int(self.declare_parameter("skip_row", -1).value)
                rows = geo.rows
                if row_only >= 0:
                    target_y = sorted(geo.rows)[row_only]
                    rows = [target_y]
                    extents = {round(target_y, 3): extents[round(target_y, 3)]}
                elif skip_row >= 0:
                    skip_y = round(sorted(geo.rows)[skip_row], 3)
                    rows = [y for y in geo.rows if round(y, 3) != skip_y]

                route = over_row_route(rows=rows, altitude=alt,
                                       extents=extents, boxes=geo.boxes)

                # Crop a complete route while preserving pass directions.
                wp_start = int(self.declare_parameter("wp_start", -1).value)
                wp_end = int(self.declare_parameter("wp_end", -1).value)
                if wp_start >= 0:
                    end = wp_end if wp_end >= 0 else len(route) - 1
                    window = route[wp_start:end + 1]
                    transit = safe_leg((0.0, 0.0, alt), window[0][:3], geo.boxes)
                    if transit is None:
                        raise ValueError(
                            f"no safe transit from spawn to waypoint {wp_start}")
                    route = [t + (window[0][3],) for t in transit] + window

                self.waypoints = route_to_px4_waypoints(route)
                self.capture_legs = row_pass_legs(route, rows=rows)
                self.get_logger().info(
                    f"Map-derived route from {map_path}: "
                    f"{len(self.waypoints)} waypoints over {len(rows)} "
                    f"row(s), canopy_top={geo.canopy_top:.2f}m (ceiling, not "
                    f"measured), alt={alt:.2f}m, "
                    f"{route_length(route):.0f}m.")
            elif route_name == "corridor":
                target = self.declare_parameter("target_distance", 345.0).value
                route = corridor_route(altitude=altitude, target_distance=target)
                self.waypoints = route_to_px4_waypoints(route)
                self.get_logger().info(
                    f"Corridor route: {len(self.waypoints)} waypoints, "
                    f"{route_length(route):.0f} m at {altitude:.1f}m "
                    f"(target {target:.0f}m), 2.85m clear width.")
            else:
                points = np.load(map_path)
                x_min, x_max = float(points[:, 0].min()), float(points[:, 0].max())
                y_min, y_max = float(points[:, 1].min()), float(points[:, 1].max())
                grid = generate_coverage_grid(x_min, x_max, y_min, y_max, altitude)
                self.waypoints = map_to_px4_waypoints(grid)
                self.get_logger().info(
                    f"Map extent x=[{x_min:.1f},{x_max:.1f}] "
                    f"y=[{y_min:.1f},{y_max:.1f}]; "
                    f"{len(self.waypoints)} coverage shots planned at "
                    f"{altitude:.1f}m.")

            os.makedirs(out_dir, exist_ok=True)
            self.out_dir = out_dir

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
            cam_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )

            self.offboard_pub = self.create_publisher(
                OffboardControlMode, "/fmu/in/offboard_control_mode", pub_qos)
            self.setpoint_pub = self.create_publisher(
                TrajectorySetpoint, "/fmu/in/trajectory_setpoint", pub_qos)
            self.command_pub = self.create_publisher(
                VehicleCommand, "/fmu/in/vehicle_command", pub_qos)

            self.create_subscription(
                VehicleOdometry, "/fmu/out/vehicle_odometry", self.odom_cb, sub_qos)
            # PX4 v1.17 publishes the status topic at the generated versioned name below.
            self.create_subscription(
                VehicleStatus, "/fmu/out/vehicle_status_v1", self.status_cb, sub_qos)
            self.create_subscription(Image, "/camera", self.image_cb, cam_qos)
            self.create_subscription(
                PointCloud2, "/lidar/points", self.lidar_cb, sub_qos)
            # Battery input is optional; no reading disables return-to-base.
            self.create_subscription(
                BatteryState, "/battery/state", self.battery_cb, sub_qos)
            self.recharge_pub = self.create_publisher(
                Bool, "/battery/recharge_start", 10)

            self.rtb_enabled = self.declare_parameter("return_to_base", True).value
            self.rtb_threshold = float(
                self.declare_parameter("rtb_battery_pct", RTB_BATTERY_PCT).value)
            self.rtb_resume = resume_threshold(self.rtb_threshold)
            # Test mode resumes immediately because simulated charging is unavailable.
            self.instant_full_charge = bool(
                self.declare_parameter("instant_full_charge_on_dock", False).value)
            # Use a transit height no lower than any mission waypoint.
            route_alt = max((-wp[2] for wp in self.waypoints), default=0.0)
            self.rtb_transit_alt = max(DOCK_TRANSIT_ALT, route_alt)
            self.battery_pct = None
            self.battery_freeze_until = None  # Simulated full-charge timeout.
            self.returning = False    # Mission is returning to dock.
            self.charging = False     # Vehicle is charging on the dock.
            self.saved_index = None   # Waypoint to resume.
            self.rtb_path = []        # Safe legs remaining to the dock.
            self.home_ned = None      # Captured from first odometry.

            self.position = None
            self.current_yaw = 0.0
            self.latest_points = np.zeros((0, 3), dtype=np.float32)
            self.stuck_ticks = 0
            self.prev_distance = float("inf")
            self.commanded_yaw = float("nan")  # ramped_yaw's running state
            self.detour_target = None
            self.latest_image = None
            self.index = 0
            self.holding = False
            self.hold_start = None
            self.captured_this_hold = False
            self.done = False
            self.landing = False

            self.armed = False
            self.nav_state = None
            self.tick = 0
            self.last_engage_tick = None

            self.timer = self.create_timer(0.1, self.timer_callback)

        # --- subscriptions ---------------------------------------------
        def odom_cb(self, msg):
            self.position = np.array(msg.position, dtype=float)
            self.current_yaw = yaw_from_quat(tuple(msg.q))
            if self.home_ned is None:
                # The launch position defines the dock's horizontal location.
                self.home_ned = self.position[:2].copy()
                self.get_logger().info(
                    f"Dock/home captured at launch: NED "
                    f"({self.home_ned[0]:.2f}, {self.home_ned[1]:.2f}).")

        def lidar_cb(self, msg):
            # Extra cloud fields require ``read_points`` rather than NumPy decoding.
            pts = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            self.latest_points = np.column_stack(
                [pts["x"], pts["y"], pts["z"]]).astype(np.float32)

        def status_cb(self, msg):
            self.armed = msg.arming_state == VehicleStatus.ARMING_STATE_ARMED
            self.nav_state = msg.nav_state

        def image_cb(self, msg):
            self.latest_image = msg

        def battery_cb(self, msg):
            if self.battery_freeze_until is not None:
                # Hold the simulated reading until the test charge expires.
                if self.get_clock().now() < self.battery_freeze_until:
                    return
                self.battery_freeze_until = None
            self.battery_pct = float(msg.percentage)

        # --- return to base ----------------------------------------------
        def dock_waypoint(self, altitude):
            """Return the launch-position dock in PX4 NED coordinates."""
            return np.array([self.home_ned[0], self.home_ned[1], -altitude])

        def _map_to_ned(self, x, y, z):
            """Same swap as dock_waypoint -- map(x,y,alt) -> NED(north,east,down)."""
            return np.array([y, x, -z])

        def begin_return(self):
            """Plan an obstacle-checked path from the vehicle to the dock."""
            map_x, map_y = float(self.position[1]), float(self.position[0])
            map_alt = -float(self.position[2])
            # free_space works in the map frame, home_ned is NED -- swap
            # back the same way _map_to_ned swaps forward.
            dock = (float(self.home_ned[1]), float(self.home_ned[0]),
                    self.rtb_transit_alt)
            boxes = (self.rtb_boxes if self.rtb_boxes is not None
                     else obstacles(self.rtb_rows))
            mids = safe_leg((map_x, map_y, map_alt), dock, boxes)
            if mids is None:
                self.get_logger().error(
                    "No safe path to the dock from here -- holding mission "
                    "course rather than forcing a straight line through an "
                    "obstacle.", throttle_duration_sec=5.0)
                return

            self.returning = True
            self.saved_index = self.index
            self.rtb_path = [self._map_to_ned(x, y, z) for x, y, z in mids]
            self.rtb_path.append(self.dock_waypoint(self.rtb_transit_alt))
            self.get_logger().warn(
                f"Battery {self.battery_pct:.0f}% (below {self.rtb_threshold:.0f}%) "
                f"-- returning to dock via a {len(self.rtb_path)}-leg safe "
                f"path, will resume at waypoint {self.index}.")

        def resume_mission(self):
            self.returning = False
            self.charging = False
            self.rtb_path = []
            self.index = self.saved_index if self.saved_index is not None else self.index
            self.saved_index = None
            self.get_logger().info(
                f"Charged to {self.battery_pct:.0f}% -- resuming mission at "
                f"waypoint {self.index}.")

        def run_return_to_base(self):
            """Follow the safe path from begin_return, then descend onto the dock and hold while it charges."""
            if self.charging:
                # Sit on the pad. The recharge trigger is edge-published in
                # the transition below, not spammed every tick.
                self.publish_setpoint(self.dock_waypoint(0.0), float("nan"))
                if (self.battery_pct is not None
                        and self.battery_pct >= self.rtb_resume):
                    self.resume_mission()
                return

            if self.rtb_path:
                # Working through the obstacle-checked path.
                target = self.rtb_path[0]
                if float(np.linalg.norm(self.position - target)) < REACH_RADIUS:
                    self.rtb_path.pop(0)
                self.publish_setpoint(target, float("nan"))
                return

            # Path consumed -- over the dock's transit point, clear of the crop.
            altitude = -float(self.position[2])
            over_dock = self.dock_waypoint(self.rtb_transit_alt)
            horizontal = float(np.linalg.norm(self.position[:2] - over_dock[:2]))
            if horizontal > DOCK_LAND_RADIUS:
                self.publish_setpoint(over_dock, float("nan"))
                return

            pad = self.dock_waypoint(0.0)
            max_step = MAX_LAND_DESCENT_RATE * TICK_PERIOD
            descend_by = min(pad[2] - self.position[2], max_step)
            setpoint = np.array([pad[0], pad[1], self.position[2] + descend_by])
            self.publish_setpoint(setpoint, float("nan"))
            if altitude < 0.25:
                if self.instant_full_charge:
                    # Resume directly so incoming battery readings cannot override it.
                    self.battery_pct = 100.0
                    self.battery_freeze_until = (
                        self.get_clock().now()
                        + Duration(seconds=BATTERY_FREEZE_SECONDS))
                    self.get_logger().info(
                        "Landed on dock -- instant_full_charge_on_dock set, "
                        "resuming immediately.")
                    self.resume_mission()
                    return
                self.charging = True
                self.recharge_pub.publish(Bool(data=True))
                self.get_logger().info(
                    "Landed on dock, recharging. Mission resumes at "
                    f"{self.rtb_resume:.0f}%.")

        # --- publishing --------------------------------------------------
        def stamp(self):
            return int(self.get_clock().now().nanoseconds / 1000)

        def publish_offboard_mode(self):
            msg = OffboardControlMode()
            msg.timestamp = self.stamp()
            msg.position = True
            self.offboard_pub.publish(msg)

        def publish_setpoint(self, xyz, yaw):
            msg = TrajectorySetpoint()
            msg.timestamp = self.stamp()
            msg.position = [float(xyz[0]), float(xyz[1]), float(xyz[2])]
            msg.yaw = float(yaw)
            self.setpoint_pub.publish(msg)

        def publish_command(self, command, **params):
            msg = VehicleCommand()
            msg.timestamp = self.stamp()
            msg.command = command
            msg.param1 = float(params.get("param1", 0.0))
            msg.param2 = float(params.get("param2", 0.0))
            msg.target_system = 1
            msg.target_component = 1
            msg.source_system = 1
            msg.source_component = 1
            msg.from_external = True
            self.command_pub.publish(msg)

        def engage_watchdog(self):
            """Retry offboard engagement until vehicle status confirms it."""
            ready = (self.armed
                     and self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD)
            can_attempt = (self.last_engage_tick is None
                           or self.tick - self.last_engage_tick >= 10)
            if self.tick >= 20 and not ready and can_attempt:
                self.publish_command(
                    VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
                self.publish_command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
                self.last_engage_tick = self.tick
                self.get_logger().info(
                    f"Offboard + arm (re-)sent (armed={self.armed}, "
                    f"nav_state={self.nav_state}).")

        def land(self, reason):
            self.landing = True
            self.timer.cancel()
            self.publish_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
            self.get_logger().info(f"Landing: {reason}")

        # --- capture -------------------------------------------------------
        def capture(self):
            if self.latest_image is None:
                self.get_logger().warn(
                    "No camera frame available yet -- skipping this shot.")
                return
            # Use position and count because survey shots fall between waypoints.
            x, y, z = (self.position if self.position is not None
                       else np.array(self.waypoints[self.index][:3]))
            path = os.path.join(
                self.out_dir,
                f"shot_{self.shot_count:05d}_x{x:.2f}_y{y:.2f}_z{z:.2f}.jpg")
            try:
                save_image(self.latest_image, path)
                self.shot_count += 1
                self.get_logger().info(f"Saved {path}")
            except ValueError as e:
                self.get_logger().warn(f"Bad camera frame, skipping shot: {e}")

        # --- live obstacle reaction -----------------------------------------
        def find_sidestep(self):
            """Return a LiDAR-checked horizontal or vertical sidestep in NED."""
            for dx_b, dy_b, dz_b in ((0.0, SIDESTEP_DISTANCE, 0.0),
                                     (0.0, -SIDESTEP_DISTANCE, 0.0),
                                     (0.0, 0.0, SIDESTEP_DISTANCE)):
                scale = sidestep_scale(dx_b, dy_b, dz_b, self.latest_points,
                                       -self.position[2])
                if scale >= 0.99:
                    n, e, d = _body_flu_to_ned(dx_b, dy_b, dz_b, self.current_yaw)
                    return self.position + np.array([n, e, d])
            return None

        # --- main loop ------------------------------------------------------
        def timer_callback(self):
            if self.landing:
                return
            self.publish_offboard_mode()
            if self.position is None:
                return

            if self.done:
                self.engage_watchdog()
                self.tick += 1
                return

            # Return-to-base takes priority over the mission but sits below the obstacle clamp.
            if self.rtb_enabled:
                if self.returning:
                    self.run_return_to_base()
                    self.tick += 1
                    return
                if (self.home_ned is not None
                        and battery_needs_return(self.battery_pct,
                                                 self.rtb_threshold)):
                    self.begin_return()
                    if self.returning:
                        # begin_return can refuse and leave self.returning False.
                        self.run_return_to_base()
                        self.tick += 1
                        return

            wp = self.waypoints[self.index]
            target, yaw = wp[:3], (wp[3] if len(wp) > 3 else float("nan"))
            distance = float(np.linalg.norm(self.position - np.array(target)))
            # Rate-limited, not the raw waypoint yaw -- see MAX_YAW_RATE.
            self.commanded_yaw = ramped_yaw(self.commanded_yaw, yaw)
            yaw = self.commanded_yaw

            if self.detour_target is not None:
                # Mid-sidestep: go there first, ignoring the real target
                # until it's reached, then resume toward it next tick.
                setpoint = self.detour_target
                if float(np.linalg.norm(self.position - setpoint)) < REACH_RADIUS:
                    self.detour_target = None
                self.publish_setpoint(setpoint, yaw)
            else:
                delta = np.array(target) - self.position
                dx, dy, dz = _ned_to_body_flu(delta[0], delta[1], delta[2],
                                              self.current_yaw)
                dx, dy, dz, scale = clamp_to_obstacle(dx, dy, dz, self.latest_points)
                dz, v_scale = clamp_vertical(dz, -self.position[2])
                scale = min(scale, v_scale)
                if scale < 1.0:
                    hit = nearest_in_cone(dx, dy, dz, self.latest_points)
                    where = (f", nearest point body-FLU {hit[0]}, "
                            f"{hit[1]:.2f}m away, vehicle NED "
                            f"{tuple(round(v, 2) for v in self.position)}"
                            if hit else "")
                    self.get_logger().warn(
                        f"Obstacle ahead -- braking ({scale * 100:.0f}% speed)"
                        f"{where}.",
                        throttle_duration_sec=1.0)
                if is_stuck(scale, distance, self.prev_distance):
                    self.stuck_ticks += 1
                else:
                    self.stuck_ticks = 0
                self.prev_distance = distance
                if self.stuck_ticks > STUCK_TICKS:
                    self.detour_target = self.find_sidestep()
                    self.stuck_ticks = 0
                    if self.detour_target is None:
                        self.get_logger().error(
                            "Blocked, left, right and up -- holding rather "
                            "than forcing through.", throttle_duration_sec=2.0)
                    setpoint = self.position
                else:
                    n, e, d = _body_flu_to_ned(dx, dy, dz, self.current_yaw)
                    setpoint = self.position + np.array([n, e, d])
                self.publish_setpoint(setpoint, yaw)

            if self.capture_spacing > 0:
                # survey mode: never stop, shoot on distance travelled -- and
                # only while this leg is actually over a row.
                shooting = (self.capture_legs is None
                            or self.capture_legs[self.index])
                if shooting and (self.last_capture_pos is None
                                 or float(np.linalg.norm(
                                     self.position - self.last_capture_pos))
                                 >= self.capture_spacing):
                    self.capture()
                    self.last_capture_pos = self.position.copy()
                if distance < REACH_RADIUS:
                    if self.index < len(self.waypoints) - 1:
                        self.index += 1
                    else:
                        self.done = True
                        self.get_logger().info(
                            "Route complete. Holding position -- Ctrl-C to land.")
            elif not self.holding:
                if distance < REACH_RADIUS:
                    self.holding = True
                    self.hold_start = self.get_clock().now()
                    self.captured_this_hold = False
            else:
                elapsed = (self.get_clock().now() - self.hold_start).nanoseconds / 1e9
                # Capture partway through the hold, not the instant of arrival.
                if not self.captured_this_hold and elapsed > HOLD_SECONDS / 2:
                    self.capture()
                    self.captured_this_hold = True
                if elapsed >= HOLD_SECONDS:
                    self.holding = False
                    if self.index < len(self.waypoints) - 1:
                        self.index += 1
                    else:
                        self.done = True
                        self.get_logger().info(
                            "Coverage grid complete. Holding position -- "
                            "Ctrl-C to land.")

            if not self.done and self.tick % 20 == 0:
                self.get_logger().info(
                    f"Shot {self.index}/{len(self.waypoints) - 1}, "
                    f"{distance:.2f}m to go.")

            self.engage_watchdog()
            self.tick += 1

    rclpy.init(args=args)
    node = CoverageFlight()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.publish_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        node.get_logger().info("Land command sent.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
