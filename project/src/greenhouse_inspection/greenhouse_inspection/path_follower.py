"""Repeat pass: fly the route the teach pass recorded."""

import math

import numpy as np

from greenhouse_inspection.relocalize import (
    decompose,
    traj_path,
    transform_points,
    yaw_transform,
)

# turning a hand-flown trace into setpoints ---------------------------- The teach pass records /odometry at ~250Hz, so 20m of hand-flown.
MAX_STEP = 0.5          # m between consecutive waypoints
MAX_YAW_STEP = 20.0     # deg of heading change between consecutive waypoints

# trust gate ----------------------------------------------------------- Both flights boot on the same floor and FAST-LIO gravity-aligns both frames, so the true.
MAX_RELOC_Z = 0.5       # m
# Gross sanity on where the route starts relative to where we are hovering.
MAX_START_DIST = 1.5    # m
# A greenhouse route doubles back.
START_SEARCH_ARC = 3.0  # m of taught route considered as a starting point

# --- flight ---------------------------------------------------------------
REACH_RADIUS = 0.3      # m; under MAX_STEP so progress stays one leg at a time
# If relocalize.py never answers, hovering until the battery dies is worse than landing on the pad we are.
RELOC_TIMEOUT = 90.0    # s from first odometry


def simplify_path(path, max_step=MAX_STEP, max_yaw=MAX_YAW_STEP):
    """Dense odometry -> the waypoints worth publishing."""
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return path
    kept = [0]
    for i in range(1, len(path)):
        anchor = path[kept[-1]]
        turned = abs(math.atan2(math.sin(path[i, 3] - anchor[3]),
                                math.cos(path[i, 3] - anchor[3])))
        if (np.linalg.norm(path[i, :3] - anchor[:3]) >= max_step
                or turned >= math.radians(max_yaw)):
            # Keep the last sample still inside the bounds, not the first one outside them, so max_step is.
            kept.append(max(i - 1, kept[-1] + 1))
    # The pilot's last position is where the route ends.
    if kept[-1] != len(path) - 1:
        kept.append(len(path) - 1)
    return path[kept]


def enu_yaw_to_ned(yaw_enu):
    """ENU heading -> NED heading."""
    return math.pi / 2 - yaw_enu


def taught_to_px4(path, T_saved_from_map):
    """Taught -> PX4 NED setpoints in THIS flight's frame."""
    path = np.asarray(path, dtype=float)
    T = np.linalg.inv(np.asarray(T_saved_from_map, dtype=float))
    xyz = transform_points(path[:, :3], T)
    yaw_enu = path[:, 3] + decompose(T)[3]
    return np.column_stack(
        [xyz[:, 1], xyz[:, 0], -xyz[:, 2], enu_yaw_to_ned(yaw_enu)])


def start_index(waypoints, position):
    """Where on the route to pick up, and how far away that is."""
    xyz = np.asarray(waypoints, dtype=float)[:, :3]
    arc = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
    window = xyz[arc <= START_SEARCH_ARC]
    d = np.linalg.norm(window - np.asarray(position, dtype=float), axis=1)
    i = int(np.argmin(d))
    return i, float(d[i])


def refuse_reason(T, waypoints, start_dist):
    """Why this route must not be flown, or '' if it may be."""
    z = decompose(T)[2]
    if abs(z) > MAX_RELOC_Z:
        return (f"relocalization shifts the map {z:+.2f}m vertically, over the "
                f"{MAX_RELOC_Z}m limit. Both flights boot on the same floor and "
                f"both frames are gravity-aligned, so a correct answer has "
                f"almost no z -- this one slid.")

    # simplify_path bounds legs at max_step, but only if the recorded trajectory is continuous.
    legs = np.linalg.norm(np.diff(np.asarray(waypoints, dtype=float)[:, :3], axis=0), axis=1)
    if len(legs) and legs.max() > 2 * MAX_STEP:
        return (f"taught route contains a {legs.max():.1f}m jump between "
                f"consecutive waypoints (limit {2 * MAX_STEP}m). The recording "
                f"is discontinuous -- SLAM lost track during the teach pass. "
                f"Re-fly it.")

    if start_dist > MAX_START_DIST:
        return (f"nearest taught waypoint is {start_dist:.2f}m away, over the "
                f"{MAX_START_DIST}m limit. The repeat pass has to start where "
                f"the teach pass started; flying that far to join the route "
                f"would cross rows on the way.")
    return ""


# ROS node.

def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener
    from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

    from px4_msgs.msg import (
        OffboardControlMode,
        TrajectorySetpoint,
        VehicleCommand,
        VehicleOdometry,
        VehicleStatus,
    )

    class PathFollower(Node):
        def __init__(self):
            super().__init__("path_follower")

            map_path = self.declare_parameter(
                "map_path", "/root/masters-thesis/teach_map.npy").value
            self.warmup_alt = self.declare_parameter("warmup_alt", 1.2).value
            # Metres to fly along the drone's boot heading after climbing, if the climb alone does not give relocalize.py.
            self.warmup_forward = self.declare_parameter("warmup_forward", 0.0).value

            self.taught = np.load(traj_path(map_path))
            self.get_logger().info(
                f"Loaded {len(self.taught)} taught route samples from "
                f"{traj_path(map_path)}.")

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

            self.offboard_pub = self.create_publisher(
                OffboardControlMode, "/fmu/in/offboard_control_mode", pub_qos)
            self.setpoint_pub = self.create_publisher(
                TrajectorySetpoint, "/fmu/in/trajectory_setpoint", pub_qos)
            self.command_pub = self.create_publisher(
                VehicleCommand, "/fmu/in/vehicle_command", pub_qos)

            self.create_subscription(
                VehicleOdometry, "/fmu/out/vehicle_odometry", self.odom_cb, sub_qos)
            # vehicle_status_v4, not vehicle_status: this PX4 build versions the topic and waypoint_follower.py's unversioned subscription silently receives nothing.
            self.create_subscription(
                VehicleStatus, "/fmu/out/vehicle_status_v4", self.status_cb, sub_qos)

            self.tf_buffer = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
            self.tf_errors = (LookupException, ConnectivityException,
                              ExtrapolationException)

            self.position = None      # latest PX4 NED position
            self.launch = None        # PX4 NED position at first odometry
            self.t0 = None
            self.waypoints = None     # None until relocalized; then (N, 4) NED
            self.index = 0
            self.at_end = False
            self.landing = False

            self.armed = False
            self.nav_state = None
            self.tick = 0
            self.last_engage_tick = None

            self.timer = self.create_timer(0.1, self.timer_callback)
            self.get_logger().info(
                "Repeat pass: waiting for PX4 odometry, then climbing to "
                f"{self.warmup_alt:.1f}m while relocalize.py works out where "
                "we are. Will not follow the route without that answer.")

        # --- subscriptions ---------------------------------------------
        def odom_cb(self, msg):
            self.position = np.array(msg.position, dtype=float)
            if self.launch is None:
                self.launch = self.position.copy()
                self.t0 = self.get_clock().now()
                self.get_logger().info(
                    f"Launch position (PX4 NED): {self.launch.round(2).tolist()}")

        def status_cb(self, msg):
            self.armed = msg.arming_state == VehicleStatus.ARMING_STATE_ARMED
            self.nav_state = msg.nav_state

        # --- publishing -------------------------------------------------
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
            """Keep the vehicle armed and in offboard."""
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
            self.get_logger().warn(f"Landing: {reason}")

        # --- relocalization gate ----------------------------------------
        def warmup_target(self):
            return [self.launch[0],
                    self.launch[1] + self.warmup_forward,
                    self.launch[2] - self.warmup_alt]

        def try_start_following(self):
            """Look for relocalize.py's answer; adopt the route if it is sound."""
            try:
                tf = self.tf_buffer.lookup_transform("saved_map", "map", Time())
            except self.tf_errors:
                elapsed = (self.get_clock().now() - self.t0).nanoseconds / 1e9
                if elapsed > RELOC_TIMEOUT:
                    self.land(
                        f"no saved_map -> map transform after {RELOC_TIMEOUT:.0f}s. "
                        "relocalize.py either never solved or refused the answer "
                        "as ambiguous. Not flying a route we cannot place -- "
                        "check its log, then re-launch closer to the taught "
                        "start pose.")
                else:
                    self.get_logger().info(
                        f"Holding at the pad: no relocalization yet ({elapsed:.0f}s "
                        f"of {RELOC_TIMEOUT:.0f}s).", throttle_duration_sec=5.0)
                return

            t, q = tf.transform.translation, tf.transform.rotation
            # relocalize.py solves 4 DOF, so the rotation is pure yaw.
            T = yaw_transform(t.x, t.y, t.z, 2 * math.atan2(q.z, q.w))

            waypoints = taught_to_px4(simplify_path(self.taught), T)
            index, distance = start_index(waypoints, self.position)
            reason = refuse_reason(T, waypoints, distance)
            if reason:
                self.land(f"REFUSING the taught route -- {reason}")
                return

            self.waypoints = waypoints
            self.index = index
            self.get_logger().info(
                f"Relocalized and accepted. {len(self.taught)} taught samples -> "
                f"{len(waypoints)} waypoints; joining at #{index} "
                f"({distance:.2f}m away).")

        # --- main loop ---------------------------------------------------
        def timer_callback(self):
            if self.landing:
                return
            self.publish_offboard_mode()
            if self.position is None:
                return

            if self.waypoints is None:
                self.publish_setpoint(self.warmup_target(), float("nan"))
                self.try_start_following()
            else:
                waypoint = self.waypoints[self.index]
                self.publish_setpoint(waypoint[:3], waypoint[3])
                distance = float(np.linalg.norm(self.position - waypoint[:3]))
                if distance < REACH_RADIUS:
                    self.advance()
                if self.tick % 20 == 0:
                    self.get_logger().info(
                        f"Waypoint {self.index}/{len(self.waypoints) - 1}, "
                        f"{distance:.2f}m to go.")

            self.engage_watchdog()
            self.tick += 1

        def advance(self):
            if self.index < len(self.waypoints) - 1:
                self.index += 1
            elif not self.at_end:
                self.at_end = True
                self.get_logger().info(
                    "Taught route complete. Holding position -- Ctrl-C to land.")

    rclpy.init(args=args)
    node = PathFollower()
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
