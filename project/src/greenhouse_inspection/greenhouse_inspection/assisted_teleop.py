import math
import sys
import termios
import threading
import time
import tty

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from sensor_msgs.msg import Joy, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleOdometry,
    VehicleStatus,
)

SPEED = 0.6           # m/s for translation keys
YAW_RATE = 0.6         # rad/s for yaw keys
# Include the airframe radius in LiDAR clearance distances.
AIRFRAME_RADIUS = 0.3  # Conservative x500 propeller radius in metres.
STOP_MARGIN = 0.6 + AIRFRAME_RADIUS   # Stop at or inside this distance.
BRAKE_START = 2.0 + AIRFRAME_RADIUS   # Begin braking at this distance.
CONE_HALF_ANGLE_DEG = 20.0
MAX_CHECK_RANGE = 4.0  # m -- ignore points farther than this

# Limit vertical motion with known floor and eave heights.
GROUND_STOP = 0.3
GROUND_BRAKE_START = 1.0
EAVE = 6.7
CEILING_STOP = 0.5
CEILING_BRAKE_START = 1.5

# Treat a missing key repeat as release.
KEY_HOLD_TIMEOUT = 0.3  # Seconds.

ZERO_CMD = (0.0, 0.0, 0.0, 0.0)

# Body-FLU velocity and yaw rate while a key repeats.
KEY_BINDINGS = {
    "w": (SPEED, 0.0, 0.0, 0.0),
    "s": (-SPEED, 0.0, 0.0, 0.0),
    "a": (0.0, SPEED, 0.0, 0.0),
    "d": (0.0, -SPEED, 0.0, 0.0),
    "r": (0.0, 0.0, SPEED, 0.0),
    "f": (0.0, 0.0, -SPEED, 0.0),
    "q": (0.0, 0.0, 0.0, -YAW_RATE),
    "e": (0.0, 0.0, 0.0, YAW_RATE),
    " ": ZERO_CMD,
}

# Default PS4 mappings; override parameters for other controllers.
JOY_AXIS_X = 1     # left stick vertical    -> forward/back
JOY_AXIS_Y = 0     # left stick horizontal  -> left/right
JOY_AXIS_Z = 4     # right stick vertical   -> up/down
JOY_AXIS_YAW = 3   # right stick horizontal -> yaw rate
JOY_ENABLE_BUTTON = 5  # R1 -- deadman switch, must be held for any command
JOY_DEADZONE = 0.15
JOY_STALE_TIMEOUT = 0.5  # s -- no /joy message this long -> treat as released

HELP = f"""
Assisted manual teleop -- hold a key to move, release to stop (RC-style),
or use a joystick (`ros2 run joy joy_node`): left stick to translate, right
stick for yaw + altitude, hold R1 to command. Both inputs work at once --
whichever moved last wins. Lidar brakes smoothly as you close in on an
obstacle ahead/beside/behind: full speed beyond {BRAKE_START}m, easing to a
stop by {STOP_MARGIN}m. The lidar can't see straight up/down, so vertical
motion is instead limited by known altitude: eases to a stop by
{GROUND_STOP}m above the floor and {CEILING_STOP}m below the eave.

  w/s : forward / back      r/f : up / down
  a/d : left / right        q/e : yaw left / right
  space : stop              Ctrl-C : land + quit
"""


class KeyReader:
    """Background raw-terminal reader so keys don't block the 10Hz control loop."""

    def __init__(self):
        self.key = None
        self._lock = threading.Lock()
        self._running = True
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        tty.setraw(self._fd)
        self._thread.start()

    def _loop(self):
        while self._running:
            ch = sys.stdin.read(1)
            with self._lock:
                self.key = ch

    def get_key(self):
        with self._lock:
            k, self.key = self.key, None
        return k

    def stop(self):
        self._running = False
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)


def yaw_from_quat(q):
    # q = (w, x, y, z), PX4 NED/FRD convention (same field order used
    # throughout this project, see slam_to_px4_odometry.py).
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _ease(distance, stop_margin, brake_start):
    """0 at/inside stop_margin, 1 at/beyond brake_start, linear between."""
    return min(1.0, max(0.0, (distance - stop_margin) / (brake_start - stop_margin)))


def joy_to_cmd(axes, buttons, axis_x=JOY_AXIS_X, axis_y=JOY_AXIS_Y,
               axis_z=JOY_AXIS_Z, axis_yaw=JOY_AXIS_YAW,
               enable_button=JOY_ENABLE_BUTTON, deadzone=JOY_DEADZONE):
    """Convert joystick input to body-FLU velocity and yaw commands."""
    if enable_button >= 0 and (
            enable_button >= len(buttons) or not buttons[enable_button]):
        return ZERO_CMD

    def scaled(axis, limit):
        if axis >= len(axes):
            return 0.0
        v = axes[axis]
        return 0.0 if abs(v) < deadzone else v * limit

    return (scaled(axis_x, SPEED), scaled(axis_y, SPEED),
            scaled(axis_z, SPEED), scaled(axis_yaw, YAW_RATE))


def clamp_vertical(vz, altitude):
    """Brake vertical motion near the floor or eave."""
    if vz < 0.0:
        scale = _ease(altitude, GROUND_STOP, GROUND_BRAKE_START)
    elif vz > 0.0:
        scale = _ease(EAVE - altitude, CEILING_STOP, CEILING_BRAKE_START)
    else:
        return vz, 1.0
    return vz * scale, scale


def clamp_to_obstacle(vx, vy, vz, points_flu):
    """Scale movement within the forward obstacle cone."""
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
    if speed < 1e-3 or points_flu.shape[0] == 0:
        return vx, vy, vz, 1.0

    direction = np.array([vx, vy, vz]) / speed
    dists = np.linalg.norm(points_flu, axis=1)
    in_range = dists < MAX_CHECK_RANGE
    if not np.any(in_range):
        return vx, vy, vz, 1.0

    pts, dists = points_flu[in_range], dists[in_range]
    cos_angle = (pts @ direction) / dists
    in_cone = cos_angle > math.cos(math.radians(CONE_HALF_ANGLE_DEG))
    if not np.any(in_cone):
        return vx, vy, vz, 1.0

    scale = _ease(dists[in_cone].min(), STOP_MARGIN, BRAKE_START)
    return vx * scale, vy * scale, vz * scale, scale


class AssistedTeleop(Node):
    def __init__(self):
        super().__init__("assisted_teleop")

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

        self.offboard_control_mode_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", pub_qos
        )
        self.trajectory_setpoint_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", pub_qos
        )
        self.vehicle_command_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", pub_qos
        )

        self.create_subscription(
            VehicleOdometry, "/fmu/out/vehicle_odometry", self.odom_cb, sub_qos
        )
        self.create_subscription(
            PointCloud2, "/lidar/points", self.lidar_cb, sub_qos
        )
        self.create_subscription(
            Joy, "/joy", self.joy_cb, sub_qos
        )
        self.joy_axis_x = self.declare_parameter("joy_axis_x", JOY_AXIS_X).value
        self.joy_axis_y = self.declare_parameter("joy_axis_y", JOY_AXIS_Y).value
        self.joy_axis_z = self.declare_parameter("joy_axis_z", JOY_AXIS_Z).value
        self.joy_axis_yaw = self.declare_parameter("joy_axis_yaw", JOY_AXIS_YAW).value
        self.joy_enable_button = self.declare_parameter(
            "joy_enable_button", JOY_ENABLE_BUTTON).value
        # Subscribe to the active PX4 VehicleStatus topic.
        self.create_subscription(
            # A stale versioned graph entry may have no writer.
            VehicleStatus, "/fmu/out/vehicle_status_v1", self.status_cb, sub_qos
        )

        self.current_yaw = 0.0
        self.current_altitude = 0.0  # -position_z (NED); 0 = spawn height
        self.latest_points = np.zeros((0, 3), dtype=np.float32)
        self.cmd = ZERO_CMD  # vx, vy, vz (body FLU), yawrate -- merged output
        self.keyboard_cmd = ZERO_CMD
        self.joy_cmd = ZERO_CMD

        self.armed = False
        self.nav_state = None
        self.offboard_setpoint_counter = 0
        self.last_engage_tick = None
        self.last_key_time = None
        self.last_joy_time = None

        self.key_reader = KeyReader()
        self.key_reader.start()

        self.timer = self.create_timer(0.1, self.timer_callback)

        print(HELP, flush=True)
        self.get_logger().info("Assisted teleop started. Waiting for odometry...")

    def odom_cb(self, msg):
        self.current_yaw = yaw_from_quat(tuple(msg.q))
        self.current_altitude = -float(msg.position[2])

    def status_cb(self, msg):
        was_ready = self.armed and self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        self.armed = msg.arming_state == VehicleStatus.ARMING_STATE_ARMED
        self.nav_state = msg.nav_state
        is_ready = self.armed and self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        if was_ready and not is_ready:
            self.get_logger().warn(
                f"Dropped out of armed+offboard (armed={self.armed}, "
                f"nav_state={self.nav_state}) -- will re-engage."
            )

    def joy_cb(self, msg):
        self.last_joy_time = time.monotonic()
        self.joy_cmd = joy_to_cmd(
            msg.axes, msg.buttons,
            axis_x=self.joy_axis_x, axis_y=self.joy_axis_y,
            axis_z=self.joy_axis_z, axis_yaw=self.joy_axis_yaw,
            enable_button=self.joy_enable_button)

    def lidar_cb(self, msg):
        # Extra cloud fields require ``read_points`` rather than NumPy decoding.
        pts = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        self.latest_points = np.column_stack(
            [pts["x"], pts["y"], pts["z"]]
        ).astype(np.float32)

    def get_timestamp(self):
        return int(self.get_clock().now().nanoseconds / 1000)

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = self.get_timestamp()
        msg.position = False
        msg.velocity = True
        self.offboard_control_mode_pub.publish(msg)

    def publish_vehicle_command(self, command, **params):
        msg = VehicleCommand()
        msg.timestamp = self.get_timestamp()
        msg.command = command
        msg.param1 = float(params.get("param1", 0.0))
        msg.param2 = float(params.get("param2", 0.0))
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.vehicle_command_pub.publish(msg)

    def timer_callback(self):
        key = self.key_reader.get_key()
        if key == "\x03":  # Ctrl-C
            raise KeyboardInterrupt
        now = time.monotonic()
        if key in KEY_BINDINGS:
            self.last_key_time = now
            if KEY_BINDINGS[key] != self.keyboard_cmd:
                self.keyboard_cmd = KEY_BINDINGS[key]
                self.get_logger().info(f"key '{key.strip() or 'space'}' -> {self.keyboard_cmd}")
        elif (self.last_key_time is not None
              and now - self.last_key_time > KEY_HOLD_TIMEOUT
              and self.keyboard_cmd != ZERO_CMD):
            self.keyboard_cmd = ZERO_CMD
            self.get_logger().info("key released -> stop")

        # Stop if joystick messages go stale.
        if (self.last_joy_time is None
                or now - self.last_joy_time > JOY_STALE_TIMEOUT):
            self.joy_cmd = ZERO_CMD

        # Prefer active keyboard input over joystick input.
        self.cmd = self.keyboard_cmd if self.keyboard_cmd != ZERO_CMD else self.joy_cmd

        self.publish_offboard_control_mode()

        # Retry offboard engagement until vehicle status confirms it.
        is_ready = self.armed and self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        can_attempt = self.last_engage_tick is None or (
            self.offboard_setpoint_counter - self.last_engage_tick >= 10
        )
        if self.offboard_setpoint_counter >= 20 and not is_ready and can_attempt:
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0
            )
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0
            )
            self.last_engage_tick = self.offboard_setpoint_counter
            self.get_logger().info(
                f"Offboard + arm (re-)sent (armed={self.armed}, nav_state={self.nav_state})."
            )
        self.offboard_setpoint_counter += 1

        vx, vy, vz, yawrate = self.cmd
        vx, vy, vz, scale = clamp_to_obstacle(vx, vy, vz, self.latest_points)
        if scale < 1.0:
            self.get_logger().warn(
                f"Obstacle ahead -- braking ({scale * 100:.0f}% speed).",
                throttle_duration_sec=1.0,
            )
        commanded_vz = vz
        vz, v_scale = clamp_vertical(vz, self.current_altitude)
        if v_scale < 1.0:
            which = "floor" if commanded_vz < 0.0 else "ceiling"
            self.get_logger().warn(
                f"Near {which} ({self.current_altitude:.1f}m) -- braking "
                f"({v_scale * 100:.0f}% speed).",
                throttle_duration_sec=1.0,
            )

        # FLU body -> FRD body (fixed axis flip) -> NED world (yaw rotation).
        x_frd, y_frd, z_frd = vx, -vy, -vz
        yaw = self.current_yaw
        n = math.cos(yaw) * x_frd - math.sin(yaw) * y_frd
        e = math.sin(yaw) * x_frd + math.cos(yaw) * y_frd
        d = z_frd

        msg = TrajectorySetpoint()
        msg.timestamp = self.get_timestamp()
        msg.position = [float("nan")] * 3
        msg.velocity = [float(n), float(e), float(d)]
        msg.yaw = float("nan")
        msg.yawspeed = float(yawrate)
        self.trajectory_setpoint_pub.publish(msg)

    def land(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Land command sent.")


def main(args=None):
    rclpy.init(args=args)
    node = AssistedTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.land()
    finally:
        node.key_reader.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
