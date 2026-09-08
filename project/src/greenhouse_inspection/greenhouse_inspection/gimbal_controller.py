"""Stationary two-axis gimbal look-at controller for the Gazebo vehicle."""

import math

from geometry_msgs.msg import PointStamped, TransformStamped
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64
from tf2_ros import TransformBroadcaster

from .gimbal_geometry import (
    JointLimits,
    angular_error,
    camera_pitch_to_joint,
    joint_pitch_to_camera,
    solve_look_at,
)


MOUNT_POSITION = (0.06, 0.0, -0.05)
PITCH_LINK_OFFSET = (0.0, 0.0, -0.035)
YAW_JOINT = "gimbal_yaw_joint"
PITCH_JOINT = "gimbal_pitch_joint"


def quaternion_from_euler(roll, pitch, yaw):
    """Return an ``(x, y, z, w)`` quaternion from fixed-axis RPY angles."""
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


class GimbalController(Node):
    """Aim the simulated camera at a point expressed in ``base_link``."""

    def __init__(self):
        super().__init__("gimbal_controller")
        self.declare_parameter("target_x", 2.0)
        self.declare_parameter("target_y", 0.0)
        self.declare_parameter("target_z", -2.0)
        self.declare_parameter("yaw_min_deg", -170.0)
        self.declare_parameter("yaw_max_deg", 170.0)
        self.declare_parameter("pitch_min_deg", -10.0)
        self.declare_parameter("pitch_max_deg", 100.0)
        self.declare_parameter("settle_tolerance_deg", 2.0)
        self.declare_parameter("settle_samples", 10)
        self.declare_parameter("command_rate_hz", 20.0)
        self.declare_parameter("clamp_unreachable", False)

        degrees = math.radians
        self.limits = JointLimits(
            yaw_min=degrees(self.get_parameter("yaw_min_deg").value),
            yaw_max=degrees(self.get_parameter("yaw_max_deg").value),
            pitch_min=degrees(self.get_parameter("pitch_min_deg").value),
            pitch_max=degrees(self.get_parameter("pitch_max_deg").value),
        )
        self.settle_tolerance = degrees(
            self.get_parameter("settle_tolerance_deg").value)
        self.settle_samples = max(
            1, int(self.get_parameter("settle_samples").value))
        self.clamp_unreachable = bool(
            self.get_parameter("clamp_unreachable").value)
        self.target_point = (
            float(self.get_parameter("target_x").value),
            float(self.get_parameter("target_y").value),
            float(self.get_parameter("target_z").value),
        )

        self.yaw_publisher = self.create_publisher(
            Float64, "/gimbal/yaw_cmd", 10)
        self.pitch_publisher = self.create_publisher(
            Float64, "/gimbal/pitch_cmd", 10)
        self.settled_publisher = self.create_publisher(
            Bool, "/gimbal/settled", 10)
        self.create_subscription(
            PointStamped, "/gimbal/look_at", self._target_callback, 10)
        self.create_subscription(
            JointState, "/gimbal/joint_states", self._joint_callback, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.actual_yaw = None
        self.actual_camera_pitch = None
        self.good_samples = 0
        self.was_settled = False
        self.target_revision = 0
        self.reported_unreachable_revision = -1

        rate = float(self.get_parameter("command_rate_hz").value)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("command_rate_hz must be positive")
        self.create_timer(1.0 / rate, self._command_timer)
        self.get_logger().info(
            "Gimbal bench controller ready: targets are base_link points on "
            "/gimbal/look_at; default target=(%.2f, %.2f, %.2f) m"
            % self.target_point)

    def _target_vector(self):
        return tuple(
            target - mount
            for target, mount in zip(self.target_point, MOUNT_POSITION)
        )

    def _target_callback(self, message):
        if message.header.frame_id not in ("", "base_link"):
            self.get_logger().error(
                "Rejected gimbal target in frame '%s'; Phase 1 accepts only "
                "base_link" % message.header.frame_id)
            return
        point = message.point
        values = (float(point.x), float(point.y), float(point.z))
        if not all(math.isfinite(value) for value in values):
            self.get_logger().error("Rejected non-finite gimbal target")
            return
        self.target_point = values
        self.target_revision += 1
        self.good_samples = 0
        self.was_settled = False
        self.get_logger().info(
            "New base_link target: (%.3f, %.3f, %.3f) m" % values)

    def _joint_callback(self, message):
        positions = dict(zip(message.name, message.position))
        if YAW_JOINT not in positions or PITCH_JOINT not in positions:
            return
        self.actual_yaw = float(positions[YAW_JOINT])
        self.actual_camera_pitch = joint_pitch_to_camera(
            positions[PITCH_JOINT])
        self._publish_gimbal_tf()

    def _command_timer(self):
        try:
            solution = solve_look_at(
                self._target_vector(), self.limits,
                clamp=self.clamp_unreachable)
        except ValueError as error:
            self._report_unreachable(str(error))
            self._publish_settled(False)
            return

        if not solution.reachable and not self.clamp_unreachable:
            self._report_unreachable(
                "requires yaw %.1f deg, pitch %.1f deg outside configured "
                "limits" % (
                    math.degrees(solution.unclamped_yaw),
                    math.degrees(solution.unclamped_pitch),
                ))
            self._publish_settled(False)
            return

        self.yaw_publisher.publish(Float64(data=solution.yaw))
        self.pitch_publisher.publish(Float64(
            data=camera_pitch_to_joint(solution.pitch)))

        settled = False
        if (
                self.actual_yaw is not None
                and self.actual_camera_pitch is not None):
            error = angular_error(
                self._target_vector(), self.actual_yaw,
                self.actual_camera_pitch)
            self.good_samples = (
                self.good_samples + 1
                if error <= self.settle_tolerance else 0)
            settled = self.good_samples >= self.settle_samples
            if settled and not self.was_settled:
                self.get_logger().info(
                    "Gimbal settled: optical-axis error %.2f deg"
                    % math.degrees(error))
        self._publish_settled(settled)

    def _report_unreachable(self, reason):
        if self.reported_unreachable_revision == self.target_revision:
            return
        self.reported_unreachable_revision = self.target_revision
        self.get_logger().error("Gimbal target rejected: %s" % reason)

    def _publish_settled(self, settled):
        self.settled_publisher.publish(Bool(data=settled))
        self.was_settled = settled

    def _publish_gimbal_tf(self):
        stamp = self.get_clock().now().to_msg()
        yaw_tf = self._transform(
            stamp, "base_link", "gimbal_yaw_link", MOUNT_POSITION,
            quaternion_from_euler(0.0, 0.0, self.actual_yaw))
        camera_tf = self._transform(
            stamp, "gimbal_yaw_link", "camera_sensor_link",
            PITCH_LINK_OFFSET,
            quaternion_from_euler(0.0, self.actual_camera_pitch, 0.0))
        # ROS optical convention: +Z forward, +X right, +Y down.  The Gazebo
        # camera link convention is +X forward, +Y left, +Z up.
        optical_tf = self._transform(
            stamp, "camera_sensor_link", "camera_optical_frame",
            (0.0, 0.0, 0.0),
            quaternion_from_euler(-math.pi / 2.0, 0.0, -math.pi / 2.0))
        self.tf_broadcaster.sendTransform([yaw_tf, camera_tf, optical_tf])

    @staticmethod
    def _transform(stamp, parent, child, translation, quaternion):
        message = TransformStamped()
        message.header.stamp = stamp
        message.header.frame_id = parent
        message.child_frame_id = child
        message.transform.translation.x = float(translation[0])
        message.transform.translation.y = float(translation[1])
        message.transform.translation.z = float(translation[2])
        message.transform.rotation.x = float(quaternion[0])
        message.transform.rotation.y = float(quaternion[1])
        message.transform.rotation.z = float(quaternion[2])
        message.transform.rotation.w = float(quaternion[3])
        return message


def main(args=None):
    rclpy.init(args=args)
    node = GimbalController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
