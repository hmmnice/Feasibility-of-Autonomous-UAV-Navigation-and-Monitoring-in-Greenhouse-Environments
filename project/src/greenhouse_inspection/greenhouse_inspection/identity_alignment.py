"""Publish the surveyed-map/Gazebo identity transform for GPS experiments."""


def main(args=None):
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

    class IdentityAlignment(Node):
        def __init__(self):
            super().__init__("gps_viewpoint_identity_alignment")
            self.broadcaster = StaticTransformBroadcaster(self)
            transform = TransformStamped()
            transform.header.stamp = self.get_clock().now().to_msg()
            transform.header.frame_id = "saved_map"
            transform.child_frame_id = "map"
            transform.transform.rotation.w = 1.0
            self.broadcaster.sendTransform(transform)
            self.get_logger().warn(
                "GPS MODE: publishing identity saved_map -> map because the "
                "surveyed planning map and Gazebo ENU frame share one origin. "
                "This is not a SLAM/relocalisation transform.")

    rclpy.init(args=args)
    node = IdentityAlignment()
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
