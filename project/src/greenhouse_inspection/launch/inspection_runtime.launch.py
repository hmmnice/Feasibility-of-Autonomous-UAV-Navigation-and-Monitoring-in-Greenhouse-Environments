"""Runtime support for GPS-first viewpoint flights."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="greenhouse_inspection",
            executable="identity_alignment",
            name="inspection_identity_alignment",
            output="screen",
        ),
        Node(
            package="greenhouse_inspection",
            executable="gimbal_controller",
            name="inspection_gimbal_controller",
            output="screen",
        ),
    ])
