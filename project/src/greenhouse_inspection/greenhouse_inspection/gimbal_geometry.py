"""Pure geometry for the greenhouse camera gimbal."""

from dataclasses import dataclass
import math
from typing import Sequence


MECHANICAL_PITCH_OFFSET = math.pi / 2.0


@dataclass(frozen=True)
class JointLimits:
    """Inclusive yaw and pitch limits, in radians."""

    yaw_min: float = math.radians(-170.0)
    yaw_max: float = math.radians(170.0)
    pitch_min: float = math.radians(-10.0)
    pitch_max: float = math.radians(100.0)

    def __post_init__(self):
        if self.yaw_min > self.yaw_max:
            raise ValueError("yaw_min must not exceed yaw_max")
        if self.pitch_min > self.pitch_max:
            raise ValueError("pitch_min must not exceed pitch_max")


@dataclass(frozen=True)
class LookAtSolution:
    """Commanded gimbal angles and whether the target was reachable."""

    yaw: float
    pitch: float
    reachable: bool
    unclamped_yaw: float
    unclamped_pitch: float


def _target_components(target: Sequence[float]) -> tuple[float, float, float]:
    if len(target) != 3:
        raise ValueError("target vector must contain exactly x, y and z")
    x, y, z = (float(value) for value in target)
    if not all(math.isfinite(value) for value in (x, y, z)):
        raise ValueError("target vector must contain finite values")
    if math.sqrt(x * x + y * y + z * z) <= 1e-9:
        raise ValueError("target vector must not be zero length")
    return x, y, z


def solve_look_at(
        target: Sequence[float],
        limits: JointLimits = JointLimits(),
        clamp: bool = False) -> LookAtSolution:
    """Return yaw and pitch that aim the camera +x axis at target."""
    x, y, z = _target_components(target)
    horizontal = math.hypot(x, y)
    yaw = 0.0 if horizontal <= 1e-9 else math.atan2(y, x)
    pitch = -math.atan2(z, horizontal)
    reachable = (
        limits.yaw_min <= yaw <= limits.yaw_max
        and limits.pitch_min <= pitch <= limits.pitch_max
    )
    command_yaw = yaw
    command_pitch = pitch
    if clamp:
        command_yaw = min(max(yaw, limits.yaw_min), limits.yaw_max)
        command_pitch = min(max(pitch, limits.pitch_min), limits.pitch_max)
    return LookAtSolution(
        yaw=command_yaw,
        pitch=command_pitch,
        reachable=reachable,
        unclamped_yaw=yaw,
        unclamped_pitch=pitch,
    )


def optical_axis(yaw: float, pitch: float) -> tuple[float, float, float]:
    """Return the camera forward axis in ``base_link`` for two joint angles."""
    cos_pitch = math.cos(pitch)
    return (
        math.cos(yaw) * cos_pitch,
        math.sin(yaw) * cos_pitch,
        -math.sin(pitch),
    )


def camera_pitch_to_joint(camera_pitch: float) -> float:
    """Convert intuitive camera pitch to the offset SDF joint coordinate."""
    return float(camera_pitch) - MECHANICAL_PITCH_OFFSET


def joint_pitch_to_camera(joint_pitch: float) -> float:
    """Convert the offset SDF joint coordinate to intuitive camera pitch."""
    return float(joint_pitch) + MECHANICAL_PITCH_OFFSET


def angular_error(
        target: Sequence[float], yaw: float, pitch: float) -> float:
    """Return the optical-axis error to ``target`` in radians."""
    x, y, z = _target_components(target)
    magnitude = math.sqrt(x * x + y * y + z * z)
    direction = (x / magnitude, y / magnitude, z / magnitude)
    axis = optical_axis(float(yaw), float(pitch))
    dot = sum(a * b for a, b in zip(direction, axis))
    return math.acos(min(1.0, max(-1.0, dot)))
