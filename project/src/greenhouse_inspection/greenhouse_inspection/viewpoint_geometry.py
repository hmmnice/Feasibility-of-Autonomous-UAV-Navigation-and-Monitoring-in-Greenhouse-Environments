"""Pure candidate-viewpoint geometry for targeted greenhouse imaging."""

from dataclasses import dataclass
import math
from typing import Sequence


Point3 = tuple[float, float, float]

# Camera optical origin relative to base_link for a level vehicle.
CAMERA_ORIGIN_IN_BASE = (0.06, 0.0, -0.085)


@dataclass(frozen=True)
class CameraConfig:
    """Changeable simulation camera and stand-off parameters."""

    horizontal_fov: float = 0.9652
    vertical_fov: float = math.radians(42.9)
    width_px: int = 1352
    height_px: int = 1013
    minimum_standoff: float = 1.0
    maximum_standoff: float = 4.0
    desired_standoff: float = 2.0

    def __post_init__(self):
        if not 0.0 < self.horizontal_fov < math.pi:
            raise ValueError("horizontal_fov must be between 0 and pi")
        if not 0.0 < self.vertical_fov < math.pi:
            raise ValueError("vertical_fov must be between 0 and pi")
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("camera resolution must be positive")
        if self.minimum_standoff <= 0.0:
            raise ValueError("minimum_standoff must be positive")
        if self.maximum_standoff < self.minimum_standoff:
            raise ValueError("maximum_standoff must not be below minimum")
        if not (
                self.minimum_standoff
                <= self.desired_standoff
                <= self.maximum_standoff):
            raise ValueError("desired_standoff must lie inside min/max")


@dataclass(frozen=True)
class CandidateSampling:
    """Requested spherical samples around a selected target point."""

    azimuths_deg: tuple[float, ...] = tuple(range(0, 360, 45))
    elevations_deg: tuple[float, ...] = (20.0, 35.0, 50.0)
    distances: tuple[float, ...] = ()

    def __post_init__(self):
        if not self.azimuths_deg:
            raise ValueError("at least one azimuth is required")
        if not self.elevations_deg:
            raise ValueError("at least one elevation is required")
        values = self.azimuths_deg + self.elevations_deg + self.distances
        if not all(math.isfinite(value) for value in values):
            raise ValueError("sampling values must be finite")
        if any(not -90.0 <= value <= 90.0
               for value in self.elevations_deg):
            raise ValueError("elevations must be between -90 and +90 deg")
        if any(value <= 0.0 for value in self.distances):
            raise ValueError("candidate distances must be positive")


@dataclass(frozen=True)
class CandidateViewpoint:
    """One unvalidated camera pose and its target-facing orientation."""

    index: int
    target: Point3
    camera_position: Point3
    distance: float
    azimuth: float
    elevation: float
    camera_yaw_map: float
    camera_pitch: float
    vehicle_yaw_map: float
    gimbal_yaw: float
    gimbal_pitch: float

    @property
    def optical_direction_map(self) -> Point3:
        """Unit ray from this camera position towards the target."""
        return direction_to_target(self.camera_position, self.target)


def _point3(values: Sequence[float], name: str) -> Point3:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly x, y and z")
    point = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in point):
        raise ValueError(f"{name} must contain finite values")
    return point


def wrap_pi(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi]`` while retaining positive pi."""
    wrapped = (float(angle) + math.pi) % (2.0 * math.pi) - math.pi
    if math.isclose(wrapped, -math.pi) and angle > 0.0:
        return math.pi
    return wrapped


def direction_to_target(camera_position, target) -> Point3:
    """Return the map-frame unit vector from camera to target."""
    camera = _point3(camera_position, "camera_position")
    target_point = _point3(target, "target")
    delta = tuple(t - c for c, t in zip(camera, target_point))
    length = math.sqrt(sum(value * value for value in delta))
    if length <= 1e-9:
        raise ValueError("camera position and target must differ")
    return tuple(value / length for value in delta)


def orientation_to_target(
        camera_position, target,
        vehicle_yaw_map: float = 0.0) -> tuple[float, float, float, float]:
    """Return map camera yaw/pitch and level-vehicle gimbal yaw/pitch."""
    direction = direction_to_target(camera_position, target)
    dx, dy, dz = direction
    horizontal = math.hypot(dx, dy)
    camera_yaw = 0.0 if horizontal <= 1e-9 else math.atan2(dy, dx)
    camera_pitch = -math.atan2(dz, horizontal)
    gimbal_yaw = wrap_pi(camera_yaw - float(vehicle_yaw_map))
    return camera_yaw, camera_pitch, gimbal_yaw, camera_pitch


def candidate_position(
        target, distance: float, azimuth: float, elevation: float) -> Point3:
    """Place a camera on a spherical shell around target."""
    target_point = _point3(target, "target")
    distance = float(distance)
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("distance must be positive and finite")
    if not all(math.isfinite(value) for value in (azimuth, elevation)):
        raise ValueError("candidate angles must be finite")
    cos_elevation = math.cos(elevation)
    offset = (
        distance * cos_elevation * math.cos(azimuth),
        distance * cos_elevation * math.sin(azimuth),
        distance * math.sin(elevation),
    )
    return tuple(value + delta for value, delta in zip(target_point, offset))


def vehicle_position_for_camera(
        camera_position, vehicle_yaw_map: float,
        camera_origin_in_base=CAMERA_ORIGIN_IN_BASE) -> Point3:
    """Return the level vehicle centre that places the camera at a map pose."""
    camera = _point3(camera_position, "camera_position")
    offset = _point3(camera_origin_in_base, "camera_origin_in_base")
    yaw = float(vehicle_yaw_map)
    if not math.isfinite(yaw):
        raise ValueError("vehicle_yaw_map must be finite")
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    offset_map = (
        cos_yaw * offset[0] - sin_yaw * offset[1],
        sin_yaw * offset[0] + cos_yaw * offset[1],
        offset[2],
    )
    return tuple(value - delta for value, delta in zip(camera, offset_map))


def generate_candidates(
        target,
        camera: CameraConfig = CameraConfig(),
        sampling: CandidateSampling = CandidateSampling(),
        vehicle_yaw_map: float = 0.0) -> list[CandidateViewpoint]:
    """Generate deterministic unvalidated camera viewpoints around a target."""
    target_point = _point3(target, "target")
    vehicle_yaw_map = float(vehicle_yaw_map)
    if not math.isfinite(vehicle_yaw_map):
        raise ValueError("vehicle_yaw_map must be finite")
    distances = sampling.distances or (camera.desired_standoff,)
    candidates = []
    for distance in distances:
        for elevation_deg in sampling.elevations_deg:
            elevation = math.radians(elevation_deg)
            for azimuth_deg in sampling.azimuths_deg:
                azimuth = math.radians(azimuth_deg)
                position = candidate_position(
                    target_point, distance, azimuth, elevation)
                camera_yaw, camera_pitch, gimbal_yaw, gimbal_pitch = (
                    orientation_to_target(
                        position, target_point, vehicle_yaw_map))
                candidates.append(CandidateViewpoint(
                    index=len(candidates),
                    target=target_point,
                    camera_position=position,
                    distance=float(distance),
                    azimuth=azimuth,
                    elevation=elevation,
                    camera_yaw_map=camera_yaw,
                    camera_pitch=camera_pitch,
                    vehicle_yaw_map=vehicle_yaw_map,
                    gimbal_yaw=gimbal_yaw,
                    gimbal_pitch=gimbal_pitch,
                ))
    return candidates
