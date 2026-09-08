"""Tile a mapped area with a systematic photo-coverage grid."""

import math

# Raspberry Pi HQ camera with a 6 mm lens.
CAMERA_HFOV = 0.9652   # rad, 55.3 deg
CAMERA_VFOV = 0.7483   # rad, 42.9 deg

# Overlap adjacent images to avoid coverage gaps.
DEFAULT_OVERLAP = 0.2


def ground_footprint(altitude, hfov=CAMERA_HFOV, vfov=CAMERA_VFOV):
    """Return ground-footprint width and height at a given altitude."""
    width = 2 * altitude * math.tan(hfov / 2)
    height = 2 * altitude * math.tan(vfov / 2)
    return width, height


def _axis_points(lo, hi, tile_size, step):
    """Return tile centres that cover one axis, including both edges."""
    span = hi - lo
    if span <= tile_size:
        return [lo + span / 2]
    points = [lo + tile_size / 2]
    while points[-1] + tile_size / 2 < hi:
        points.append(points[-1] + step)
    return points


def generate_coverage_grid(
        x_min, x_max, y_min, y_max, altitude,
        hfov=CAMERA_HFOV, vfov=CAMERA_VFOV, overlap=DEFAULT_OVERLAP):
    """Return serpentine map-frame capture points for the requested area."""
    tile_w, tile_h = ground_footprint(altitude, hfov, vfov)
    step_x = tile_w * (1 - overlap)
    step_y = tile_h * (1 - overlap)
    if step_x <= 0 or step_y <= 0:
        raise ValueError(f"overlap={overlap} leaves no forward progress per tile")

    xs = _axis_points(x_min, x_max, tile_w, step_x)
    ys = _axis_points(y_min, y_max, tile_h, step_y)

    points = []
    for i, y in enumerate(ys):
        row_xs = xs if i % 2 == 0 else list(reversed(xs))
        for x in row_xs:
            points.append((x, y, altitude))
    return points
