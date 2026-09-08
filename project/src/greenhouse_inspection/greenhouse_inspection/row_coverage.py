"""Plan map-frame camera poses and routes for crop-row imaging."""

import math

from greenhouse_inspection.free_space import (
    VENLO_COLUMN_WIDTH,
    VENLO_COLUMN_YS,
    X500_WIDTH,
    _free_values,
    _obstacle_lines,
    safe_route,
    segment_blocked,
)

# Raspberry Pi HQ camera with a 6 mm lens.
CAMERA_HFOV = 0.9652   # rad, 55.3 deg
CAMERA_VFOV = 0.7483   # rad, 42.9 deg

# Require a usable camera-to-face incidence angle.
MAX_OBLIQUITY = math.radians(60.0)

# Sensor pixels across the tilt plane, for ground-sample-distance reporting.
IMAGE_WIDTH_PX = 4056


def venlo_rows(path_width=2.0, row_pitch=1.6, canopy_thick=0.75, half_width=12.0):
    """Return greenhouse_venlo crop-row centre positions."""
    rows = []
    y = path_width / 2.0 + 0.8
    while y + canopy_thick / 2.0 < half_width - 1.0:
        rows += [-y, y]
        y += row_pitch
    return sorted(rows)


def face_positions(rows, canopy_thick=0.75):
    """Return the two outward-facing vertical sides of each row."""
    faces = []
    for cy in rows:
        faces.append((cy - canopy_thick / 2.0, -1.0))
        faces.append((cy + canopy_thick / 2.0, +1.0))
    return faces


def covered_span(cam_y, cam_z, tilt, face_y, normal_sign,
                 z_bottom=1.0, z_top=3.2, fov=CAMERA_VFOV,
                 max_obliquity=MAX_OBLIQUITY, samples=200):
    """Return the vertical face span visible at a usable incidence angle."""
    axis = (math.sin(tilt), -math.cos(tilt))       # (y, z) unit optical axis
    normal = (normal_sign, 0.0)                    # face normal, horizontal

    hits = []
    for i in range(samples + 1):
        z = z_bottom + (z_top - z_bottom) * i / samples
        dy, dz = face_y - cam_y, z - cam_z
        dist = math.hypot(dy, dz)
        if dist < 1e-6:
            continue
        ray = (dy / dist, dz / dist)

        # (a)+(b): angle away from the optical axis, in the tilt plane
        if ray[0] * axis[0] + ray[1] * axis[1] <= 0:
            continue
        off_axis = abs(math.atan2(ray[0] * axis[1] - ray[1] * axis[0],
                                  ray[0] * axis[0] + ray[1] * axis[1]))
        if off_axis > fov / 2.0:
            continue

        # incidence on the face itself.
        cos_inc = -(ray[0] * normal[0] + ray[1] * normal[1])
        if cos_inc <= 0 or math.acos(min(1.0, cos_inc)) > max_obliquity:
            continue
        hits.append(z)

    if not hits:
        return None
    return (min(hits), max(hits))


def coverage_fraction(cam_y, cam_z, tilt, face_y, normal_sign,
                      z_bottom=1.0, z_top=3.2, **kw):
    span = covered_span(cam_y, cam_z, tilt, face_y, normal_sign,
                        z_bottom=z_bottom, z_top=z_top, **kw)
    if span is None:
        return 0.0
    return (span[1] - span[0]) / (z_top - z_bottom)


def ground_sample_distance(distance, fov=CAMERA_HFOV, pixels=IMAGE_WIDTH_PX):
    """Metres per pixel at `distance`, before foreshortening."""
    return 2.0 * distance * math.tan(fov / 2.0) / pixels


def best_pose_for_face(face_y, normal_sign, aisle_lo, aisle_hi,
                       z_bottom=1.0, z_top=3.2,
                       z_options=None, tilt_options=None, **kw):
    """Search a small grid of for the pose covering most of a face, with the camera constrained."""
    cam_y = (aisle_lo + aisle_hi) / 2.0
    if z_options is None:
        z_options = [z_bottom + (z_top - z_bottom) * f
                     for f in (0.0, 0.25, 0.5, 0.75, 1.0)] + [z_top + 0.8]
    if tilt_options is None:
        tilt_options = [math.radians(d) for d in range(0, 91, 10)]

    best = (0.0, None)
    for z in z_options:
        for tilt in tilt_options:
            signed = tilt * (1.0 if face_y > cam_y else -1.0)
            frac = coverage_fraction(cam_y, z, signed, face_y, normal_sign,
                                     z_bottom=z_bottom, z_top=z_top, **kw)
            if frac > best[0]:
                best = (frac, {"y": cam_y, "z": z, "tilt": signed})
    return best


# Account for structural columns when checking aisle clearance.


def aisles(rows, canopy_thick=0.75, corridor_half=1.425, half_width=12.0):
    """Clear spans a vehicle could fly in, excluding the corridor."""
    out = []
    for a, b in zip(rows, rows[1:]):
        lo, hi = a + canopy_thick / 2.0, b - canopy_thick / 2.0
        if hi <= lo:
            continue
        # The central corridor is kept, not skipped.
        out.append((lo, hi))
    if rows:
        out.append((-half_width, rows[0] - canopy_thick / 2.0))
        out.append((rows[-1] + canopy_thick / 2.0, half_width))
    return sorted(out)


def passable_width(aisle, columns=VENLO_COLUMN_YS,
                   column_width=VENLO_COLUMN_WIDTH):
    """Widest continuous gap through an aisle once columns are accounted for."""
    lo, hi = aisle
    best = hi - lo
    for cy in columns:
        c_lo, c_hi = cy - column_width / 2.0, cy + column_width / 2.0
        if c_hi <= lo or c_lo >= hi:
            continue
        best = min(best, max(c_lo - lo, hi - c_hi))
    return max(0.0, best)


def passable_aisles(rows, drone_width=X500_WIDTH, **kw):
    """Split aisles into those a given vehicle fits through, and those it does not."""
    ok, blocked = [], []
    for a in aisles(rows, **kw):
        (ok if passable_width(a) >= drone_width else blocked).append(a)
    return ok, blocked


def max_drone_width(rows, **kw):
    """Return the widest vehicle that fits through every aisle."""
    spans = [passable_width(a) for a in aisles(rows, **kw)]
    return min(spans) if spans else 0.0


def unreachable_faces(rows, drone_width=X500_WIDTH, canopy_thick=0.75, **kw):
    """Faces with no passable aisle adjacent to them."""
    ok, blocked = passable_aisles(rows, drone_width, canopy_thick=canopy_thick, **kw)
    reachable = set()
    for lo, hi in ok:
        reachable.add(round(lo, 3))
        reachable.add(round(hi, 3))
    stranded = []
    for fy, sign in face_positions(rows, canopy_thick):
        if round(fy, 3) not in reachable:
            stranded.append((fy, sign))
    return stranded


# --- route generation -------------------------------------------------------
def inspection_route(rows=None, drone_width=X500_WIDTH, altitude=2.0,
                     x_start=-4.0, x_end=22.0, spacing=1.0, **kw):
    """Serpentine route down every aisle the vehicle actually fits through."""
    if rows is None:
        rows = venlo_rows()
    usable, _ = passable_aisles(rows, drone_width, **kw)

    route = []
    for i, (lo, hi) in enumerate(usable):
        y = (lo + hi) / 2.0
        forward = (i % 2 == 0)
        xs = [x_start + j * spacing
              for j in range(int((x_end - x_start) / spacing) + 1)]
        if not forward:
            xs.reverse()
        for x in xs:
            route.append((x, y, altitude, 0.0 if forward else math.pi))
    return safe_route(route, rows)


def route_length(route):
    """Total path length in metres, for battery/time estimates."""
    total = 0.0
    for a, b in zip(route, route[1:]):
        total += math.dist(a[:3], b[:3])
    return total


def corridor_route(rows=None, altitude=2.0, x_start=-4.0, x_end=22.0,
                   spacing=1.0, target_distance=345.0, **kw):
    """Back-and-forth passes down the central corridor, for a SLAM comparison at matched distance against a route."""
    if rows is None:
        rows = venlo_rows()
    xs = [x_start + j * spacing for j in range(int((x_end - x_start) / spacing) + 1)]
    one_way = (len(xs) - 1) * spacing

    route, forward, travelled = [], True, 0.0
    while travelled < target_distance:
        leg = xs if forward else list(reversed(xs))
        yaw = 0.0 if forward else math.pi
        for x in leg:
            route.append((x, 0.0, altitude, yaw))
        travelled += one_way
        forward = not forward
    return safe_route(route, rows)


# over-row imaging route ------------------------------------------------- Croptimus already runs its detection models on fixed top-mounted downward cameras, so the drone's.
OVER_ROW_ALTITUDE = 3.92   # footprint 0.73 m vs a 0.75 m row; GSD 0.185 mm/px


def imaging_altitude(canopy_top=3.2, canopy_thick=0.75, fov=CAMERA_HFOV):
    """Height at which the camera footprint matches the row width."""
    return canopy_top + canopy_thick / (2.0 * math.tan(fov / 2.0))


# Step size, local-probe distance, and search bound for _clear_endpoint's inward walk.
ENDPOINT_TRIM_STEP = 0.1
ENDPOINT_PROBE_DIST = 1.0
ENDPOINT_TRIM_MAX = 4.0


def _clear_endpoint(interior_x, y, target_x, altitude, boxes,
                    step=ENDPOINT_TRIM_STEP, probe_dist=ENDPOINT_PROBE_DIST,
                    max_trim=ENDPOINT_TRIM_MAX):
    """Walk target_x back toward interior_x until a short local probe into it clears every real sensed box."""
    if boxes is None:
        return target_x
    direction = 1.0 if target_x >= interior_x else -1.0
    span = abs(target_x - interior_x)
    trimmed_by = 0.0
    while trimmed_by <= max_trim and trimmed_by <= span:
        trimmed = target_x - direction * trimmed_by
        probe_from = trimmed - direction * min(probe_dist, span - trimmed_by)
        if segment_blocked((probe_from, y, altitude), (trimmed, y, altitude),
                           boxes) is None:
            return trimmed
        trimmed_by += step
    return interior_x


# Fine local offsets tried before the wider _obstacle_lines candidates.
_LOCAL_Y_OFFSETS = [0.1 * n for n in range(1, 11)]  # 0.1 .. 1.0 m, both signs


def _row_clear_at(prev_x, x, cy, altitude, boxes):
    """Does the two-segment path prev_x -> at constant cy, via a lateral hop at prev_x to get there."""
    lateral = (prev_x[0], cy, altitude)
    cruise_end = (x, cy, altitude)
    return (segment_blocked(prev_x, lateral, boxes) is None and
            segment_blocked(lateral, cruise_end, boxes) is None)


def _nudge_row_waypoints(xs, y, altitude, boxes, y_lines):
    """Each x in xs paired with the row's own y, or a small nearby offset if that exact."""
    if boxes is None:
        return [(x, y) for x in xs]
    out = []
    prev = (xs[0], y, altitude)
    for x in xs:
        candidates = ([y] + [y + o for o in _LOCAL_Y_OFFSETS]
                      + [y - o for o in _LOCAL_Y_OFFSETS]
                      + _free_values(y_lines, y))
        for cy in candidates:
            if _row_clear_at(prev, x, cy, altitude, boxes):
                out.append((x, cy))
                prev = (x, cy, altitude)
                break
        else:
            prev = (x, y, altitude)  # keep walking from the row's true line
    return out


def over_row_route(rows=None, altitude=None, x_start=-4.0, x_end=22.0,
                   waypoint_spacing=5.0, extents=None, boxes=None):
    """Passes directly above every crop row, for top-down imaging."""
    if rows is None:
        rows = venlo_rows()
    if altitude is None:
        altitude = imaging_altitude()
    extents = extents or {}

    route = []
    for i, y in enumerate(sorted(rows)):
        row_lo, row_hi = extents.get(round(y, 3), (x_start, x_end))
        # Trim each end back toward the row's own midpoint if it's too close to something real.
        mid = (row_lo + row_hi) / 2.0
        row_lo = _clear_endpoint(mid, y, row_lo, altitude, boxes)
        row_hi = _clear_endpoint(mid, y, row_hi, altitude, boxes)
        forward = (i % 2 == 0)
        lo, hi = (row_lo, row_hi) if forward else (row_hi, row_lo)
        step = waypoint_spacing if forward else -waypoint_spacing
        n = int(abs(hi - lo) / waypoint_spacing)
        xs = [lo + j * step for j in range(n + 1)]
        if abs(xs[-1] - hi) > 1e-6:
            xs.append(hi)
        y_lines = _obstacle_lines(boxes, 1) or list(VENLO_COLUMN_YS)
        for x, wp_y in _nudge_row_waypoints(xs, y, altitude, boxes, y_lines):
            route.append((x, wp_y, altitude, 0.0 if forward else math.pi))
    return safe_route(route, rows, boxes=boxes)


def row_pass_legs(route, rows=None, tol=0.05):
    """Which legs of a route are an imaging pass along a crop row."""
    if rows is None:
        rows = venlo_rows()
    # _nudge_row_waypoints deliberately moves a waypoint by its first 0.10 m local offset when a mapped obstacle is too.
    capture_row_tol = max(tol, _LOCAL_Y_OFFSETS[0] + 0.05)
    flags = [False]
    for a, b in zip(route, route[1:]):
        flags.append(abs(a[1] - b[1]) <= capture_row_tol
                     and abs(a[0] - b[0]) > tol
                     and any(abs(a[1] - r) <= capture_row_tol
                             and abs(b[1] - r) <= capture_row_tol
                             for r in rows))
    return flags


def column_clearance(rows=None, drone_width=X500_WIDTH, columns=VENLO_COLUMN_YS):
    """Lateral gap between a vehicle flying above each row and the nearest column."""
    if rows is None:
        rows = venlo_rows()
    out = {}
    for y in rows:
        nearest = min(columns, key=lambda c: abs(c - y))
        out[y] = abs(nearest - y) - drone_width / 2.0 - VENLO_COLUMN_WIDTH / 2.0
    return out
