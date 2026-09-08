"""Define collision-free map-frame routes for the greenhouse vehicle."""

import math

# --- the world's collision geometry, from make_greenhouse_venlo.py ----------
VENLO_COLUMN_XS = (-6.0, -2.0, 2.0, 6.0, 10.0, 14.0, 18.0, 22.0)   # truss pitch
VENLO_COLUMN_YS = (-12.0, -4.0, 4.0, 12.0)      # every other gutter line
VENLO_COLUMN_WIDTH = 0.10
EAVE = 6.7                 # columns run the full height; nothing flies over them

CANOPY_X0, CANOPY_X1 = -4.0, 22.0
CANOPY_THICK = 0.75
CANOPY_BOTTOM, CANOPY_TOP = 1.0, 3.2
CANOPY_JITTER_Y = 0.07     # per-segment sideways wobble
CANOPY_JITTER_H = 0.18     # ...and how much taller a segment can be

# --- the vehicle ------------------------------------------------------------
X500_WIDTH = 0.60          # rotor hubs at +/-0.174 m plus 10" props
DRONE_HALF = X500_WIDTH / 2.0
DRONE_HALF_Z = 0.15        # much thinner than it is wide

# Extra margin for position-hold error near structural columns.
TRACKING_ALLOWANCE = 0.15

# Transit height above the tallest inflated canopy.
TRANSIT_Z = CANOPY_TOP + CANOPY_JITTER_H + DRONE_HALF_Z + 0.45


def obstacles(rows, column_margin=DRONE_HALF + TRACKING_ALLOWANCE,
              canopy_margin=DRONE_HALF):
    """Return vehicle-inflated crop and column collision boxes."""
    boxes = []
    h = VENLO_COLUMN_WIDTH / 2.0 + column_margin
    for x in VENLO_COLUMN_XS:
        for y in VENLO_COLUMN_YS:
            boxes.append((x - h, x + h, y - h, y + h, 0.0, EAVE))
    hy = CANOPY_THICK / 2.0 + CANOPY_JITTER_Y + canopy_margin
    for y in rows:
        boxes.append((CANOPY_X0 - canopy_margin, CANOPY_X1 + canopy_margin,
                      y - hy, y + hy,
                      CANOPY_BOTTOM - DRONE_HALF_Z,
                      CANOPY_TOP + CANOPY_JITTER_H + DRONE_HALF_Z))
    return boxes


def segment_blocked(p, q, boxes):
    """Return the first collision box entered by a straight segment."""
    for box in boxes:
        if _hits(p, q, box):
            return box
    return None


def _hits(p, q, box):
    t_lo, t_hi = 0.0, 1.0
    for i in range(3):
        lo, hi = box[2 * i], box[2 * i + 1]
        d = q[i] - p[i]
        if abs(d) < 1e-12:
            if not lo <= p[i] <= hi:
                return False
            continue
        a, b = (lo - p[i]) / d, (hi - p[i]) / d
        if a > b:
            a, b = b, a
        t_lo, t_hi = max(t_lo, a), min(t_hi, b)
        if t_lo > t_hi:
            return False
    return True


def _free_values(lines, near):
    """Midpoints between consecutive obstacle lines, nearest `near` first."""
    return sorted(((a + b) / 2.0 for a, b in zip(lines, lines[1:])),
                  key=lambda m: abs(m - near))


def _obstacle_lines(boxes, axis):
    """Return deduplicated obstacle edges along one horizontal axis."""
    if not boxes:
        return []
    lo, hi = axis * 2, axis * 2 + 1
    edges = sorted({round(b[lo], 1) for b in boxes} | {round(b[hi], 1) for b in boxes})
    out = []
    for e in edges:
        if not out or e - out[-1] > 0.3:
            out.append(e)
    return out


def _detours(p, q, z):
    """Yield simple column-grid detour candidates at a given height."""
    yield []
    yield [(q[0], p[1], z)]
    yield [(p[0], q[1], z)]
    for x in _free_values(VENLO_COLUMN_XS, p[0]):
        yield [(x, p[1], z), (x, q[1], z)]
    for y in _free_values(VENLO_COLUMN_YS, p[1]):
        yield [(p[0], y, z), (q[0], y, z)]


def _dedupe(points):
    out = []
    for pt in points:
        if not out or math.dist(out[-1], pt) > 1e-9:
            out.append(pt)
    return out


def safe_leg(p, q, boxes, transit_z=TRANSIT_Z):
    """Waypoints to insert between p and q so every segment of the leg is clear, or None if."""
    flat = max(p[2], q[2])
    for z in dict.fromkeys([flat, max(flat, transit_z)]):
        for mids in _detours(p, q, z):
            chain = _dedupe([tuple(p[:3]), (p[0], p[1], z)] + mids
                            + [(q[0], q[1], z), tuple(q[:3])])
            if all(segment_blocked(a, b, boxes) is None
                   for a, b in zip(chain, chain[1:])):
                return chain[1:-1]
    return None


SUBDIVIDE_STEPS = 10  # for _safe_leg_subdivided's greedy walk


def _safe_leg_subdivided(p, q, boxes, transit_z, steps=SUBDIVIDE_STEPS):
    """safe_leg, and if that finds no detour, walk from p to q in steps short hops, ducking sideways."""
    mids = safe_leg(p, q, boxes, transit_z)
    if mids is not None:
        return mids

    out = []
    here = p
    y_lines = _obstacle_lines(boxes, 1) or list(VENLO_COLUMN_YS)
    for i in range(1, steps + 1):
        frac = i / steps
        target = q if i == steps else tuple(
            a + frac * (b - a) for a, b in zip(p[:3], q[:3]))
        hop = safe_leg(here, target, boxes, transit_z)
        if hop is not None:
            out.extend(hop)
            if i < steps:
                out.append(target)
            here = target
            continue
        if i == steps:
            return None  # q itself unreachable -- not this function's problem to fix
        for y in _free_values(y_lines, target[1])[:3]:
            nudged = (target[0], y, target[2])
            hop = safe_leg(here, nudged, boxes, transit_z)
            if hop is not None:
                out.extend(hop)
                out.append(nudged)
                here = nudged
                break
        else:
            return None
    return out


def safe_route(route, rows, start=(0.0, 0.0, 0.0), transit_z=TRANSIT_Z,
               boxes=None, **kw):
    """route with legs inserted wherever a straight line would hit something."""
    if boxes is None:
        boxes = obstacles(rows, **kw)
    out, here = [], tuple(start)[:3]
    for wp in route:
        mids = _safe_leg_subdivided(here, wp[:3], boxes, transit_z)
        if mids is None:
            raise ValueError(f"no obstacle-free path from {here} to "
                             f"{tuple(wp[:3])}")
        out.extend(m + tuple(wp[3:]) for m in mids)
        out.append(tuple(wp))
        here = tuple(wp[:3])
    return out


def route_violations(route, rows, start=(0.0, 0.0, 0.0), **kw):
    """Every leg that enters an obstacle, as."""
    boxes = obstacles(rows, **kw)
    bad, here = [], tuple(start)[:3]
    for wp in route:
        box = segment_blocked(here, wp[:3], boxes)
        if box is not None:
            bad.append((here, tuple(wp[:3]), box))
        here = tuple(wp[:3])
    return bad
