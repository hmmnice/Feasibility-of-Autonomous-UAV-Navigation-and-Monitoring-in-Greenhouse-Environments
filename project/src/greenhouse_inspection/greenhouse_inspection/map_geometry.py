"""Derive greenhouse route geometry from a saved map-frame point cloud."""

from collections import namedtuple

import numpy as np

from greenhouse_inspection.free_space import (
    DRONE_HALF,
    DRONE_HALF_Z,
    TRACKING_ALLOWANCE,
)

# LiDAR extraction thresholds, independent of the greenhouse layout.

# Voxel edge and histogram-bin size.
CELL = 0.10

# Required local density contrast for structure.
CONTRAST = 1.3

# Local background percentile and window width.
BG_PCT = 10.0
BG_WINDOW = 4.0

# Merge narrow gaps within a crop hedge.
CLOSE_GAP = 0.30

# Minimum crop-row thickness.
MIN_THICK = 0.40

# Required mapped margin on both sides of a crop row.
EDGE_MARGIN = 0.50

# Ignore returns near the floor.
FLOOR_CLEAR = 0.30

# Vertical headroom reserved for flight.
HEADROOM = 1.50

# Required low-level share for a crop band.
CROP_FRAC = 0.25

# Minimum voxel count per height slice.
MIN_CELLS = 5

# Required occupied height share for a structural post.
POST_FILL = 0.5

# Trim row-end outliers and split long unmapped gaps.
TRIM = 1.0
X_GAP = 1.0

# Keep detected row spans clear of end walls.
WALL_SAFETY_TRIM = 1.0

Geometry = namedtuple("Geometry", "rows canopy_top extents boxes")


def _runs(mask, close=0):
    """Return true-index runs, optionally joining small gaps."""
    edge = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    out = list(zip(edge[::2], edge[1::2]))
    if close:
        merged = []
        for run in out:
            if merged and run[0] - merged[-1][1] < close:
                merged[-1] = (merged[-1][0], run[1])
            else:
                merged.append(run)
        out = merged
    return out


def _longest(mask, close=0):
    """The one run of `mask` that is actually the thing being looked for."""
    return max(_runs(mask, close), key=lambda r: r[1] - r[0])


def _voxels(cloud, cell):
    """Return one representative point per occupied voxel."""
    return (np.unique(np.floor(np.asarray(cloud, float) / cell), axis=0) + 0.5) * cell


def _floor(z, cell=CELL):
    """Estimate floor height from the first dense horizontal slice."""
    edges = np.arange(z.min(), z.max() + cell, cell)
    n, _ = np.histogram(z, bins=edges)
    return float(edges[np.argmax(n >= 0.25 * n.max())])


def _bands(y, cell=CELL, contrast=CONTRAST):
    """Return dense cross-row bands that may represent crop rows."""
    edges = np.arange(y.min(), y.max() + cell, cell)
    n, _ = np.histogram(y, bins=edges)

    # Use local background because map coverage is uneven.
    w = int(BG_WINDOW / cell) | 1
    pad = np.pad(n.astype(float), w // 2, mode="edge")
    bg = np.percentile(np.lib.stride_tricks.sliding_window_view(pad, w), BG_PCT, axis=-1)
    occ = n > contrast * np.maximum(bg, 1.0)

    # Ignore stray returns beyond robust map bounds.
    lo_edge, hi_edge = np.percentile(y, (0.5, 99.5))
    return [(edges[a], edges[b]) for a, b in _runs(occ, int(CLOSE_GAP / cell))
            if (b - a) * cell >= MIN_THICK
            and edges[a] > lo_edge + EDGE_MARGIN
            and edges[b] < hi_edge - EDGE_MARGIN]


def _canopy_top(v, bands, cell=CELL, contrast=CONTRAST):
    """Estimate canopy top from the highest row-to-aisle contrast band."""
    y, z = v[:, 1], v[:, 2]
    inrow = _in_bands(y, bands)
    w_row = sum(hi - lo for lo, hi in bands)
    w_gap = max(y.max() - y.min() - w_row, cell)
    edges = np.arange(z.min(), z.max() + cell, cell)
    a, _ = np.histogram(z[inrow], bins=edges)
    b, _ = np.histogram(z[~inrow], bins=edges)
    hit = (a >= MIN_CELLS) & (a / w_row > contrast * b / w_gap)
    if not hit.any():
        return float(edges[0])      # no canopy anywhere; geometry() will say so
    # Prefer the longest band to isolated roof returns.
    return float(edges[_longest(hit, 2)[1]])


def _in_bands(y, bands):
    m = np.zeros(len(y), bool)
    for lo, hi in bands:
        m |= (y >= lo) & (y < hi)
    return m


def _crop_bands(v, bands, canopy_top):
    """The candidate bands that actually hold crop."""
    y, low = v[:, 1], v[:, 2] < canopy_top
    return [(lo, hi) for lo, hi in bands
            if (low & (y >= lo) & (y < hi)).sum()
            > CROP_FRAC * ((y >= lo) & (y < hi)).sum()]


def _span(x, cell, safety_trim=WALL_SAFETY_TRIM):
    """The stretch of x this row is actually crop over."""
    edges = np.arange(x.min(), x.max() + cell, cell)
    n, _ = np.histogram(x, bins=edges)
    a, b = _longest(n > 0, int(X_GAP / cell))
    lo, hi = np.percentile(x[(x >= edges[a]) & (x < edges[b])],
                           (TRIM, 100.0 - TRIM))
    trim = safety_trim + DRONE_HALF + TRACKING_ALLOWANCE
    lo, hi = lo + trim, hi - trim
    if hi <= lo:
        mid = (lo + hi) / 2.0
        lo, hi = mid, mid
    return float(lo), float(hi)


def geometry(cloud, cell=CELL, contrast=CONTRAST, headroom=HEADROOM,
             canopy_ceiling=3.5):
    """Crop rows, canopy top, per-row x span and collision boxes, from a cloud."""
    v = _voxels(cloud, cell)
    floor = _floor(v[:, 2], cell)
    v = v[v[:, 2] > floor + FLOOR_CLEAR]

    bands = _bands(v[:, 1], cell, contrast)
    canopy_top = canopy_ceiling
    bands = _crop_bands(v, bands, canopy_top)
    if not bands:
        # Loudly, because everything downstream is derived from the rows.
        raise ValueError("no crop rows in this cloud: re-fly the mapping pass "
                         "lower and down the aisles")

    crop = v[v[:, 2] < canopy_top]
    rows, extents, boxes = [], [], []
    for lo, hi in bands:
        in_band = (crop[:, 1] >= lo) & (crop[:, 1] < hi)
        x0, x1 = _span(crop[in_band, 0], cell)
        # Centre of the sub-canopy surface, not the middle of the band.
        rows.append(float(crop[in_band, 1].mean()))
        extents.append((float(x0), float(x1)))
        # Margins mirror free_space.obstacles: geometric only against the canopy, because an aisle has no room to give.
        boxes.append((float(x0) - DRONE_HALF, float(x1) + DRONE_HALF,
                      lo - DRONE_HALF, hi + DRONE_HALF,
                      floor, canopy_top + DRONE_HALF_Z))

    boxes += _tall_boxes(v, floor, canopy_top, cell, headroom)
    return Geometry(rows, canopy_top, extents, boxes)


def _tall_boxes(v, floor, canopy_top, cell, headroom):
    """Everything that is not crop and still reaches into the flight slab."""
    lo, hi = canopy_top + DRONE_HALF_Z, canopy_top + headroom
    hit = v[(v[:, 2] > lo) & (v[:, 2] < hi)]
    if not len(hit):
        return []
    g = np.floor(hit[:, :2] / cell).astype(np.int64)
    org = g.min(axis=0)
    n = np.zeros(tuple(g.max(axis=0) - org + 1), int)
    np.add.at(n, tuple((g - org).T), 1)

    fill = max(2, int(POST_FILL * (hi - lo) / cell))
    m = DRONE_HALF + TRACKING_ALLOWANCE
    return [((org[0] + a) * cell - m, (org[0] + b) * cell + m,
             (org[1] + j) * cell - m, (org[1] + j + 1) * cell + m,
             floor, hi)
            for j, col in enumerate(n.T)
            for a, b in _runs(col >= fill)]
