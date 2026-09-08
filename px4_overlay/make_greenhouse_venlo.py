#!/usr/bin/env python3
"""Generate the Venlo greenhouse SDF used by the thesis experiments."""

import math
import os
import random
import re
from pathlib import Path

import trimesh

WORLD_RESOURCE_DIR = (
    Path(__file__).parent / "Tools/simulation/gz/worlds")
OUT = Path(os.environ.get(
    "GREENHOUSE_WORLD_OUT",
    WORLD_RESOURCE_DIR / "greenhouse_venlo.sdf"))

# house ------------------------------------------------------------------ The footprint is kept small enough for practical simulation runs.
BAY = 4.0            # Venlo span, gutter to gutter
N_BAYS = 6           # -> 24 m wide
TRUSS = 4.0          # column pitch along x
LENGTH = 30.0
X_MIN = -6.0         # drone spawns at (0,0) -> inside, 6 m from the end wall
# The vertical dimensions match the greenhouse dimensions used in the study.
EAVE = 6.7
ROOF_PITCH = math.radians(22.0)

# --- crop -------------------------------------------------------------------
ROW_PITCH = 1.6
PATH_WIDTH = 2.0     # central concrete corridor (kept clear for the UAV)
CROP_X0, CROP_X1 = X_MIN + 2.0, X_MIN + LENGTH - 2.0
GUTTER_Z = 0.65
# Crop and wire dimensions used by the static planning map.
CANOPY_BOTTOM, CANOPY_TOP, CANOPY_THICK = 1.0, 3.2, 0.75
WIRE_Z = 3.4
RAIL_GAUGE = 0.55
STEM_PITCH = 0.5     # per stem line
STEM_LINES = (-0.25, 0.25)
SEG = 4.0            # one model per 4 m of row -- raise to thin the world out
HOOK_EVERY = 1       # tomahook spool every Nth plant; 2 halves the visual count
JITTER_H = 0.18      # per-segment canopy wobble -- a perfectly flat hedge gives
JITTER_Y = 0.07      # LiDAR unrealistically clean returns

# Tomato meshes were generated from the Apache-2.0 AOC tomato farm assets.
AOC_SUBMESH_COLOUR = {
    "Branch1": (0.32, 0.24, 0.13, 1),
    "Leaf1": (0.18, 0.46, 0.15, 1), "Leaf2": (0.24, 0.52, 0.18, 1),
    "Fruit1": (0.95, 0.10, 0.05, 1), "Fruit2": (0.90, 0.20, 0.03, 1),
    "Fruit3": (1.00, 0.30, 0.05, 1), "Fruit4": (0.85, 0.45, 0.02, 1),
    "Blossom1": (1.00, 0.92, 0.15, 1), "Blossom2": (1.00, 0.95, 0.20, 1),
    "Blossom3": (0.98, 0.90, 0.12, 1),
}
# The selected target receives a cyan overlay for post-capture measurement.
TARGET_PLANT_ROW = int(os.environ.get("HIGHLIGHT_PLANT_ROW", "6"))
TARGET_PLANT_INDEX = int(os.environ.get("HIGHLIGHT_PLANT_INDEX", "7"))


def highlighted_plants_from_environment():
    """Return one or more ``(row, plant)`` pairs for evaluation overlays."""
    specification = os.environ.get("HIGHLIGHT_PLANTS", "").strip()
    if not specification:
        return ((TARGET_PLANT_ROW, TARGET_PLANT_INDEX),)
    pairs = []
    for item in specification.split(","):
        values = item.strip().split(":")
        if len(values) != 2:
            raise ValueError(
                "HIGHLIGHT_PLANTS must use comma-separated row:plant pairs")
        pair = (int(values[0]), int(values[1]))
        if pair not in pairs:
            pairs.append(pair)
    if not pairs:
        raise ValueError("HIGHLIGHT_PLANTS contains no targets")
    return tuple(pairs)


TARGET_PLANTS = highlighted_plants_from_environment()
HIGHLIGHT_COLOUR = (0.0, 1.0, 1.0, 0.38)
HIGHLIGHT_TRANSPARENCY = 0.58
HIGHLIGHT_SCALE = 1.008
AOC_VARIANT_SEEDS = [11, 22, 33, 44, 55, 66] + list(range(100, 130))
PLANT_SPACING = 1.2
PLANT_YAW_JITTER = 0.4
PLANT_RNG_SEED = 42
PLANT_SCALE = 1.25

# --- colours ----------------------------------------------------------------
STEEL = (0.62, 0.63, 0.65, 1)
GALV = (0.75, 0.76, 0.78, 1)
GLASS = (0.86, 0.91, 0.94, 1)
GLASS_T = 0.62
LEAF = (0.16, 0.40, 0.14, 1)
LEAF_LIT = (0.22, 0.48, 0.17, 1)
STEM = (0.30, 0.44, 0.18, 1)
WHITE = (0.93, 0.93, 0.91, 1)
ROCKWOOL = (0.80, 0.76, 0.66, 1)
YELLOW = (0.95, 0.85, 0.05, 1)
CONCRETE = (0.72, 0.72, 0.70, 1)
FLOOR = (0.84, 0.84, 0.82, 1)
BLACK = (0.12, 0.12, 0.13, 1)


def mat(c, transparency=0.0):
    r, g, b, a = c
    return (f"<material><ambient>{r} {g} {b} {a}</ambient>"
            f"<diffuse>{r} {g} {b} {a}</diffuse></material>"
            f"<transparency>{transparency}</transparency>")


def pose(x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
    return f"<pose>{x:.4f} {y:.4f} {z:.4f} {roll:.5f} {pitch:.5f} {yaw:.5f}</pose>"


def box_geo(sx, sy, sz):
    return f"<geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>"


def cyl_geo(r, l):
    return f"<geometry><cylinder><radius>{r:.4f}</radius><length>{l:.4f}</length></cylinder></geometry>"


def vis(name, geo, p, c, transparency=0.0):
    return f'<visual name="{name}">{p}{geo}{mat(c, transparency)}</visual>'


def col(name, geo, p):
    # names must be unique across visuals AND collisions within one link
    return f'<collision name="col_{name}">{p}{geo}</collision>'


# Outward-wound box faces for OBJ export.
OBJ_QUADS = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (3, 7, 6, 2), (0, 4, 7, 3), (1, 2, 6, 5)]


def obj_box(verts, faces, cx, cy, cz, sx, sy, sz):
    base = len(verts)
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    for a, b, c in [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, hy, -hz), (-hx, hy, -hz),
                    (-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]:
        verts.append((cx + a, cy + b, cz + c))
    faces.extend(tuple(base + i + 1 for i in q) for q in OBJ_QUADS)


def write_obj(path, verts, faces):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"v {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in verts)
    body += "".join("f " + " ".join(str(i) for i in f) + "\n" for f in faces)
    path.write_text(body)


def mesh_vis(name, uri, c, scale=1.0):
    return (f'<visual name="{name}"><geometry><mesh><uri>{uri}</uri>'
            f'<scale>{scale} {scale} {scale}</scale></mesh>'
            f'</geometry>{mat(c)}</visual>')


def submesh_vis(name, uri, submesh, colour, scale):
    """Create one flat-coloured plant submesh visual."""
    r, g, b, a = colour
    sx, sy, sz = scale
    return (f'<visual name="{name}"><geometry><mesh><uri>{uri}</uri>'
            f'<submesh><name>{submesh}</name></submesh>'
            f'<scale>{sx:.4f} {sy:.4f} {sz:.4f}</scale></mesh></geometry>'
            f'<material><ambient>{r} {g} {b} {a}</ambient>'
            f'<diffuse>{r} {g} {b} {a}</diffuse></material></visual>')


def highlight_submesh_vis(name, uri, submesh, scale):
    """Translucent cyan skin over one normally coloured plant submesh."""
    r, g, b, a = HIGHLIGHT_COLOUR
    sx, sy, sz = (float(value) * HIGHLIGHT_SCALE for value in scale)
    return (f'<visual name="highlight_{name}">'
            f'<geometry><mesh><uri>{uri}</uri>'
            f'<submesh><name>{submesh}</name></submesh>'
            f'<scale>{sx:.4f} {sy:.4f} {sz:.4f}</scale></mesh></geometry>'
            f'<material><ambient>{r} {g} {b} {a}</ambient>'
            f'<diffuse>{r} {g} {b} {a}</diffuse>'
            f'<emissive>0.0 0.18 0.18 {a}</emissive></material>'
            f'<transparency>{HIGHLIGHT_TRANSPARENCY}</transparency>'
            f'<cast_shadows>false</cast_shadows></visual>')


def load_aoc_variants(mesh_dir, seeds, known_submeshes):
    """Load the available plant variants and their bounds."""
    variants = []
    for seed in seeds:
        rel = f"meshes/aoc_tomato/variants/tomato_{seed}.dae"
        dae = mesh_dir / rel
        text = dae.read_text(errors="ignore")
        names = sorted(set(re.findall(r'<node id="[^"]*" name="([^"]+)"', text)))
        submeshes = [n for n in names if n in known_submeshes]
        geo = trimesh.load(str(dae), force="scene").to_geometry()
        variants.append({
            "uri": rel, "submeshes": submeshes,
            "base_z": float(geo.bounds[0][2]), "extents": geo.extents,
        })
    return variants


def model(name, x, y, z, parts):
    """Create one static model containing the supplied parts."""
    return (f'\n    <model name="{name}"><static>true</static>{pose(x, y, z)}'
            f'<link name="link">{"".join(parts)}</link></model>')


# Cylinders are z-axis in SDF; these rotate them onto x / y.
ALONG_X = (0.0, math.pi / 2, 0.0)
ALONG_Y = (math.pi / 2, 0.0, 0.0)

out = []

AOC_PLANTS = load_aoc_variants(
    WORLD_RESOURCE_DIR, AOC_VARIANT_SEEDS, AOC_SUBMESH_COLOUR)
# Maximum lateral extent after applying the allowed yaw jitter.
AOC_MAX_EXTENT_Y = 2 * max(
    v["extents"][0] / 2 * math.sin(PLANT_YAW_JITTER) + v["extents"][1] / 2 * math.cos(PLANT_YAW_JITTER)
    for v in AOC_PLANTS
) * PLANT_SCALE

# Floor: white concrete, plus the central corridor the UAV flies down.
half_w = N_BAYS * BAY / 2.0
x_mid = X_MIN + LENGTH / 2.0
out.append(model("floor", x_mid, 0, 0.005, [
    vis("v", box_geo(LENGTH, 2 * half_w, 0.01), pose(0, 0, 0), FLOOR),
]))
out.append(model("main_path", x_mid, 0, 0.02, [
    vis("v", box_geo(LENGTH, PATH_WIDTH, 0.02), pose(0, 0, 0), CONCRETE),
]))

# Flat visual landing pad at the launch position.
DOCK_X, DOCK_Y = 0.0, 0.0
DOCK_SIZE = 1.2
out.append(model("charging_dock", DOCK_X, DOCK_Y, 0.04, [
    vis("pad", box_geo(DOCK_SIZE, DOCK_SIZE, 0.04), pose(0, 0, 0), BLACK),
    # High-contrast landing marker.
    vis("markx", box_geo(DOCK_SIZE * 0.75, 0.10, 0.01), pose(0, 0, 0.025), WHITE),
    vis("marky", box_geo(0.10, DOCK_SIZE * 0.75, 0.01), pose(0, 0, 0.025), WHITE),
    # Cabinet placed away from the take-off and landing path.
    vis("cabinet", box_geo(0.35, 0.30, 0.55), pose(0, DOCK_SIZE / 2 + 2.0, 0.25), GALV),
]))

# Structure: columns on every other gutter line, trusses across, gutters, ridges, glass roof panes, walls.
gutter_ys = [-half_w + i * BAY for i in range(N_BAYS + 1)]
column_ys = gutter_ys[::2]                      # -20, -12, -4, 4, 12, 20 -> path is clear
truss_xs = [X_MIN + i * TRUSS for i in range(int(LENGTH / TRUSS) + 1)]

for i, x in enumerate(truss_xs):
    parts = []
    for y in column_ys:
        parts.append(vis(f"c{y:.0f}", box_geo(0.10, 0.10, EAVE), pose(0, y, EAVE / 2), STEEL))
        parts.append(col(f"c{y:.0f}", box_geo(0.10, 0.10, EAVE), pose(0, y, EAVE / 2)))
    # lattice truss carrying the intermediate gutters
    parts.append(vis("truss", box_geo(0.12, 2 * half_w, 0.32), pose(0, 0, EAVE - 0.30), STEEL))
    out.append(model(f"frame_{i}", x, 0, 0, parts))

# Gutters (the classic Venlo look: many small ridges, one gutter every 4 m).
for i, y in enumerate(gutter_ys):
    out.append(model(f"gutter_{i}", x_mid, y, EAVE, [
        vis("v", box_geo(LENGTH, 0.22, 0.14), pose(0, 0, 0), GALV),
    ]))

# Roof: two glass panes per bay meeting at a small ridge.
rise = (BAY / 2.0) * math.tan(ROOF_PITCH)
slope = math.hypot(BAY / 2.0, rise)
for i in range(N_BAYS):
    y_ridge = gutter_ys[i] + BAY / 2.0
    out.append(model(f"ridge_{i}", x_mid, y_ridge, EAVE + rise, [
        vis("v", box_geo(LENGTH, 0.09, 0.09), pose(0, 0, 0), GALV),
    ]))
    for sign in (-1, 1):
        out.append(model(f"roof_{i}_{'p' if sign > 0 else 'm'}",
                         x_mid, y_ridge + sign * BAY / 4.0, EAVE + rise / 2.0, [
            vis("v", box_geo(LENGTH, slope, 0.015), pose(0, 0, 0), GLASS, GLASS_T),
        ]))

# Glass walls use slightly thicker collision boxes than their visual panes.
WALL_COLLISION_THICK = 0.15
for label, x_end in (("front", X_MIN), ("back", X_MIN + LENGTH)):
    out.append(model(f"endwall_{label}", x_end, 0, EAVE / 2, [
        vis("glass", box_geo(0.02, 2 * half_w, EAVE), pose(0, 0, 0), GLASS, GLASS_T),
        col("glass", box_geo(WALL_COLLISION_THICK, 2 * half_w, EAVE), pose(0, 0, 0)),
    ] + [
        vis(f"mullion{y:.0f}", box_geo(0.06, 0.06, EAVE), pose(0, y, 0), STEEL)
        for y in [-half_w + k * 2.0 for k in range(int(2 * half_w / 2.0) + 1)]
    ]))
for label, y_end in (("left", half_w), ("right", -half_w)):
    out.append(model(f"sidewall_{label}", x_mid, y_end, EAVE / 2, [
        vis("glass", box_geo(LENGTH, 0.02, EAVE), pose(0, 0, 0), GLASS, GLASS_T),
        col("glass", box_geo(LENGTH, WALL_COLLISION_THICK, EAVE), pose(0, 0, 0)),
    ]))

# Extraction fans in the end wall + circulation fans under the gutters.
for k, frac in enumerate([-0.7, -0.25, 0.25, 0.7]):
    out.append(model(f"fan_endwall_{k}", X_MIN + 0.25, frac * half_w, EAVE * 0.77, [
        vis("ring", cyl_geo(0.62, 0.30), pose(0, 0, 0, *ALONG_X), BLACK),
        vis("hub", cyl_geo(0.16, 0.36), pose(0, 0, 0, *ALONG_X), STEEL),
    ]))
for k, xfrac in enumerate([0.2, 0.5, 0.8]):
    x = X_MIN + xfrac * LENGTH
    for y in (-half_w * 0.55, half_w * 0.55):
        out.append(model(f"fan_circ_{k}_{y:.0f}", x, y, EAVE * 0.87, [
            vis("ring", cyl_geo(0.34, 0.22), pose(0, 0, 0, *ALONG_X), BLACK),
            vis("mount", box_geo(0.06, 0.06, 0.8), pose(0, 0, 0.5), STEEL),
        ]))

# Crop rows. Full-length "furniture" is one model per row.
_sv, _sf, _hv, _hf = [], [], [], []
for _li, _ly in enumerate(STEM_LINES):
    for _p in range(int(SEG / STEM_PITCH)):
        _px = -SEG / 2 + (_p + 0.5) * STEM_PITCH
        # Floor to canopy top, not GUTTER_Z to canopy top.
        obj_box(_sv, _sf, _px, _ly, CANOPY_TOP / 2, 0.026, 0.026, CANOPY_TOP)
        if _p % HOOK_EVERY == 0:
            # Tomahook spools must sit PROUD of the canopy face.
            obj_box(_hv, _hf, _px, math.copysign(CANOPY_THICK / 2 + 0.03, _ly),
                    CANOPY_TOP - 0.15, 0.11, 0.035, 0.11)
write_obj(WORLD_RESOURCE_DIR / "meshes/crop_stems.obj", _sv, _sf)
write_obj(WORLD_RESOURCE_DIR / "meshes/crop_hooks.obj", _hv, _hf)

crop_len = CROP_X1 - CROP_X0
crop_mid = (CROP_X0 + CROP_X1) / 2.0

row_ys = []
y = PATH_WIDTH / 2.0 + 0.8
while y + CANOPY_THICK / 2.0 < half_w - 1.0:
    row_ys += [-y, y]
    y += ROW_PITCH
row_ys.sort()
n_plants_row = int(crop_len / PLANT_SPACING)
for target_row, target_plant in TARGET_PLANTS:
    if not 0 <= target_row < len(row_ys):
        raise ValueError(
            "highlight row must be between 0 and %d"
            % (len(row_ys) - 1))
    if not 0 <= target_plant < n_plants_row:
        raise ValueError(
            "highlight plant must be between 0 and %d"
            % (n_plants_row - 1))

n_seg = int(round(crop_len / SEG))
n_stems = 0
n_plants = 0
plant_rng = random.Random(PLANT_RNG_SEED)

for r, ry in enumerate(row_ys):
    inward = -1.0 if ry > 0 else 1.0     # side of the row facing the centre path

    out.append(model(f"row_{r}_rig", crop_mid, ry, 0, [
        # twin 51 mm heating pipes -- also the harvest-trolley rail
        vis("rail_a", cyl_geo(0.026, crop_len), pose(0, -RAIL_GAUGE / 2, 0.075, *ALONG_X), STEEL),
        vis("rail_b", cyl_geo(0.026, crop_len), pose(0, RAIL_GAUGE / 2, 0.075, *ALONG_X), STEEL),
        vis("hanger_l", cyl_geo(0.008, GUTTER_Z), pose(0, -0.12, GUTTER_Z / 2), GALV),
        # yellow sticky trap ribbon along the aisle face of the canopy.
        vis("trap", box_geo(crop_len, 0.03, 0.14),
            pose(0, inward * (CANOPY_THICK / 2 + 0.06), CANOPY_TOP * 0.62), YELLOW),
        # crop wires the plants hang from
        vis("wire_a", cyl_geo(0.005, crop_len), pose(0, -0.25, WIRE_Z, *ALONG_X), GALV),
        vis("wire_b", cyl_geo(0.005, crop_len), pose(0, 0.25, WIRE_Z, *ALONG_X), GALV),
    ]))

    # Real plant x-positions for this row, computed up front so the hedge loop below can cut a gap.
    plant_xs = [CROP_X0 + (p + 0.5) * PLANT_SPACING for p in range(n_plants_row)]
    GAP_WIDTH = 1.8   # clear of the largest scaled plant's own footprint

    for s in range(n_seg):
        sx = CROP_X0 + (s + 0.5) * SEG
        jitter_h = JITTER_H * math.sin(s * 1.7 + r)
        jitter_y = JITTER_Y * math.sin(s * 2.3 + r * 0.7)
        h = (CANOPY_TOP - CANOPY_BOTTOM) + jitter_h
        cz = CANOPY_BOTTOM + h / 2

        # The hedge is now COLLISION-ONLY.
        out.append(model(f"row_{r}_seg_{s}", sx, ry, 0, [
            col("hedge", box_geo(SEG, CANOPY_THICK, h), pose(0, jitter_y, cz)),
            mesh_vis("stems", "meshes/crop_stems.obj", STEM),
            mesh_vis("hooks", "meshes/crop_hooks.obj", WHITE),
        ]))
        n_stems += len(STEM_LINES) * int(SEG / STEM_PITCH)

    # Real leafy tomato plants , centred in the gap cut above.
    base_yaw = math.pi if ry > 0 else 0.0
    for p, px in enumerate(plant_xs):
        variant = AOC_PLANTS[(r * 7 + p * 3) % len(AOC_PLANTS)]
        # Stretch z so the plant actually REACHES the canopy top.
        target_top = CANOPY_TOP * plant_rng.uniform(0.90, 1.02)
        z_scale = target_top / variant["extents"][2]
        # <scale> scales about the mesh origin, so the mesh's lowest point in world space is origin_z + base_z*z_scale.
        pz = -variant["base_z"] * z_scale
        plant_yaw = base_yaw + plant_rng.uniform(-PLANT_YAW_JITTER, PLANT_YAW_JITTER)
        is_target_plant = (r, p) in TARGET_PLANTS
        plant_scale = (PLANT_SCALE, PLANT_SCALE, z_scale)
        parts = [
            submesh_vis(name, variant["uri"], name,
                        AOC_SUBMESH_COLOUR[name], plant_scale)
            for name in variant["submeshes"]
        ]
        if is_target_plant:
            parts.extend(
                highlight_submesh_vis(
                    name, variant["uri"], name, plant_scale)
                for name in variant["submeshes"])
        out.append(
            f'\n    <model name="row_{r}_plant_{p}"><static>true</static>'
            f'{pose(px, ry, pz, 0, 0, plant_yaw)}<link name="link">'
            f'{"".join(parts)}</link></model>'
        )
    n_plants += n_plants_row

# ---------------------------------------------------------------------------
world = f"""<?xml version="1.0" encoding="UTF-8"?>
<sdf version="1.9">
  <world name="greenhouse_venlo">
    <physics type="ode">
      <max_step_size>0.004</max_step_size>
      <real_time_factor>2.0</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
    </physics>
    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type="adiabatic"/>
    <scene>
      <grid>false</grid>
      <ambient>0.55 0.56 0.55 1</ambient>
      <background>0.82 0.85 0.88 1</background>
      <shadows>true</shadows>
    </scene>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>1 1</size></plane></geometry>
          <surface><friction><ode/></friction><bounce/><contact/></surface>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>500 500</size></plane></geometry>
          <material>
            <ambient>0.55 0.55 0.53 1</ambient>
            <diffuse>0.55 0.55 0.53 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    <light name="sunUTC" type="directional">
      <pose>0 0 500 0 -0 0</pose>
      <cast_shadows>true</cast_shadows>
      <intensity>1</intensity>
      <direction>0.001 0.625 -0.78</direction>
      <diffuse>0.904 0.904 0.904 1</diffuse>
      <specular>0.271 0.271 0.271 1</specular>
      <attenuation>
        <range>2000</range><linear>0</linear><constant>1</constant><quadratic>0</quadratic>
      </attenuation>
      <spot><inner_angle>0</inner_angle><outer_angle>0</outer_angle><falloff>0</falloff></spot>
    </light>
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>51.98</latitude_deg>
      <longitude_deg>4.13</longitude_deg>
      <elevation>0</elevation>
    </spherical_coordinates>
{"".join(out)}
  </world>
</sdf>
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(world)

# Check the main layout assumptions before writing the world.
_corridor = PATH_WIDTH / 2.0
assert all(abs(y) - CANOPY_THICK / 2.0 - JITTER_Y > _corridor for y in row_ys), \
    "crop hedge overhangs the central flight corridor"
assert CANOPY_THICK / 2 + 0.03 > max(abs(v) for v in STEM_LINES), \
    "tomahook discs sit inside the opaque hedge box and will never render"
assert all(abs(y) > _corridor for y in column_ys), "a column stands in the corridor"
assert CROP_X0 < 0 < CROP_X1 and X_MIN < 0, "spawn point (0,0) is outside the house"
assert CANOPY_TOP + 0.5 < EAVE, "no headspace between canopy and gutter"
assert row_ys[0] > -half_w and row_ys[-1] < half_w, "crop rows poke through the side walls"
assert all(abs(y) - AOC_MAX_EXTENT_Y / 2 > _corridor
           for y in row_ys), "real plant mesh pokes into the flight corridor"

n_models = world.count("<model name=")
n_visuals = world.count("<visual name=")
print(f"Wrote {OUT}")
print(f"  house      {LENGTH:.0f} m x {2 * half_w:.0f} m, {EAVE:.1f} m gutter, "
      f"{N_BAYS} x {BAY:.1f} m Venlo bays")
print(f"  crop       {len(row_ys)} rows @ {ROW_PITCH} m, {n_stems} stems, "
      f"canopy {CANOPY_BOTTOM}-{CANOPY_TOP} m, {n_plants} real tomato plant meshes")
print(f"  clear      central path y=+-{PATH_WIDTH / 2:.1f} m, "
      f"headspace {CANOPY_TOP:.1f}-{EAVE - 0.6:.1f} m above canopy")
print(f"  {n_models} models / {n_visuals} visuals "
      f"(raise SEG or HOOK_EVERY if the GUI drags)")
print(f"  highlights  {TARGET_PLANTS} "
      f"(normal materials + translucent cyan overlays)")
