"""Surveyed geometry and crop catalogue for greenhouse_venlo."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from greenhouse_inspection.free_space import (
    CANOPY_BOTTOM,
    CANOPY_JITTER_H,
    CANOPY_JITTER_Y,
    CANOPY_THICK,
    CANOPY_TOP,
    CANOPY_X0,
    CANOPY_X1,
    DRONE_HALF,
    DRONE_HALF_Z,
    EAVE,
    TRACKING_ALLOWANCE,
    VENLO_COLUMN_WIDTH,
    VENLO_COLUMN_XS,
    VENLO_COLUMN_YS,
)
from greenhouse_inspection.map_geometry import Geometry
from greenhouse_inspection.row_coverage import venlo_rows


MAP_SCHEMA = "gps_greenhouse_static_map/v1"
SELECTION_SCHEMA = "gps_greenhouse_target_selection/v1"
CELL_SCHEMA = "gps_greenhouse_square_metre_cell/v1"
CELL_SIZE_M = 1.0
PLANT_SPACING = 1.2
TARGET_HEIGHT = 3.5
TARGET_FACE_OFFSET = 0.30
HOUSE_X_BOUNDS = (-6.0, 24.0)
HOUSE_Y_BOUNDS = (-12.0, 12.0)


def rows():
    """Return the twelve dimensioned crop-row centre lines."""
    return tuple(float(value) for value in venlo_rows())


def plant_x_positions():
    """Return the exact plant x coordinates used by the world generator."""
    count = int((CANOPY_X1 - CANOPY_X0) / PLANT_SPACING)
    return tuple(
        CANOPY_X0 + (index + 0.5) * PLANT_SPACING
        for index in range(count))


def static_geometry():
    """Create planner geometry with crops first and structure second."""
    crop_rows = rows()
    canopy_half_y = (
        CANOPY_THICK / 2.0 + CANOPY_JITTER_Y + DRONE_HALF)
    crop_boxes = [
        (
            CANOPY_X0 - DRONE_HALF,
            CANOPY_X1 + DRONE_HALF,
            row_y - canopy_half_y,
            row_y + canopy_half_y,
            0.0,
            TARGET_HEIGHT + DRONE_HALF_Z,
        )
        for row_y in crop_rows
    ]
    column_half = (
        VENLO_COLUMN_WIDTH / 2.0 + DRONE_HALF + TRACKING_ALLOWANCE)
    column_boxes = [
        (
            column_x - column_half,
            column_x + column_half,
            column_y - column_half,
            column_y + column_half,
            0.0,
            EAVE,
        )
        for column_x in VENLO_COLUMN_XS
        for column_y in VENLO_COLUMN_YS
    ]
    return Geometry(
        crop_rows,
        TARGET_HEIGHT,
        tuple((CANOPY_X0, CANOPY_X1) for _ in crop_rows),
        tuple(crop_boxes + column_boxes),
    )


def crop_catalogue():
    """Return selectable surface targets for both faces of every plant."""
    catalogue = []
    identifier = 1
    for row_index, row_y in enumerate(rows()):
        for plant_index, plant_x in enumerate(plant_x_positions()):
            for face_sign, face_name in (
                    (-1, "negative_y"), (1, "positive_y")):
                catalogue.append({
                    "target_id": identifier,
                    "crop_id": "row_%02d_plant_%02d" % (
                        row_index, plant_index),
                    "row_index": row_index,
                    "plant_index": plant_index,
                    "face": face_name,
                    "target_greenhouse_map": [
                        float(plant_x),
                        float(row_y + face_sign * TARGET_FACE_OFFSET),
                        TARGET_HEIGHT,
                    ],
                })
                identifier += 1
    return tuple(catalogue)


def cell_catalogue(cell_size=CELL_SIZE_M):
    """Return deterministic 1 m² operator-query cells for the greenhouse."""
    cell_size = float(cell_size)
    if cell_size <= 0.0:
        raise ValueError("cell size must be positive")
    x_count = int(round(
        (HOUSE_X_BOUNDS[1] - HOUSE_X_BOUNDS[0]) / cell_size))
    y_count = int(round(
        (HOUSE_Y_BOUNDS[1] - HOUSE_Y_BOUNDS[0]) / cell_size))
    cells = []
    identifier = 1
    for y_index in range(y_count):
        y_min = HOUSE_Y_BOUNDS[0] + y_index * cell_size
        for x_index in range(x_count):
            x_min = HOUSE_X_BOUNDS[0] + x_index * cell_size
            cells.append({
                "schema": CELL_SCHEMA,
                "cell_id": identifier,
                "x_index": x_index,
                "y_index": y_index,
                "bounds_xy_m": [
                    [float(x_min), float(x_min + cell_size)],
                    [float(y_min), float(y_min + cell_size)],
                ],
                "centre_xy_m": [
                    float(x_min + cell_size / 2.0),
                    float(y_min + cell_size / 2.0),
                ],
                "area_m2": cell_size * cell_size,
            })
            identifier += 1
    return tuple(cells)


def cell_from_id(cell_id):
    """Return one validated cell by its stable one-based identifier."""
    cells = cell_catalogue()
    cell_id = int(cell_id)
    if not 1 <= cell_id <= len(cells):
        raise ValueError("cell ID is outside the greenhouse grid")
    return cells[cell_id - 1]


def map_record():
    """Return the auditable static-map record used by the selector."""
    geometry = static_geometry()
    return {
        "schema": MAP_SCHEMA,
        "world": "greenhouse_venlo",
        "frame": "greenhouse_map",
        "localisation": "PX4 EKF using Gazebo simulated GPS and IMU",
        "source": "dimensioned simulation world, not SLAM",
        "house_bounds_xy": [list(HOUSE_X_BOUNDS), list(HOUSE_Y_BOUNDS)],
        "canopy": {
            "x_extent_m": [CANOPY_X0, CANOPY_X1],
            "bottom_m": CANOPY_BOTTOM,
            "nominal_top_m": CANOPY_TOP,
            "height_jitter_m": CANOPY_JITTER_H,
            "thickness_m": CANOPY_THICK,
        },
        "row_centres_y_m": list(geometry.rows),
        "plant_spacing_m": PLANT_SPACING,
        "plant_count": len(rows()) * len(plant_x_positions()),
        "selectable_face_count": len(crop_catalogue()),
        "operator_grid": {
            "schema": CELL_SCHEMA,
            "cell_size_m": CELL_SIZE_M,
            "cell_count": len(cell_catalogue()),
        },
        "columns_xy_m": [
            [float(x_value), float(y_value)]
            for x_value in VENLO_COLUMN_XS
            for y_value in VENLO_COLUMN_YS
        ],
        "targets": list(crop_catalogue()),
    }


def plot_map(
        selected=(), output_path=None, interactive=False,
        show_square_metre_grid=False):
    """Draw the static map and optionally show an interactive window."""
    figure, axis = plt.subplots(figsize=(14, 9))
    axis.set_facecolor("#f4f4f1")
    for row_index, row_y in enumerate(rows()):
        axis.add_patch(Rectangle(
            (CANOPY_X0, row_y - CANOPY_THICK / 2.0),
            CANOPY_X1 - CANOPY_X0,
            CANOPY_THICK,
            facecolor="#4f7f45",
            edgecolor="#294b25",
            alpha=0.55,
        ))
        axis.text(
            CANOPY_X1 + 0.25, row_y, "R%02d" % row_index,
            va="center", fontsize=8)
    plants_x = plant_x_positions()
    for row_y in rows():
        axis.scatter(
            plants_x, np.full(len(plants_x), row_y), s=9,
            c="#1f5a24", alpha=0.75, zorder=3)
    columns = np.asarray([
        (x_value, y_value)
        for x_value in VENLO_COLUMN_XS
        for y_value in VENLO_COLUMN_YS
    ])
    axis.scatter(
        columns[:, 0], columns[:, 1], marker="s", s=40,
        c="#474747", label="structural columns", zorder=4)
    axis.add_patch(Rectangle(
        (HOUSE_X_BOUNDS[0], -1.0),
        HOUSE_X_BOUNDS[1] - HOUSE_X_BOUNDS[0], 2.0,
        facecolor="#d7d5ce", edgecolor="none", alpha=0.9,
        label="central path"))
    if show_square_metre_grid:
        x_ticks = np.arange(
            HOUSE_X_BOUNDS[0], HOUSE_X_BOUNDS[1] + CELL_SIZE_M,
            CELL_SIZE_M)
        y_ticks = np.arange(
            HOUSE_Y_BOUNDS[0], HOUSE_Y_BOUNDS[1] + CELL_SIZE_M,
            CELL_SIZE_M)
        for value in x_ticks:
            axis.axvline(value, color="#356da4", linewidth=0.35,
                         alpha=0.28, zorder=1)
        for value in y_ticks:
            axis.axhline(value, color="#356da4", linewidth=0.35,
                         alpha=0.28, zorder=1)
    if selected:
        points = np.asarray([
            item["target_greenhouse_map"][:2] for item in selected])
        axis.scatter(
            points[:, 0], points[:, 1], marker="*", s=190,
            c="#e4572e", edgecolor="white", linewidth=0.8,
            label="selected target", zorder=6)
    axis.set(
        title=(
            "GPS viewpoint map — click a 1 m² query cell"
            if show_square_metre_grid else
            "GPS viewpoint map — click a crop row near the wanted plant"),
        xlabel="Greenhouse x (m)",
        ylabel="Greenhouse y (m)",
        xlim=(HOUSE_X_BOUNDS[0] - 0.5, HOUSE_X_BOUNDS[1] + 1.5),
        ylim=(HOUSE_Y_BOUNDS[0] - 0.8, HOUSE_Y_BOUNDS[1] + 0.8),
    )
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, alpha=0.2)
    axis.legend(loc="upper right")
    figure.tight_layout()
    if output_path is not None:
        figure.savefig(output_path, dpi=180)
    if interactive:
        plt.show()
    return figure, axis


def write_static_assets(output_dir):
    """Write the source-of-truth JSON and a presentation-ready PNG."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "greenhouse_static_map.json"
    png_path = directory / "greenhouse_static_map.png"
    json_path.write_text(
        json.dumps(map_record(), indent=2) + "\n", encoding="utf-8")
    figure, _ = plot_map(output_path=png_path)
    plt.close(figure)
    return json_path, png_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate the surveyed greenhouse crop map")
    parser.add_argument(
        "--output-dir", default="gps_viewpoint_map")
    arguments = parser.parse_args()
    json_path, png_path = write_static_assets(arguments.output_dir)
    print("Static GPS map JSON:", json_path)
    print("Static GPS map PNG: ", png_path)


if __name__ == "__main__":
    main()
