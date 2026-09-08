"""Clickable crop selector for the surveyed greenhouse map."""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

from .static_greenhouse import (
    CELL_SIZE_M,
    MAP_SCHEMA,
    SELECTION_SCHEMA,
    TARGET_FACE_OFFSET,
    TARGET_HEIGHT,
    cell_catalogue,
    cell_from_id,
    crop_catalogue,
    plant_x_positions,
    plot_map,
    rows,
    write_static_assets,
)

MAX_CONTINUOUS_TARGETS = 4


def snap_click(x_value, y_value, maximum_row_distance=0.85):
    """Snap an x/y click to the nearest plant and selected row face."""
    if not all(math.isfinite(value) for value in (x_value, y_value)):
        raise ValueError("click coordinates must be finite")
    crop_rows = rows()
    row_index = min(
        range(len(crop_rows)),
        key=lambda index: abs(crop_rows[index] - y_value))
    row_y = crop_rows[row_index]
    if abs(row_y - y_value) > maximum_row_distance:
        raise ValueError("click is not close enough to a crop row")
    plants = plant_x_positions()
    plant_index = min(
        range(len(plants)), key=lambda index: abs(plants[index] - x_value))
    face_positive = y_value >= row_y
    face_index = 1 if face_positive else 0
    target_id = (
        row_index * len(plants) * 2 + plant_index * 2 + face_index + 1)
    target = crop_catalogue()[target_id - 1]
    expected_y = row_y + (1 if face_positive else -1) * TARGET_FACE_OFFSET
    if not math.isclose(
            target["target_greenhouse_map"][1], expected_y, abs_tol=1e-9):
        raise RuntimeError("static target catalogue indexing mismatch")
    return target


def resolve_cell(cell, side_hint_y=None):
    """Resolve one square-metre query cell to a deterministic nearby crop."""
    centre_x, centre_y = cell["centre_xy_m"]
    crop_rows = rows()
    row_index = min(
        range(len(crop_rows)),
        key=lambda index: abs(crop_rows[index] - centre_y))
    plants = plant_x_positions()
    plant_index = min(
        range(len(plants)),
        key=lambda index: abs(plants[index] - centre_x))
    if abs(plants[plant_index] - centre_x) > 0.9:
        raise ValueError("selected square-metre cell does not contain a crop")
    if abs(crop_rows[row_index] - centre_y) > 1.5:
        raise ValueError("selected square-metre cell is too far from a crop row")
    comparison_y = centre_y if side_hint_y is None else float(side_hint_y)
    face_positive = comparison_y >= crop_rows[row_index]
    face_index = 1 if face_positive else 0
    target_id = (
        row_index * len(plants) * 2 + plant_index * 2 + face_index + 1)
    return {
        **crop_catalogue()[target_id - 1],
        "operator_cell": cell,
        "cell_resolution": "nearest_catalogued_plant_to_cell_centre",
    }


def snap_cell(x_value, y_value):
    """Convert a greenhouse-map click into a 1 m² query and resolved crop."""
    if not all(math.isfinite(value) for value in (x_value, y_value)):
        raise ValueError("click coordinates must be finite")
    cells = cell_catalogue()
    for cell in cells:
        x_bounds, y_bounds = cell["bounds_xy_m"]
        if (x_bounds[0] <= x_value < x_bounds[1]
                and y_bounds[0] <= y_value < y_bounds[1]):
            return resolve_cell(cell, side_hint_y=y_value)
    raise ValueError("click is outside the greenhouse square-metre grid")


def selection_record(
        selected, map_path, distance=2.0, pitch_deg=62.1,
        selection_mode="plant", side_policy="both"):
    """Create non-executable operator intent from snapped targets."""
    if not selected:
        raise ValueError("at least one crop target must be selected")
    if len(selected) > MAX_CONTINUOUS_TARGETS:
        raise ValueError(
            "one continuous mission is limited to four crop targets")
    crop_ids = [item["crop_id"] for item in selected]
    if len(set(crop_ids)) != len(crop_ids):
        raise ValueError(
            "select each physical crop at most once; both row sides are "
            "planned automatically")
    return {
        "schema": SELECTION_SCHEMA,
        "mode": "offline_surveyed_map_only",
        "selection_mode": str(selection_mode),
        "side_policy": str(side_policy),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "map_schema": MAP_SCHEMA,
        "map_path": str(map_path),
        "planning_frame": "greenhouse_map",
        "localisation_for_live_preflight": (
            "PX4 EKF using Gazebo simulated GPS and IMU"),
        "distance_m": float(distance),
        "pitch_deg": float(pitch_deg),
        "target_height_m": TARGET_HEIGHT,
        "target_count": len(selected),
        "targets": [
            {
                **item,
                "target_saved_map": list(item["target_greenhouse_map"]),
            }
            for item in selected
        ],
        "dry_run": True,
        "flight_command_capable": False,
        "flight_commanded": False,
        "executable_by_viewpoint_execution": False,
    }


def write_selection(
        selected, output_dir, distance=2.0, pitch_deg=62.1,
        selection_mode="plant", side_policy="both"):
    """Write selected-target JSON and a PNG showing the operator choice."""
    directory = Path(output_dir)
    map_path, _ = write_static_assets(directory)
    record = selection_record(
        selected, map_path, distance, pitch_deg,
        selection_mode, side_policy)
    json_path = directory / "selected_targets.json"
    png_path = directory / "selected_targets.png"
    json_path.write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")
    figure, axis = plot_map(
        selected, output_path=None,
        show_square_metre_grid=selection_mode == "cell")
    for item in selected:
        cell = item.get("operator_cell")
        if cell:
            x_bounds, y_bounds = cell["bounds_xy_m"]
            axis.add_patch(plt.Rectangle(
                (x_bounds[0], y_bounds[0]), CELL_SIZE_M, CELL_SIZE_M,
                facecolor="#2a9d8f", edgecolor="#073b4c",
                linewidth=2.0, alpha=0.25, zorder=5))
            axis.text(
                cell["centre_xy_m"][0], cell["centre_xy_m"][1],
                "C%d" % cell["cell_id"], ha="center", va="center",
                fontsize=7, zorder=8)
    figure.savefig(png_path, dpi=180)
    plt.close(figure)
    return json_path, png_path


class InteractiveSelector:
    """Small matplotlib UI whose only output is a selection artifact."""

    def __init__(
            self, output_dir, distance, pitch_deg,
            selection_mode="plant", side_policy="both"):
        self.output_dir = output_dir
        self.distance = distance
        self.pitch_deg = pitch_deg
        self.selection_mode = selection_mode
        self.side_policy = side_policy
        self.selected = []
        self.figure, self.axis = plot_map(
            show_square_metre_grid=selection_mode == "cell")
        self.figure.canvas.mpl_connect("button_press_event", self.on_click)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.selected_artist = None
        self.message = self.axis.text(
            0.01, 0.01,
            "Left click: select | right click: undo | Enter: save | q: quit",
            transform=self.axis.transAxes,
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
            zorder=10,
        )

    def redraw_selected(self):
        if self.selected_artist is not None:
            self.selected_artist.remove()
            self.selected_artist = None
        if self.selected:
            x_values = [item["target_greenhouse_map"][0]
                        for item in self.selected]
            y_values = [item["target_greenhouse_map"][1]
                        for item in self.selected]
            self.selected_artist = self.axis.scatter(
                x_values, y_values, marker="*", s=190,
                c="#e4572e", edgecolor="white", linewidth=0.8,
                zorder=7)
        self.figure.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes is not self.axis:
            return
        if event.button == 3:
            if self.selected:
                removed = self.selected.pop()
                self.message.set_text(
                    "Removed target %d" % removed["target_id"])
                self.redraw_selected()
            return
        if event.button != 1:
            return
        try:
            if self.selection_mode == "cell":
                target = snap_cell(float(event.xdata), float(event.ydata))
            else:
                target = snap_click(float(event.xdata), float(event.ydata))
        except ValueError as error:
            self.message.set_text(str(error))
            self.figure.canvas.draw_idle()
            return
        if any(item["crop_id"] == target["crop_id"]
               for item in self.selected):
            self.message.set_text(
                "That crop is already selected; both sides are automatic")
            self.figure.canvas.draw_idle()
            return
        if len(self.selected) >= MAX_CONTINUOUS_TARGETS:
            self.message.set_text(
                "This continuous mission is limited to four crops")
            self.figure.canvas.draw_idle()
            return
        self.selected.append(target)
        cell = target.get("operator_cell")
        prefix = "cell %d -> " % cell["cell_id"] if cell else ""
        self.message.set_text(
            "Selected %s%s (%s), target %d; mission side=%s" % (
                prefix, target["crop_id"], target["face"],
                target["target_id"], self.side_policy))
        self.redraw_selected()

    def on_key(self, event):
        if event.key in ("enter", "return"):
            if not self.selected:
                self.message.set_text("Select at least one crop first")
                self.figure.canvas.draw_idle()
                return
            json_path, png_path = write_selection(
                self.selected, self.output_dir,
                self.distance, self.pitch_deg,
                self.selection_mode, self.side_policy)
            print("Selection JSON:", json_path)
            print("Selection PNG: ", png_path)
            plt.close(self.figure)
        elif event.key in ("q", "escape"):
            plt.close(self.figure)

    def run(self):
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Click plants on the surveyed greenhouse map")
    parser.add_argument(
        "--output-dir", default="gps_viewpoint_plan")
    parser.add_argument("--distance", type=float, default=2.0)
    parser.add_argument("--pitch-deg", type=float, default=62.1)
    parser.add_argument(
        "--selection-mode", choices=("plant", "cell"), default="plant")
    parser.add_argument(
        "--side", choices=("both", "positive_y", "negative_y"),
        default="both")
    parser.add_argument(
        "--select-id", type=int, action="append", default=[],
        help=("Headless/test mode: save this catalogue target ID; repeat "
              "the option to select multiple crops"))
    parser.add_argument(
        "--cell-id", type=int, action="append", default=[],
        help=("Headless/test mode: resolve and save this 1 m² cell ID; "
              "repeat the option to select multiple cells"))
    arguments = parser.parse_args()
    catalogue = crop_catalogue()
    if arguments.select_id and arguments.cell_id:
        parser.error("use only one of --select-id and --cell-id")
    if arguments.select_id:
        if any(not 1 <= identifier <= len(catalogue)
               for identifier in arguments.select_id):
            parser.error("--select-id is outside the target catalogue")
        if len(set(arguments.select_id)) != len(arguments.select_id):
            parser.error("--select-id contains a duplicate target")
        if len(arguments.select_id) > MAX_CONTINUOUS_TARGETS:
            parser.error("at most four crops can be selected")
        targets = [catalogue[identifier - 1]
                   for identifier in arguments.select_id]
        if len({target["crop_id"] for target in targets}) != len(targets):
            parser.error(
                "select each physical crop once; both sides are automatic")
        json_path, png_path = write_selection(
            targets, arguments.output_dir,
            arguments.distance, arguments.pitch_deg,
            "plant", arguments.side)
        print("Selection JSON:", json_path)
        print("Selection PNG: ", png_path)
        return
    if arguments.cell_id:
        if len(arguments.cell_id) > MAX_CONTINUOUS_TARGETS:
            parser.error("at most four cells can be selected")
        try:
            targets = [resolve_cell(cell_from_id(identifier))
                       for identifier in arguments.cell_id]
        except ValueError as error:
            parser.error(str(error))
        crop_ids = [target["crop_id"] for target in targets]
        if len(set(crop_ids)) != len(crop_ids):
            parser.error(
                "selected cells resolve to a duplicate crop target")
        json_path, png_path = write_selection(
            targets, arguments.output_dir,
            arguments.distance, arguments.pitch_deg,
            "cell", arguments.side)
        print("Selection JSON:", json_path)
        print("Selection PNG: ", png_path)
        return
    InteractiveSelector(
        arguments.output_dir,
        arguments.distance,
        arguments.pitch_deg,
        arguments.selection_mode,
        arguments.side,
    ).run()


if __name__ == "__main__":
    main()
