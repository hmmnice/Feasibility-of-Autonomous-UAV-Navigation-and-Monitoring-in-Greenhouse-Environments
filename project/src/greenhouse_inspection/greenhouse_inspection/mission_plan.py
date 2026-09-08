"""Build an ordered, auditable preflight list for one or more crop targets."""

import argparse
import json
from pathlib import Path


LIST_SCHEMA = "gps_greenhouse_mission_preflight_list/v1"


def preflight_names(method, side_policy, views_per_target, clicked_face):
    """Return filenames in the requested per-target capture order."""
    if side_policy not in ("both", "positive_y", "negative_y"):
        raise ValueError("invalid side policy")
    if method not in ("fixed", "geometric"):
        raise ValueError("invalid viewpoint method")
    if views_per_target not in (1, 2, 4):
        raise ValueError("views_per_target must be 1, 2 or 4")
    if views_per_target == 4 and (method != "geometric" or side_policy != "both"):
        raise ValueError("four views require geometric planning on both sides")
    if side_policy != "both":
        suffix = "pos_y" if side_policy == "positive_y" else "neg_y"
        prefix = "auto_" if method == "geometric" else "side_"
        return (prefix + suffix + "_preflight.json",)

    first, second = (
        ("positive_y", "negative_y")
        if clicked_face == "positive_y" else
        ("negative_y", "positive_y"))
    if views_per_target == 4:
        return tuple(
            "auto_%s_%s_preflight.json" % (side, offset)
            for side in (first, second)
            for offset in ("left", "right"))
    if method == "geometric":
        return tuple("auto_%s_preflight.json" % side
                     for side in (first, second))
    return tuple(
        "side_%s_preflight.json" % (
            "pos_y" if side == "positive_y" else "neg_y")
        for side in (first, second))


def build_preflight_list(
        selection_record, preflight_dir, method="geometric",
        side_policy="both", views_per_target=2):
    """Resolve preflight filenames while preserving operator selection order."""
    targets = selection_record.get("targets", [])
    if not isinstance(targets, list) or not targets:
        raise ValueError("selection contains no crop targets")
    if len(targets) > 4:
        raise ValueError("one continuous mission is limited to four crops")
    if len(targets) > 1 and (side_policy != "both" or views_per_target != 2):
        raise ValueError(
            "multi-target missions require two standard opposite-side views per crop")
    root = Path(preflight_dir)
    paths = []
    target_order = []
    for target in targets:
        identifier = int(target["target_id"])
        target_order.append(identifier)
        directory = root / ("target_%03d" % identifier)
        for name in preflight_names(
                method, side_policy, views_per_target, target.get("face")):
            path = directory / name
            if not path.is_file():
                raise ValueError("expected preflight was not generated: %s" % path)
            paths.append(str(path))
    return {
        "schema": LIST_SCHEMA,
        "selection_mode": selection_record.get("selection_mode"),
        "side_policy": side_policy,
        "viewpoint_method": method,
        "target_count": len(targets),
        "target_order": target_order,
        "view_count": len(paths),
        "preflight_jsons": paths,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build an ordered multi-target preflight list")
    parser.add_argument("--selection-json", required=True)
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--method", choices=("fixed", "geometric"), default="geometric")
    parser.add_argument(
        "--side", choices=("both", "positive_y", "negative_y"),
        default="both")
    parser.add_argument("--views-per-target", type=int, default=2)
    arguments = parser.parse_args()
    selection = json.loads(
        Path(arguments.selection_json).read_text(encoding="utf-8"))
    result = build_preflight_list(
        selection,
        arguments.preflight_dir,
        arguments.method,
        arguments.side,
        arguments.views_per_target,
    )
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Mission preflight list:", output)
    print("Targets:", result["target_order"])
    print("Views:", result["view_count"])


if __name__ == "__main__":
    main()
