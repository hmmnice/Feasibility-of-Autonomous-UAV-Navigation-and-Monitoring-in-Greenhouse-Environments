#!/usr/bin/env bash
# One-command orchestration for the GPS-first crop viewpoint experiment.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
THESIS_DIR=$(cd -- "${SCRIPT_DIR}/../../.." && pwd)
export THESIS_DIR
WORKSPACE_DIR=${THESIS_DIR}/project
RUN_ROOT=${THESIS_DIR}/gps_viewpoint_runs
WORLD_NAME=greenhouse_venlo
FLY=0
REBUILD=1
MANUAL_VIEW=0
SELECTION_MODE=plant
VIEWPOINT_METHOD=geometric
ROUTE_PLANNER=structured
SIDE_POLICY=both
TARGET_ID_ARGS=()
CELL_ID_ARGS=()
RUN_ID_OVERRIDE=
DISTANCE=2.0
PITCH_DEG=62.1
VIEW_COUNT=2
MISSION_TIMEOUT_SEC=240.0
MISSION_TIMEOUT_EXPLICIT=0

usage() {
    echo "One-command GPS crop selection and Gazebo viewpoint flight."
    echo "Usage: bash $0 [--fly] [--selection-mode plant|cell] [--target-id ID ...|--cell-id ID ...]"
    echo "              [--viewpoint-method fixed|geometric] [--route-planner structured|astar]"
    echo "              [--side both|positive_y|negative_y] [--views 2|4] [--distance M] [--pitch-deg DEG]"
    echo "              [--mission-timeout-sec S]"
    echo
    echo "  --fly          After click + automatic dry validation, fly in Gazebo."
    echo "  --no-rebuild   Use the currently installed package."
    echo "  --manual-view  Use fixed distance/pitch and the clicked row face."
    echo "  --distance     Manual-view stand-off (default: 2.0 m)."
    echo "  --pitch-deg    Manual-view downward pitch (default: 62.1 degrees)."
    echo "  --target-id    Repeatable plant selection without clicking; may be supplied more than once."
    echo "  --cell-id      Repeatable 1 m² selection without clicking; may be supplied more than once."
    echo "  --run-id       Stable run-directory name for reproducible batches."
    echo "  --views        Capture 2 standard views or 4 oblique views (default: 2)."
    echo "  --mission-timeout-sec  Simulated-time mission limit (default: 240 s)."
}

while (($#)); do
    case "$1" in
        --fly)
            FLY=1
            shift
            ;;
        --no-rebuild)
            REBUILD=0
            shift
            ;;
        --manual-view)
            MANUAL_VIEW=1
            VIEWPOINT_METHOD=fixed
            shift
            ;;
        --selection-mode)
            [ "$#" -ge 2 ] || { echo "ERROR: --selection-mode needs a value" >&2; exit 2; }
            SELECTION_MODE=$2
            shift 2
            ;;
        --viewpoint-method)
            [ "$#" -ge 2 ] || { echo "ERROR: --viewpoint-method needs a value" >&2; exit 2; }
            VIEWPOINT_METHOD=$2
            shift 2
            ;;
        --route-planner)
            [ "$#" -ge 2 ] || { echo "ERROR: --route-planner needs a value" >&2; exit 2; }
            ROUTE_PLANNER=$2
            shift 2
            ;;
        --side)
            [ "$#" -ge 2 ] || { echo "ERROR: --side needs a value" >&2; exit 2; }
            SIDE_POLICY=$2
            shift 2
            ;;
        --target-id)
            [ "$#" -ge 2 ] || { echo "ERROR: --target-id needs a value" >&2; exit 2; }
            TARGET_ID_ARGS+=("$2")
            shift 2
            ;;
        --cell-id)
            [ "$#" -ge 2 ] || { echo "ERROR: --cell-id needs a value" >&2; exit 2; }
            CELL_ID_ARGS+=("$2")
            shift 2
            ;;
        --run-id)
            [ "$#" -ge 2 ] || { echo "ERROR: --run-id needs a value" >&2; exit 2; }
            RUN_ID_OVERRIDE=$2
            shift 2
            ;;
        --distance)
            [ "$#" -ge 2 ] || { echo "ERROR: --distance needs a value" >&2; exit 2; }
            DISTANCE=$2
            shift 2
            ;;
        --pitch-deg)
            [ "$#" -ge 2 ] || { echo "ERROR: --pitch-deg needs a value" >&2; exit 2; }
            PITCH_DEG=$2
            shift 2
            ;;
        --views)
            [ "$#" -ge 2 ] || { echo "ERROR: --views needs a value" >&2; exit 2; }
            VIEW_COUNT=$2
            shift 2
            ;;
        --mission-timeout-sec)
            [ "$#" -ge 2 ] || { echo "ERROR: --mission-timeout-sec needs a value" >&2; exit 2; }
            MISSION_TIMEOUT_SEC=$2
            MISSION_TIMEOUT_EXPLICIT=1
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$SELECTION_MODE" in plant|cell) ;; *) echo "ERROR: invalid selection mode" >&2; exit 2;; esac
case "$VIEWPOINT_METHOD" in fixed|geometric) ;; *) echo "ERROR: invalid viewpoint method" >&2; exit 2;; esac
case "$ROUTE_PLANNER" in structured|astar) ;; *) echo "ERROR: invalid route planner" >&2; exit 2;; esac
case "$SIDE_POLICY" in both|positive_y|negative_y) ;; *) echo "ERROR: invalid side policy" >&2; exit 2;; esac
case "$VIEW_COUNT" in 2|4) ;; *) echo "ERROR: --views must be 2 or 4" >&2; exit 2;; esac
if [ "${#TARGET_ID_ARGS[@]}" -gt 0 ] && [ "${#CELL_ID_ARGS[@]}" -gt 0 ]; then
    echo "ERROR: use only one of --target-id and --cell-id" >&2
    exit 2
fi
if [ "$VIEWPOINT_METHOD" = fixed ] && [ "$ROUTE_PLANNER" != structured ]; then
    echo "ERROR: A* execution-route comparison currently requires --viewpoint-method geometric" >&2
    exit 2
fi
if [ "$VIEW_COUNT" -eq 4 ] && { [ "$VIEWPOINT_METHOD" != geometric ] || [ "$SIDE_POLICY" != both ]; }; then
    echo "ERROR: --views 4 requires geometric viewpoint planning and --side both" >&2
    exit 2
fi
if [ ! -f /opt/ros/humble/setup.bash ] || [ ! -d "$THESIS_DIR" ]; then
    echo "ERROR: run this script inside the ROS container." >&2
    echo "Enter it first with: docker exec -it ros_humble_persistent bash" >&2
    exit 1
fi

if [ -z "${DISPLAY:-}" ]; then
    echo "ERROR: DISPLAY is not set, so the clickable map cannot open." >&2
    echo "Run this from the same graphical container terminal used for Gazebo." >&2
    exit 1
fi

set +u
# shellcheck source=/dev/null
source /opt/ros/humble/setup.bash
if [ -f "${WORKSPACE_DIR}/install/setup.bash" ]; then
    # shellcheck source=/dev/null
    source "${WORKSPACE_DIR}/install/setup.bash"
fi
set -u

assert_no_active_viewpoint_executor() {
    # Do not let an old executor publish into a new PX4 instance.
    local active
    active=$(pgrep -af 'greenhouse_inspection.*(execute_viewpoint|execute_two_viewpoints)' || true)
    if [ -n "$active" ]; then
        echo "ERROR: a previous GPS viewpoint executor is still running:" >&2
        echo "$active" >&2
        echo "Stop or allow that run to finish before starting a new experiment." >&2
        echo "No new simulator, planner or PX4 commands were started." >&2
        exit 1
    fi
}

assert_no_active_viewpoint_executor

if [ "$REBUILD" -eq 1 ]; then
    echo "[1/9] Building the isolated GPS viewpoint package..."
    (
        cd "$WORKSPACE_DIR"
        colcon build --packages-select greenhouse_inspection --symlink-install
    )
    # ROS setup scripts expect nounset to be disabled.
    set +u
    # shellcheck source=/dev/null
    source "${WORKSPACE_DIR}/install/setup.bash"
    set -u
else
    echo "[1/9] Using the currently installed GPS viewpoint package."
fi

RUN_ID=${RUN_ID_OVERRIDE:-$(date -u +%Y%m%dT%H%M%SZ)}
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "ERROR: --run-id may contain only letters, digits, dot, underscore and hyphen" >&2
    exit 2
fi
RUN_DIR=${RUN_ROOT}/${RUN_ID}
if [ -e "$RUN_DIR" ]; then
    echo "ERROR: run directory already exists: $RUN_DIR" >&2
    exit 1
fi
SELECTION_DIR=${RUN_DIR}/selection
PREFLIGHT_DIR=${RUN_DIR}/preflight
PHOTO_DIR=${RUN_DIR}/photos
mkdir -p "$SELECTION_DIR" "$PREFLIGHT_DIR" "$PHOTO_DIR"

echo "[2/9] Resolving the greenhouse operator request before Gazebo starts."
SELECT_ARGS=(
    --output-dir "$SELECTION_DIR"
    --distance "$DISTANCE"
    --pitch-deg "$PITCH_DEG"
    --selection-mode "$SELECTION_MODE"
    --side "$SIDE_POLICY"
)
if [ "${#TARGET_ID_ARGS[@]}" -gt 0 ]; then
    for identifier in "${TARGET_ID_ARGS[@]}"; do
        SELECT_ARGS+=(--select-id "$identifier")
    done
elif [ "${#CELL_ID_ARGS[@]}" -gt 0 ]; then
    for identifier in "${CELL_ID_ARGS[@]}"; do
        SELECT_ARGS+=(--cell-id "$identifier")
    done
else
    echo "      Left click one or more ${SELECTION_MODE} targets, then press Enter to save."
    echo "      Right click undoes a selection; q cancels."
fi
ros2 run greenhouse_inspection select_crop \
    "${SELECT_ARGS[@]}"

SELECTION_JSON=${SELECTION_DIR}/selected_targets.json
if [ ! -f "$SELECTION_JSON" ]; then
    echo "ERROR: no selection was saved; flight cancelled." >&2
    exit 1
fi

mapfile -t TARGET_RECORDS < <(
    /usr/bin/python3 -c '
import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
targets = record.get("targets", [])
if not targets:
    raise SystemExit("Select at least one crop for an automatic flight")
if len(targets) > 4:
    raise SystemExit("A continuous mission is limited to four selected crops")
for target in targets:
    x, y, z = target["target_greenhouse_map"]
    print("\t".join(map(str, (
        target["target_id"], x, y, z, target["face"],
        target["row_index"], target["plant_index"]))))
' "$SELECTION_JSON")

TARGET_COUNT=${#TARGET_RECORDS[@]}
IFS=$'\t' read -r TARGET_ID TARGET_X TARGET_Y TARGET_Z CLICKED_FACE \
    TARGET_ROW TARGET_PLANT <<<"${TARGET_RECORDS[0]}"
echo "      selected ${TARGET_COUNT} crop target(s):"
printf '      %s\n' "${TARGET_RECORDS[@]}"
if [ "$TARGET_COUNT" -gt 1 ] && { [ "$SIDE_POLICY" != both ] || [ "$VIEW_COUNT" -ne 2 ]; }; then
    echo "ERROR: multi-target missions require --side both and --views 2." >&2
    exit 2
fi
if [ "$TARGET_COUNT" -gt 1 ] && [ "$VIEWPOINT_METHOD" != geometric ]; then
    echo "ERROR: multi-target missions currently require geometric viewpoint planning." >&2
    exit 2
fi
if [ "$TARGET_COUNT" -gt 1 ] && [ "$MISSION_TIMEOUT_EXPLICIT" -eq 0 ]; then
    # ROS 2 inferred a numeric parameter type from the literal.
    MISSION_TIMEOUT_SEC="$((240 * TARGET_COUNT)).0"
    echo "      multi-target simulated-time limit: ${MISSION_TIMEOUT_SEC} s"
fi
if [ "$MANUAL_VIEW" -eq 1 ]; then
    SIDE_POLICY=$CLICKED_FACE
fi
RUN_CONFIG=${RUN_DIR}/run_configuration.json
/usr/bin/python3 -c '
import hashlib, json, os, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

keys = ("run_id", "selection_mode", "viewpoint_method", "route_planner",
        "side_policy", "target_id", "cell_id", "distance_m", "pitch_deg",
        "view_count")
values = sys.argv[2:]
record = dict(zip(keys, values))

def tree_sha256(root):
    digest = hashlib.sha256()
    root = Path(root)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()

def revision(path):
    try:
        return subprocess.check_output(
            ["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None

selection = json.load(open(sys.argv[-1], encoding="utf-8"))
selected_targets = selection.get("targets", [])
record.update({
    "schema": "gps_greenhouse_run_configuration/v2",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "target_id": int(record["target_id"]),
    "cell_id": int(record["cell_id"]),
    "distance_m": float(record["distance_m"]),
    "pitch_deg": float(record["pitch_deg"]),
    "view_count": int(record["view_count"]),
    "target_count": len(selected_targets),
    "target_ids": [int(item["target_id"]) for item in selected_targets],
    "cell_ids": [int(item["operator_cell"]["cell_id"])
                 for item in selected_targets if item.get("operator_cell")],
    "safety_mode": "STATIC_MAP_ONLY",
    "world_name": "greenhouse_venlo",
    "localisation": "PX4 EKF using Gazebo simulated GPS and IMU",
    "slam_used": False,
    "cyan_highlight_used_for_planning": False,
    "software": {
        "ros_distro": os.environ.get("ROS_DISTRO"),
        "python": platform.python_version(),
        "px4_revision": revision(os.path.join(
            os.environ["THESIS_DIR"], "PX4-Autopilot")),
        "inspection_source_sha256": tree_sha256(
            os.path.join(os.environ["THESIS_DIR"],
                         "project/src/greenhouse_inspection")),
    },
})
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(record, stream, indent=2)
    stream.write("\n")
' "$RUN_CONFIG" "$RUN_ID" "$SELECTION_MODE" "$VIEWPOINT_METHOD" "$ROUTE_PLANNER" \
  "$SIDE_POLICY" "$TARGET_ID" "${CELL_ID_ARGS[0]:-0}" "$DISTANCE" "$PITCH_DEG" \
  "$((VIEW_COUNT * TARGET_COUNT))" \
  "$SELECTION_JSON"
echo "[3/9] Baking translucent cyan highlights onto the selected plant(s)..."
HIGHLIGHT_SPEC=$(/usr/bin/python3 -c '
import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
print(",".join("%d:%d" % (target["row_index"], target["plant_index"])
               for target in record["targets"]))
' "$SELECTION_JSON")
HIGHLIGHT_PLANTS=$HIGHLIGHT_SPEC \
    /usr/bin/python3 "${THESIS_DIR}/PX4-Autopilot/make_greenhouse_venlo.py"
/usr/bin/python3 -c '
import hashlib, json, sys
from pathlib import Path
config_path, world_path = map(Path, sys.argv[1:])
record = json.loads(config_path.read_text(encoding="utf-8"))
record["world_sdf_sha256"] = hashlib.sha256(world_path.read_bytes()).hexdigest()
config_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
' "$RUN_CONFIG" "${THESIS_DIR}/PX4-Autopilot/Tools/simulation/gz/worlds/${WORLD_NAME}.sdf"

echo "[4/9] Starting a clean ${WORLD_NAME} simulation in GPS mode..."
echo "      This intentionally stops any old greenhouse/teleop/SLAM run."
BOOTSTRAP_LOG=/tmp/gps_viewpoint_bootstrap.log
rm -f "$BOOTSTRAP_LOG"
# Start the simulator without attaching this script to its tmux session.
setsid env WORLD="${WORLD_NAME}" NO_ATTACH=1 bash "${THESIS_DIR}/run_greenhouse.sh" \
    >"$BOOTSTRAP_LOG" 2>&1 < /dev/null &

# Wait until all five simulator panes exist.
SIM_FOUND=0
for _ in $(seq 1 90); do
    if tmux has-session -t greenhouse 2>/dev/null; then
        PANE_COUNT=$(tmux list-panes -t greenhouse:sim 2>/dev/null | wc -l || true)
        if [ "${PANE_COUNT:-0}" -ge 5 ]; then
            SIM_FOUND=1
            break
        fi
    fi
    sleep 1
done
if [ "$SIM_FOUND" -ne 1 ]; then
    echo "ERROR: greenhouse tmux stack did not finish starting." >&2
    tail -n 80 "$BOOTSTRAP_LOG" >&2 || true
    exit 1
fi

wait_for_message() {
    local topic=$1
    local description=$2
    local attempts=${3:-90}
    # PX4 telemetry uses best-effort reliability.
    local qos_args=()
    if [ "$topic" = "/fmu/out/vehicle_odometry" ]; then
        qos_args=(--qos-reliability best_effort)
    fi
    local probe_file
    probe_file=$(mktemp /tmp/gps_viewpoint_probe.XXXXXX)
    for _ in $(seq 1 "$attempts"); do
        # Use a fresh CLI process after each simulator restart.
        if timeout 7 ros2 topic echo "$topic" --once --no-daemon "${qos_args[@]}" \
                >"$probe_file" 2>/dev/null && [ -s "$probe_file" ]; then
            rm -f "$probe_file"
            echo "      ready: ${description}"
            return 0
        fi
        : >"$probe_file"
        sleep 2
    done
    rm -f "$probe_file"
    echo "ERROR: timed out waiting for ${description} on ${topic}." >&2
    return 1
}

wait_for_message /fmu/out/vehicle_odometry "PX4 GPS-backed odometry"
echo "      safety mode: static known map only"
wait_for_message /camera "camera images"

if ! tmux capture-pane -p -t greenhouse:sim.4 -S -30 2>/dev/null \
        | grep -q "Press ENTER to start the waypoint flight"; then
    echo "ERROR: the unrelated waypoint pane is not safely paused." >&2
    echo "Inspect it with: tmux attach -t greenhouse" >&2
    exit 1
fi

echo "[5/9] Starting map alignment and gimbal control..."
tmux kill-window -t greenhouse:gpsruntime 2>/dev/null || true
tmux new-window -d -t greenhouse -n gpsruntime -c "$WORKSPACE_DIR" \
    "bash -lc 'source /opt/ros/humble/setup.bash && source ${WORKSPACE_DIR}/install/setup.bash && exec ros2 launch greenhouse_inspection inspection_runtime.launch.py'"
sleep 3
if ! tmux capture-pane -p -t greenhouse:gpsruntime -S -40 \
        | grep -q "Gimbal bench controller ready"; then
    echo "ERROR: gimbal runtime did not become ready." >&2
    tmux capture-pane -p -t greenhouse:gpsruntime -S -80 >&2 || true
    exit 1
fi

PREFLIGHT_TARGET_ID=$TARGET_ID
if [ "$TARGET_COUNT" -gt 1 ]; then
    PREFLIGHT_TARGET_ID=0
fi
if [ "$VIEWPOINT_METHOD" = fixed ]; then
    echo "      fixed baseline: distance=${DISTANCE} m, pitch=${PITCH_DEG} deg, side=${SIDE_POLICY}"
    echo "[6/9] Building fixed-view GPS preflight(s)..."
    ros2 run greenhouse_inspection gps_preflight --ros-args \
        -p selection_json:="$SELECTION_JSON" \
        -p output_dir:="$PREFLIGHT_DIR" \
        -p target_id:="$PREFLIGHT_TARGET_ID"
else
    echo "      geometric mode: distances, pitches and yaws will be searched; side=${SIDE_POLICY}"
    echo "[6/9] Optimising requested shot(s) from PX4's GPS-backed pose..."
    ros2 run greenhouse_inspection gps_auto_preflight --ros-args \
        -p selection_json:="$SELECTION_JSON" \
        -p output_dir:="$PREFLIGHT_DIR" \
        -p target_id:="$PREFLIGHT_TARGET_ID" \
        -p side_policy:="$SIDE_POLICY" \
        -p route_planner:="$ROUTE_PLANNER" \
        -p views_per_side:="$((VIEW_COUNT / 2))"
fi

MISSION_PREFLIGHT_LIST=${PREFLIGHT_DIR}/mission_preflight_list.json
ros2 run greenhouse_inspection build_mission_plan \
    --selection-json "$SELECTION_JSON" \
    --preflight-dir "$PREFLIGHT_DIR" \
    --output "$MISSION_PREFLIGHT_LIST" \
    --method "$VIEWPOINT_METHOD" \
    --side "$SIDE_POLICY" \
    --views-per-target "$VIEW_COUNT"
mapfile -t PREFLIGHT_JSONS < <(
    /usr/bin/python3 -c '
import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
print("\n".join(record["preflight_jsons"]))
' "$MISSION_PREFLIGHT_LIST")

summarise_preflight() {
    local preflight_json=$1
    /usr/bin/python3 -c '
import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
if record.get("status") != "PREFLIGHT_READY":
    raise SystemExit("Preflight is not ready: " + str(record.get("detail")))
route = record["route"]
print("      preflight accepted: %.3f m, %d reviewed waypoints" %
      (route["length_m"], route["waypoint_count_including_start"]))
auto = record.get("auto_viewpoint")
if auto:
    print("      automatic choice: side=%s, distance=%.2f m, pitch=%.1f deg, "
          "azimuth=%.1f deg, camera_z=%.2f m" %
          (auto["selected_side"], auto["selected_distance_m"],
           auto["selected_pitch_deg"], auto["selected_azimuth_deg"],
           auto["selected_camera_height_m"]))
    projection = auto["predicted_plant_projection"]
    print("      predicted plant: %.1f%% of image area; fully framed=%s" %
          (100.0 * projection["image_area_fraction"],
           projection["fully_inside_92_percent_fov"]))
' "$preflight_json"
}

dry_validate() {
    local preflight_json=$1
    local dry_result=$2
    ros2 run greenhouse_inspection execute_viewpoint --ros-args \
        -p preflight_json:="$preflight_json" \
        -p result_json:="$dry_result" \
        -p use_lidar_safety:=false
    /usr/bin/python3 -c '
import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
if (record.get("status") != "EXECUTION_DISABLED"
        or record.get("flight_commanded") is not False):
    raise SystemExit("Dry-run interlock validation failed")
print("      dry-run accepted; no flight was commanded")
' "$dry_result"
}

for preflight_json in "${PREFLIGHT_JSONS[@]}"; do
    if [ ! -f "$preflight_json" ]; then
        echo "ERROR: expected preflight was not generated: $preflight_json" >&2
        exit 1
    fi
    summarise_preflight "$preflight_json"
done

echo "[7/9] Passing every artifact through the non-arming executor check..."
for preflight_json in "${PREFLIGHT_JSONS[@]}"; do
    shot_name=$(basename "$preflight_json" _preflight.json)
    dry_validate "$preflight_json" "${RUN_DIR}/dry_run_${shot_name}.json"
done

if [ "${#PREFLIGHT_JSONS[@]}" -ge 2 ]; then
    COMBINED_DRY_RESULT=${RUN_DIR}/dry_run_multi_viewpoint_mission.json
    COMBINED_DRY_CSV=${RUN_DIR}/dry_run_multi_viewpoint_mission.csv
    ros2 run greenhouse_inspection execute_two_viewpoints --ros-args \
        -p preflight_list_json:="$MISSION_PREFLIGHT_LIST" \
        -p result_json:="$COMBINED_DRY_RESULT" \
        -p result_csv:="$COMBINED_DRY_CSV" \
        -p use_lidar_safety:=false
    /usr/bin/python3 -c '
import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
if (record.get("status") != "EXECUTION_DISABLED"
        or record.get("flight_commanded") is not False):
    raise SystemExit("Combined-mission dry-run validation failed")
planned = record["planned"]
print("      combined one-flight route accepted: %.3f m; clearance %.3f m" %
      (planned["total_route_length_before_landing_m"],
       planned["minimum_clearance_to_inflated_obstacles_m"]))
' "$COMBINED_DRY_RESULT"
fi

if [ "$FLY" -ne 1 ]; then
    echo "[8/9] Flight not requested. The reviewed run is ready at:"
    echo "      $RUN_DIR"
    echo "Run this script with --fly when you want automatic Gazebo execution."
    exit 0
fi


echo "[8/9] --fly was explicitly supplied. Gazebo flight begins in 5 seconds."
echo "      Press Ctrl-C now to cancel; during flight Ctrl-C requests landing."
if [ "${#PREFLIGHT_JSONS[@]}" -eq 1 ]; then
    echo "      Single-side mode performs one selected-side sortie."
else
    echo "      ${#PREFLIGHT_JSONS[@]}-view mode uses ONE takeoff and one final landing."
fi
for seconds in 5 4 3 2 1; do
    echo "      ${seconds}..."
    sleep 1
done

FLIGHT_RESULTS=()
if [ "${#PREFLIGHT_JSONS[@]}" -eq 1 ]; then
    preflight_json=${PREFLIGHT_JSONS[0]}
    shot_name=$(basename "$preflight_json" _preflight.json)
    flight_result=${RUN_DIR}/flight_result_${shot_name}.json
    FLIGHT_RESULTS+=("$flight_result")
    echo "      Starting manual view: ${shot_name}"
    ros2 run greenhouse_inspection execute_viewpoint --ros-args \
        -p preflight_json:="$preflight_json" \
        -p result_json:="$flight_result" \
        -p out_dir:="$PHOTO_DIR" \
        -p start_flight:=true \
        -p return_after_capture:=true \
        -p land_after_return:=true \
        -p use_lidar_safety:=false

    /usr/bin/python3 -c '
import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
execution = record.get("execution", {})
print("      inspection status:", record.get("status"))
print("      image:", execution.get("image_path"))
print("      direction error (deg):",
      execution.get("achieved_direction_error_deg"))
print("      distance error (m):",
      execution.get("achieved_distance_error_m"))
' "$flight_result"
else
    mission_result=${RUN_DIR}/multi_viewpoint_mission_result.json
    mission_csv=${RUN_DIR}/multi_viewpoint_mission_results.csv
    FLIGHT_RESULTS+=("$mission_result")
    echo "      Starting continuous ${#PREFLIGHT_JSONS[@]}-view mission..."
    ros2 run greenhouse_inspection execute_two_viewpoints --ros-args \
        -p preflight_list_json:="$MISSION_PREFLIGHT_LIST" \
        -p result_json:="$mission_result" \
        -p result_csv:="$mission_csv" \
        -p out_dir:="$PHOTO_DIR" \
        -p start_flight:=true \
        -p mission_timeout_sec:="$MISSION_TIMEOUT_SEC" \
        -p use_lidar_safety:=false
    /usr/bin/python3 -c '
import json, sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
execution = record.get("execution", {})
print("      mission status:", record.get("status"))
print("      images captured: %s/%s" % (
    execution.get("images_captured", 0), record.get("view_count")))
print("      simulated mission duration (s):",
      execution.get("mission_duration_sim_s"))
print("      wall-clock runtime (s):",
      execution.get("mission_duration_wall_s"))
print("      actual path length (m):", execution.get("actual_path_length_m"))
print("      actual minimum clearance (m):",
      execution.get("minimum_clearance_to_inflated_obstacles_m"))
print("      landing outcome:", execution.get("landing_outcome"))
for shot in record.get("shots", []):
    print("      target %s %s: %s | image=%s | optical error=%s deg" % (
        shot.get("target_id"), shot.get("side"), shot.get("status"), shot.get("image_path"),
        shot.get("optical_axis_error_deg")))
' "$mission_result"
fi

echo "[9/9] Requested viewpoint mission finished."
echo
echo "Complete run directory: $RUN_DIR"
echo "Selection map:         ${SELECTION_DIR}/selected_targets.png"
for flight_result in "${FLIGHT_RESULTS[@]}"; do
    echo "Flight result:         $flight_result"
done
echo "Gazebo/tmux:           tmux attach -t greenhouse"
