#!/usr/bin/env bash
# Install the project-specific files into a compatible PX4 checkout.
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PX4_DIR=${1:-"${ROOT_DIR}/PX4-Autopilot"}
OVERLAY_DIR="${ROOT_DIR}/px4_overlay"

if [ ! -f "${PX4_DIR}/ROMFS/px4fmu_common/init.d-posix/airframes/CMakeLists.txt" ]; then
    echo "PX4 checkout not found: ${PX4_DIR}" >&2
    exit 1
fi

cp -a "${OVERLAY_DIR}/make_greenhouse_venlo.py" "${PX4_DIR}/"
cp -a "${OVERLAY_DIR}/ROMFS/." "${PX4_DIR}/ROMFS/"
cp -a "${OVERLAY_DIR}/Tools/." "${PX4_DIR}/Tools/"

AIRFRAMES="${PX4_DIR}/ROMFS/px4fmu_common/init.d-posix/airframes/CMakeLists.txt"
if ! grep -q '4022_gz_x500_lidar_3d' "${AIRFRAMES}"; then
    sed -i '/4021_gz_x500_flow/a\	4022_gz_x500_lidar_3d' "${AIRFRAMES}"
fi

GZ_INIT="${PX4_DIR}/ROMFS/px4fmu_common/init.d-posix/px4-rc.gzsim"
if ! grep -q -- '--headless-rendering' "${GZ_INIT}"; then
    sed -i 's/-r -s "${PX4_GZ_WORLDS}/-r -s --headless-rendering "${PX4_GZ_WORLDS}/' "${GZ_INIT}"
fi

python3 "${PX4_DIR}/make_greenhouse_venlo.py"
echo "PX4 overlay installed in ${PX4_DIR}"
