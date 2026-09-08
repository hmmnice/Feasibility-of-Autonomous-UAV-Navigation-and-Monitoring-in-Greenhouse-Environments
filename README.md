# Feasibility of Autonomous UAV Navigation and Monitoring in Greenhouse Environments

This folder contains the custom code and simulation assets used for the
dissertation. It does not include ROS 2, PX4, Gazebo or the Micro XRCE-DDS
Agent because those projects are too large to duplicate in a submission.

The main program opens a Matplotlib selection window, associates the selected
map location with a crop, generates camera viewpoints and collision-checked
routes, starts PX4 SITL and Gazebo, and captures two crop images in one flight.

## Contents

- `project/src/greenhouse_inspection`: the complete inspection package: the
  operator interface, viewpoint search, routing, geometry, safety, gimbal
  control and PX4 mission execution.
- `px4_overlay`: the Venlo world generator, crop meshes, UAV sensor model and
  custom PX4 airframe. The greenhouse world is generated during installation.

## Tested environment

The evaluation used:

- Ubuntu 22.04 in the ROS container;
- ROS 2 Humble;
- Gazebo Harmonic (`gz-sim` 8);
- PX4 ;
- `px4_msgs` v1.17.0;
- Micro XRCE-DDS Agent v2.4.3; and
- Python 3 with NumPy, Matplotlib, SciPy and trimesh.

An NVIDIA 3060 GPU was used to run the tests. 

## Installation

Install ROS 2 Humble, Gazebo Harmonic and the normal PX4 Ubuntu dependencies
first. The ROS/Gazebo bridge must match Harmonic; the Humble package is
`ros-humble-ros-gzharmonic`.

From the extracted submission directory, clone the external projects:

```bash
git clone --recursive https://github.com/PX4/PX4-Autopilot.git
git -C PX4-Autopilot checkout d6f12ad1c4f70ad3230afd7d86e971421e02fef4

git clone https://github.com/PX4/px4_msgs.git project/src/px4_msgs
git -C project/src/px4_msgs checkout v1.17.0

git clone https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
git -C Micro-XRCE-DDS-Agent checkout v2.4.3
```

Build and install the DDS agent according to its upstream instructions, then
confirm that `MicroXRCEAgent` is available on `PATH`.

Install the Python dependencies and apply the custom PX4 files:

```bash
python3 -m pip install -r requirements.txt
bash install_px4_overlay.sh
```

Build the ROS workspace:

```bash
source /opt/ros/humble/setup.bash
cd project
colcon build --symlink-install \
  --packages-select px4_msgs greenhouse_inspection
source install/setup.bash
cd ..
```

## Run the inspection pipeline

Run from a graphical terminal in the configured environment:

```bash
bash project/src/greenhouse_inspection/run_greenhouse_inspection.sh --fly
```

The Matplotlib window is the top-down crop-selection interface shown in the
thesis; it is not MATLAB. Left-click a crop, right-click to undo, and press
Enter to confirm. The program then generates and reviews the two viewpoints,
starts the simulation, performs one take-off, visits the nearest reachable
viewpoint first, flies directly between the viewpoints, and returns for one
landing.

Omit `--fly` to perform planning and validation without arming the simulated
vehicle:

```bash
bash project/src/greenhouse_inspection/run_greenhouse_inspection.sh
```

Use a known catalogue identifier to skip the graphical selection:

```bash
bash project/src/greenhouse_inspection/run_greenhouse_inspection.sh \
  --fly --target-id 268
```

Compare the two route planners with the same target:

```bash
bash project/src/greenhouse_inspection/run_greenhouse_inspection.sh \
  --fly --target-id 115 --route-planner structured

bash project/src/greenhouse_inspection/run_greenhouse_inspection.sh \
  --fly --target-id 115 --route-planner astar
```

Run a three-crop mission:

```bash
bash project/src/greenhouse_inspection/run_greenhouse_inspection.sh \
  --fly --target-id 85 --target-id 268 --target-id 395
```

Every run is stored under `gps_viewpoint_runs/<run-id>/`. The folder contains
the selection record, preflight plans, route arrays, captured images, mission
JSON and one CSV row per capture.

## Safety scope

The submitted evaluation uses the predefined static map. Candidate viewpoints
and every route segment are checked against inflated crop and structural
obstacles. All flight commands here control PX4 SITL; physical flight requires separate hardware validation and
safety procedures.

## Asset licence

The tomato meshes were generated from the Apache-2.0 AOC tomato farm assets.
Their licence is included at `licenses/AOC_TOMATO_FARM_LICENSE`.
