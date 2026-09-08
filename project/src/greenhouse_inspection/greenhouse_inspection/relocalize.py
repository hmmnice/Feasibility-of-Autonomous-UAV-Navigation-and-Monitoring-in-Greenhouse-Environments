"""Teach-and-repeat relocalization: align a new flight's FAST-LIO frame to a map recorded on an earlier flight."""

import math
import os

import numpy as np
from scipy.spatial import cKDTree

# --- map / cloud handling -------------------------------------------------
MAP_VOXEL = 0.10        # m, saved-map resolution
# Accumulate every Nth scan only.
ACCUM_EVERY = 5
LIVE_VOXEL = 0.15       # m, live cloud fed to ICP (coarser: ICP cost is linear in point count)
MIN_RANGE = 0.3         # m, matches preprocess.blind in config/greenhouse.yaml
MAX_RANGE = 30.0        # m, matches mapping.det_range

# search prior --------------------------------------------------------- Seeds must sit closer together than the distance ICP can pull.
SEARCH_XY = 1.2         # m, half-width of the launch-pose search box
SEARCH_XY_STEP = 0.6    # m
SEARCH_YAW = 30.0       # deg, half-width of the launch-heading search
SEARCH_YAW_STEP = 15.0  # deg

# --- ICP ------------------------------------------------------------------
ICP_ITERS = 30
CORR_DIST_START = 1.0   # m, correspondence gate on iteration 0
CORR_DIST_END = 0.2     # m, correspondence gate on the last iteration
# What counts as a match when scoring a hypothesis.
INLIER_DIST = 0.10      # m
# ICP cost is linear in live points and we run it once per seed, so the search runs.
MAX_SEARCH_POINTS = 1500

# --- accept / reject ------------------------------------------------------
MIN_FITNESS = 0.45      # reject outright: too little of the live cloud matched
# Sits in the measured gap between correct answers (28-38%) and genuinely
# ambiguous ones (5-8%), biased towards rejecting.
AMBIGUITY_MARGIN = 0.15
DISTINCT_XY = 0.5       # m, hypotheses closer than this are the same answer
DISTINCT_YAW = 10.0     # deg

# Repeat pass: wait for BOTH before solving.
WARMUP_SEC = 5.0
MIN_WARMUP_TRAVEL = 1.0  # m of path flown before there is enough parallax to solve


def traj_path(map_path):
    """Where the teach pass's flown route sits, given where its map sits."""
    return os.path.splitext(map_path)[0] + "_traj.npy"


def voxel_downsample(points, voxel):
    """One point per occupied voxel."""
    if len(points) == 0:
        return points
    _, idx = np.unique(np.floor(points / voxel).astype(np.int64), axis=0, return_index=True)
    return points[idx]


def yaw_transform(x, y, z, yaw):
    """4x4 rigid transform with rotation about z only."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([
        [c, -s, 0.0, x],
        [s, c, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0],
    ])


def transform_points(points, T):
    return points @ T[:3, :3].T + T[:3, 3]


def decompose(T):
    """(x, y, z, yaw_radians) from a 4-DOF transform."""
    return T[0, 3], T[1, 3], T[2, 3], math.atan2(T[1, 0], T[0, 0])


def fit_yaw_rigid(src, dst):
    """Least-squares taking src onto dst, in closed form."""
    sc, dc = src.mean(axis=0), dst.mean(axis=0)
    s, d = src - sc, dst - dc
    yaw = math.atan2(
        np.sum(s[:, 0] * d[:, 1] - s[:, 1] * d[:, 0]),
        np.sum(s[:, 0] * d[:, 0] + s[:, 1] * d[:, 1]),
    )
    T = yaw_transform(0.0, 0.0, 0.0, yaw)
    T[:3, 3] = dc - T[:3, :3] @ sc
    return T


def icp(live, map_points, map_tree, T_init, iters=ICP_ITERS):
    """Point-to-point ICP constrained to 4 DOF."""
    T = np.array(T_init, dtype=float)
    # Wide correspondence gate first so a seed can walk in from a metre out, tightened geometrically so.
    gates = np.geomspace(CORR_DIST_START, CORR_DIST_END, iters)

    dist = None
    for gate in gates:
        moved = transform_points(live, T)
        dist, idx = map_tree.query(moved, workers=-1)
        keep = dist < gate
        if keep.sum() < 20:
            break
        step = fit_yaw_rigid(moved[keep], map_points[idx[keep]])
        T = step @ T
        # Converged: the correction this iteration is below the map resolution.
        if np.linalg.norm(step[:3, 3]) < 1e-4 and abs(step[1, 0]) < 1e-4:
            break

    dist, _ = map_tree.query(transform_points(live, T), workers=-1)
    inliers = dist < INLIER_DIST
    fitness = float(inliers.mean())
    rmse = float(np.sqrt(np.mean(dist[inliers] ** 2))) if inliers.any() else float("inf")
    return T, fitness, rmse


def _symmetric_steps(half_width, step):
    """Offsets spanning +/-half_width that always include 0."""
    half = np.arange(0.0, half_width + 1e-9, step)
    return np.unique(np.concatenate([-half, half]))


def seed_transforms(prior=None):
    """Grid of initial guesses around the taught launch pose."""
    base = np.eye(4) if prior is None else np.asarray(prior, dtype=float)
    steps = _symmetric_steps(SEARCH_XY, SEARCH_XY_STEP)
    yaws = np.radians(_symmetric_steps(SEARCH_YAW, SEARCH_YAW_STEP))
    # base @ offset, not offset @ base.
    return [base @ yaw_transform(dx, dy, 0.0, dyaw)
            for dyaw in yaws for dx in steps for dy in steps]


def _same_answer(a, b):
    ax, ay, _, ayaw = decompose(a)
    bx, by, _, byaw = decompose(b)
    dyaw = abs(math.degrees(math.atan2(math.sin(ayaw - byaw), math.cos(ayaw - byaw))))
    return math.hypot(ax - bx, ay - by) < DISTINCT_XY and dyaw < DISTINCT_YAW


def relocalize(live, map_points, prior=None):
    """Find the transform putting the live SLAM frame into the saved map frame."""
    live = voxel_downsample(np.asarray(live, dtype=float), LIVE_VOXEL)
    map_points = np.asarray(map_points, dtype=float)
    if len(live) < 100 or len(map_points) < 100:
        return {"transform": None, "reason": "not enough points",
                "fitness": 0.0, "rmse": float("inf"), "runner_up": 0.0, "margin": 0.0}

    tree = cKDTree(map_points)
    coarse = live[:: max(1, len(live) // MAX_SEARCH_POINTS)]
    results = [icp(coarse, map_points, tree, T0) for T0 in seed_transforms(prior)]
    results.sort(key=lambda r: -r[1])

    # Runner-up = best-scoring hypothesis that is a genuinely DIFFERENT answer, not just the same optimum reached.
    best_T = results[0][0]
    runner_up = next((f for T, f, _ in results[1:] if not _same_answer(T, best_T)), 0.0)

    best_T, best_fit, best_rmse = icp(live, map_points, tree, best_T)
    margin = (best_fit - runner_up) / best_fit if best_fit > 0 else 0.0

    out = {"transform": best_T, "fitness": best_fit, "rmse": best_rmse,
           "runner_up": runner_up, "margin": margin, "reason": ""}
    if best_fit < MIN_FITNESS:
        out["transform"], out["reason"] = None, f"fitness {best_fit:.2f} < {MIN_FITNESS}"
    elif margin < AMBIGUITY_MARGIN:
        out["transform"], out["reason"] = None, (
            f"ambiguous: runner-up {runner_up:.2f} vs best {best_fit:.2f} "
            f"(margin {margin:.1%} < {AMBIGUITY_MARGIN:.0%}) -- likely adjacent row")
    return out


# ROS node.

def _cloud_to_xyz(msg):
    """x/y/z out of a PointCloud2 without sensor_msgs_py."""
    # Slicing by fixed offset is only safe while the layout really is three contiguous float32s at the front.
    layout = [(f.name, f.offset, f.datatype) for f in msg.fields[:3]]
    if layout != [("x", 0, 7), ("y", 4, 7), ("z", 8, 7)]:
        raise ValueError(
            f"unexpected PointCloud2 layout {layout}; expected float32 x,y,z at 0,4,8")

    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
    xyz = raw[:, 0:12].copy().view(np.float32).reshape(-1, 3).astype(np.float64)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    d = np.linalg.norm(xyz, axis=1)
    return xyz[(d > MIN_RANGE) & (d < MAX_RANGE)]


def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2
    from tf2_ros import StaticTransformBroadcaster

    # lidar_sensor_link -> base_link/imu, from config/greenhouse.yaml mapping.extrinsic_T.
    T_IMU_LIDAR = np.eye(4)
    T_IMU_LIDAR[:3, 3] = [0.0, 0.0, 0.18]

    class Relocalizer(Node):
        def __init__(self):
            super().__init__("relocalizer")
            default_map = "/root/masters-thesis/teach_map.npy"
            self.map_path = self.declare_parameter("map_path", default_map).value
            self.save_map = self.declare_parameter("save_map", False).value

            sub_qos = QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
            )
            self.create_subscription(PointCloud2, "/lidar/points", self.lidar_cb, sub_qos)
            self.create_subscription(Odometry, "/odometry", self.odom_cb, sub_qos)

            self.tf_broadcaster = StaticTransformBroadcaster(self)
            self.pose = None          # latest FAST-LIO map->imu as 4x4
            self.traj = []            # teach pass only: (x, y, z, yaw) per odom sample
            self.chunks = []
            self.scan_count = 0
            self.start_time = None
            self.travelled = 0.0
            self.last_position = None
            self.done = False

            if self.save_map:
                self.get_logger().info(
                    f"TEACH: accumulating map + flown route, will write "
                    f"{self.map_path} and {traj_path(self.map_path)} on exit.")
            else:
                self.saved_map = np.load(self.map_path)
                self.get_logger().info(
                    f"REPEAT: loaded {len(self.saved_map)} map points from {self.map_path}; "
                    f"relocalizing after {WARMUP_SEC:.0f}s of scans.")

        def odom_cb(self, msg):
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            # quaternion -> rotation matrix, inline to keep this node's only heavy dependency scipy-in-the-algorithm, not here.
            x, y, z, w = q.x, q.y, q.z, q.w
            self.pose = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), p.x],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), p.y],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), p.z],
                [0.0, 0.0, 0.0, 1.0],
            ])
            # Record raw, at the full ~250Hz.
            if self.save_map:
                self.traj.append(decompose(self.pose))

        def lidar_cb(self, msg):
            # Before FAST-LIO's first odometry there is no frame to put scans
            # in -- drop them rather than stack them at the origin.
            if self.pose is None or self.done:
                return
            now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if self.start_time is None:
                self.start_time = now

            # Path length, sampled at scan rate rather than the 250Hz odometry so per-sample jitter doesn't accumulate into fake.
            position = self.pose[:3, 3]
            if self.last_position is not None:
                self.travelled += float(np.linalg.norm(position - self.last_position))
            self.last_position = position

            # Only every ACCUM_EVERY'th scan is accumulated, and each is downsampled on arrival rather than raw-stacked.
            self.scan_count += 1
            if self.scan_count % ACCUM_EVERY:
                return
            T = self.pose @ T_IMU_LIDAR
            self.chunks.append(
                voxel_downsample(transform_points(_cloud_to_xyz(msg), T), MAP_VOXEL))
            # Keep memory flat over a long teach flight; MAP_VOXEL is the
            # resolution we save at anyway, so this loses nothing.
            if len(self.chunks) >= 50:
                self.chunks = [voxel_downsample(np.vstack(self.chunks), MAP_VOXEL)]

            if self.save_map or now - self.start_time < WARMUP_SEC:
                return
            if self.travelled < MIN_WARMUP_TRAVEL:
                self.get_logger().warn(
                    f"Waiting to relocalize: flown {self.travelled:.2f}m of the "
                    f"{MIN_WARMUP_TRAVEL:.1f}m needed. A stationary drone sees one "
                    f"viewpoint, which fits several rows equally well -- fly a short leg.",
                    throttle_duration_sec=5.0)
                return
            self.done = True
            self.solve()

        def cloud(self):
            if not self.chunks:
                return np.zeros((0, 3))
            return voxel_downsample(np.vstack(self.chunks), MAP_VOXEL)

        def solve(self):
            result = relocalize(self.cloud(), self.saved_map)
            if result["transform"] is None:
                self.get_logger().error(
                    f"RELOCALIZATION REJECTED -- {result['reason']}. Not publishing a "
                    f"transform. Re-launch closer to the taught start pose.")
                return
            T = result["transform"]
            x, y, z, yaw = decompose(T)
            self.get_logger().info(
                f"Relocalized: x={x:.3f} y={y:.3f} z={z:.3f} yaw={math.degrees(yaw):.2f}deg "
                f"fitness={result['fitness']:.3f} rmse={result['rmse']:.3f} "
                f"margin={result['margin']:.1%}")

            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "saved_map"
            t.child_frame_id = "map"
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = z
            t.transform.rotation.z, t.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            self.tf_broadcaster.sendTransform(t)

        def save_teach_pass(self):
            pts = self.cloud()
            if len(pts) == 0:
                self.get_logger().warn("No scans accumulated -- nothing to save.")
                return
            np.save(self.map_path, pts.astype(np.float32))
            self.get_logger().info(f"Saved {len(pts)} map points to {self.map_path}.")

            # Both or neither: a map with no route cannot be repeated, and a
            # route with no map cannot be relocalized into.
            if not self.traj:
                self.get_logger().error(
                    "No odometry received -- map saved but NO route. The repeat "
                    "pass has nothing to fly; re-run the teach pass with "
                    "FAST-LIO2 publishing /odometry.")
                return
            np.save(traj_path(self.map_path), np.asarray(self.traj, dtype=np.float32))
            self.get_logger().info(
                f"Saved {len(self.traj)} route samples to {traj_path(self.map_path)}.")

    rclpy.init(args=args)
    node = Relocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.save_map:
            node.save_teach_pass()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
