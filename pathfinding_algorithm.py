import cv2
import openvino as ov
import numpy as np
import rerun as rr
import threading
import queue
from scipy.interpolate import splprep, splev

# ── Setup Rerun ───────────────────────────────────────────────────────────────
rr.init("intel_igpu_3d_mapping", spawn=True)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

# ── Camera & iGPU ─────────────────────────────────────────────────────────────
width, height = 640, 480
focal_length  = width * 0.8

core           = ov.Core()
compiled_model = core.compile_model("Test Env/depth_anything_v2_vitb.xml", "GPU")

rr.log("world/camera", rr.Transform3D(translation=[0, 0, 0]), static=True)
rr.log(
    "world/camera",
    rr.Pinhole(
        resolution=[width, height],
        focal_length=focal_length,
        camera_xyz=rr.ViewCoordinates.RDF,
    ),
    static=True,
)

# ── Depth config ──────────────────────────────────────────────────────────────
DEPTH_MIN_M = 0.3
DEPTH_MAX_M = 4.0

# ── Floor / obstacle config ───────────────────────────────────────────────────
Y_BINS              = 200
FLOOR_SLAB_M        = 0.08
FLOOR_COLOR         = np.array([0, 200, 0],   dtype=np.uint8)
OBSTACLE_SLAB_M     = 0.10
OBSTACLE_SLABS      = 2
OBSTACLE_COLOR      = np.array([255, 0, 0],   dtype=np.uint8)
XZ_BIN_SIZE         = 0.05
MIN_OBSTACLE_POINTS = 3

# ── Pathfinding config ────────────────────────────────────────────────────────
GRID_RESOLUTION          = 0.1
GRID_X_MIN, GRID_X_MAX   = -2.0, 2.0
GRID_Z_MIN, GRID_Z_MAX   = 0.5, 4.0
PATH_COLOR               = np.array([0, 0, 255], dtype=np.uint8)
GOAL_DISTANCE            = 3.0

fx = fy = focal_length
cx, cy  = width / 2.0, height / 2.0

# ── Pipeline queues ───────────────────────────────────────────────────────────
RAW_Q    = queue.Queue(maxsize=2)
DEPTH_Q  = queue.Queue(maxsize=2)
LOG_Q    = queue.Queue(maxsize=2)
SENTINEL = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def infer_depth(frame: np.ndarray) -> np.ndarray:
    inp = cv2.resize(frame, (518, 518))
    inp = inp.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    raw   = compiled_model([inp])[compiled_model.output(0)].squeeze()
    depth = cv2.resize(raw, (width, height))

    d_min, d_max = depth.min(), depth.max()
    disp_norm    = (depth - d_min) / (d_max - d_min + 1e-6)
    a = 1.0 / DEPTH_MIN_M - 1.0 / DEPTH_MAX_M
    b = 1.0 / DEPTH_MAX_M
    depth_metric = (1.0 / (a * disp_norm + b)).astype(np.float32)
    depth_metric[(depth_metric < DEPTH_MIN_M) | (depth_metric > DEPTH_MAX_M)] = 0.0
    return depth_metric


# Pre-compute the ray-scale grid once
_u  = np.arange(width,  dtype=np.float32)
_v  = np.arange(height, dtype=np.float32)
_uu, _vv = np.meshgrid(_u, _v)
_RAY_SCALE = np.sqrt(
    ((_uu - cx) / fx) ** 2 +
    ((_vv - cy) / fy) ** 2 +
    1.0
)

def backproject(depth_metric: np.ndarray):
    valid       = depth_metric > 0.0
    z_corrected = depth_metric / _RAY_SCALE
    x = (_uu - cx) / fx * z_corrected
    y = (_vv - cy) / fy * z_corrected
    return np.stack([x[valid], y[valid], z_corrected[valid]], axis=-1), valid


def detect_floor_histogram(points: np.ndarray):
    y_vals = points[:, 1]
    y_min, y_max = y_vals.min(), y_vals.max()
    if y_max - y_min < 1e-3:
        return np.zeros(len(points), dtype=bool), 0.0

    counts, bin_edges = np.histogram(y_vals, bins=Y_BINS)
    best_bin       = np.argmax(counts)
    floor_y_centre = (bin_edges[best_bin] + bin_edges[best_bin + 1]) / 2.0
    floor_mask = (
        (y_vals >= floor_y_centre - FLOOR_SLAB_M) &
        (y_vals <= floor_y_centre + FLOOR_SLAB_M)
    )
    return floor_mask, floor_y_centre


def _xz_keys(pts: np.ndarray) -> np.ndarray:
    xi = np.floor(pts[:, 0] / XZ_BIN_SIZE).astype(np.int32)
    zi = np.floor(pts[:, 2] / XZ_BIN_SIZE).astype(np.int32)
    return xi.astype(np.int64) * 1_000_003 + zi.astype(np.int64)


def detect_obstacles_above_floor(
    points: np.ndarray,
    floor_mask: np.ndarray,
    floor_y_centre: float,
):
    y_vals = points[:, 1]

    above_floor_top   = floor_y_centre - FLOOR_SLAB_M
    obstacle_zone_top = above_floor_top - OBSTACLE_SLABS * OBSTACLE_SLAB_M

    candidate_mask = (y_vals >= obstacle_zone_top) & (y_vals < above_floor_top)
    if candidate_mask.sum() == 0:
        return np.zeros(len(points), dtype=bool), floor_mask.copy()

    cand_pts  = points[candidate_mask]

    cand_keys = _xz_keys(cand_pts)
    unique_keys, counts = np.unique(cand_keys, return_counts=True)
    obstacle_keys = unique_keys[counts >= MIN_OBSTACLE_POINTS]

    if len(obstacle_keys) == 0:
        return np.zeros(len(points), dtype=bool), floor_mask.copy()

    obstacle_mask = np.zeros(len(points), dtype=bool)
    cand_in_obs   = np.isin(cand_keys, obstacle_keys)
    cand_indices  = np.where(candidate_mask)[0]
    obstacle_mask[cand_indices[cand_in_obs]] = True

    floor_mask_clean = floor_mask.copy()
    floor_pts = points[floor_mask]
    if len(floor_pts) > 0:
        floor_keys    = _xz_keys(floor_pts)
        floor_blocked = np.isin(floor_keys, obstacle_keys)
        floor_indices = np.where(floor_mask)[0]
        floor_mask_clean[floor_indices[floor_blocked]] = False

    return obstacle_mask, floor_mask_clean


# ── Occupancy grid ────────────────────────────────────────────────────────────

def build_occupancy_grid(points: np.ndarray, floor_mask: np.ndarray, obstacle_mask: np.ndarray):
    n_x = int((GRID_X_MAX - GRID_X_MIN) / GRID_RESOLUTION) + 1
    n_z = int((GRID_Z_MAX - GRID_Z_MIN) / GRID_RESOLUTION) + 1

    grid = np.full((n_z, n_x), 2, dtype=np.uint8)  # 2 = unknown

    if floor_mask.any():
        floor_pts = points[floor_mask]
        xi = ((floor_pts[:, 0] - GRID_X_MIN) / GRID_RESOLUTION).astype(int)
        zi = ((floor_pts[:, 2] - GRID_Z_MIN) / GRID_RESOLUTION).astype(int)
        valid = (xi >= 0) & (xi < n_x) & (zi >= 0) & (zi < n_z)
        grid[zi[valid], xi[valid]] = 0

    if obstacle_mask.any():
        obs_pts = points[obstacle_mask]
        xi = ((obs_pts[:, 0] - GRID_X_MIN) / GRID_RESOLUTION).astype(int)
        zi = ((obs_pts[:, 2] - GRID_Z_MIN) / GRID_RESOLUTION).astype(int)
        valid = (xi >= 0) & (xi < n_x) & (zi >= 0) & (zi < n_z)
        grid[zi[valid], xi[valid]] = 1

    # Dilate obstacles for safety margin
    kernel = np.ones((3, 3), np.uint8)
    obstacle_dilated = cv2.dilate((grid == 1).astype(np.uint8), kernel, iterations=2)
    grid[obstacle_dilated == 1] = 1

    return grid


# ── A* pathfinding ────────────────────────────────────────────────────────────

def find_safe_path(grid: np.ndarray):
    """
    A* from camera position (x=0, nearest z) to goal (x=0, z=GOAL_DISTANCE).
    Returns list of (x, z) world coordinates, or None.
    """
    import heapq

    n_z, n_x = grid.shape

    start_xi = int((0 - GRID_X_MIN) / GRID_RESOLUTION)
    start_zi = 0

    goal_xi = int((0 - GRID_X_MIN) / GRID_RESOLUTION)
    goal_zi = int((GOAL_DISTANCE - GRID_Z_MIN) / GRID_RESOLUTION)

    if start_xi < 0 or start_xi >= n_x or goal_zi < 0 or goal_zi >= n_z:
        return None

    # Snap start to nearest free cell if blocked
    if grid[start_zi, start_xi] == 1:
        free_cells = np.argwhere(grid == 0)
        if len(free_cells) == 0:
            return None
        dists = np.abs(free_cells[:, 1] - start_xi) + np.abs(free_cells[:, 0] - start_zi)
        nearest = free_cells[np.argmin(dists)]
        start_zi, start_xi = int(nearest[0]), int(nearest[1])

    # Snap goal to nearest free cell if blocked/unknown
    if grid[goal_zi, goal_xi] != 0:
        free_cells = np.argwhere(grid == 0)
        if len(free_cells) == 0:
            return None
        dists = np.abs(free_cells[:, 1] - goal_xi) + np.abs(free_cells[:, 0] - goal_zi)
        nearest = free_cells[np.argmin(dists)]
        goal_zi, goal_xi = int(nearest[0]), int(nearest[1])

    def heuristic(zi, xi):
        # Octile distance — admissible for 8-connected grid
        dx = abs(xi - goal_xi)
        dz = abs(zi - goal_zi)
        return max(dx, dz) + (np.sqrt(2) - 1) * min(dx, dz)

    open_heap = [(heuristic(start_zi, start_xi), 0.0, start_zi, start_xi)]
    came_from = {}
    g_score   = {(start_zi, start_xi): 0.0}
    closed    = set()

    neighbours = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    while open_heap:
        _, g, zi, xi = heapq.heappop(open_heap)

        if (zi, xi) in closed:
            continue
        closed.add((zi, xi))

        if (zi, xi) == (goal_zi, goal_xi):
            path = []
            cur  = (goal_zi, goal_xi)
            while cur in came_from:
                z_world = GRID_Z_MIN + cur[0] * GRID_RESOLUTION
                x_world = GRID_X_MIN + cur[1] * GRID_RESOLUTION
                path.append((x_world, z_world))
                cur = came_from[cur]
            z_world = GRID_Z_MIN + start_zi * GRID_RESOLUTION
            x_world = GRID_X_MIN + start_xi * GRID_RESOLUTION
            path.append((x_world, z_world))
            return path[::-1]

        for dzi, dxi in neighbours:
            nzi, nxi = zi + dzi, xi + dxi
            if not (0 <= nzi < n_z and 0 <= nxi < n_x):
                continue
            if grid[nzi, nxi] != 0:
                continue
            if (nzi, nxi) in closed:
                continue

            move_cost   = np.sqrt(2) if (dzi != 0 and dxi != 0) else 1.0
            tentative_g = g + move_cost

            if tentative_g < g_score.get((nzi, nxi), float('inf')):
                g_score[(nzi, nxi)] = tentative_g
                came_from[(nzi, nxi)] = (zi, xi)
                f = tentative_g + heuristic(nzi, nxi)
                heapq.heappush(open_heap, (f, tentative_g, nzi, nxi))

    return None


# ── Path smoothing ────────────────────────────────────────────────────────────

def smooth_path_spline(path_xz, floor_y_centre, num_points=120):
    """
    Smooth raw A* waypoints with a cubic B-spline then lift onto the floor
    plane (Y = floor_y_centre).  Returns an (N, 3) float32 array or None.
    """
    if path_xz is None or len(path_xz) < 2:
        return None

    if len(path_xz) < 4:
        # Too few points to fit a cubic spline — lift as-is
        return np.array([[x, floor_y_centre, z] for x, z in path_xz], dtype=np.float32)

    xs = np.array([p[0] for p in path_xz], dtype=np.float64)
    zs = np.array([p[1] for p in path_xz], dtype=np.float64)

    try:
        # s > 0 relaxes the fit → rounds sharp corners without oscillating
        tck, _ = splprep([xs, zs], s=len(path_xz) * 0.6, k=3)
        u_new  = np.linspace(0, 1, num_points)
        xs_s, zs_s = splev(u_new, tck)
    except Exception:
        xs_s, zs_s = xs, zs

    ys_s = np.full(len(xs_s), floor_y_centre, dtype=np.float32)
    return np.stack([xs_s.astype(np.float32), ys_s, zs_s.astype(np.float32)], axis=-1)


# ── Project smoothed path onto the camera image ───────────────────────────────

def project_path_to_image(path_3d: np.ndarray) -> np.ndarray | None:
    """
    Re-project 3-D path points (X, Y, Z in camera space) to pixel coordinates
    using the same pinhole model used for back-projection.
    Returns an (N, 2) int32 array of (u, v) pixel coords with Z > 0, or None.
    """
    if path_3d is None or len(path_3d) < 2:
        return None

    x, y, z = path_3d[:, 0], path_3d[:, 1], path_3d[:, 2]
    valid = z > 0.0

    u = (fx * x[valid] / z[valid] + cx).astype(np.int32)
    v = (fy * y[valid] / z[valid] + cy).astype(np.int32)

    # Keep only pixels that land inside the frame
    in_frame = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not in_frame.any():
        return None

    return np.stack([u[in_frame], v[in_frame]], axis=-1)


# ── Thread 1 — Camera capture ─────────────────────────────────────────────────
def camera_thread(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            RAW_Q.put(SENTINEL)
            return
        if RAW_Q.full():
            try:
                RAW_Q.get_nowait()
            except queue.Empty:
                pass
        RAW_Q.put(frame)


# ── Thread 2 — GPU Inference ──────────────────────────────────────────────────
def inference_thread():
    while True:
        frame = RAW_Q.get()
        if frame is SENTINEL:
            DEPTH_Q.put(SENTINEL)
            return
        depth_metric = infer_depth(frame)
        frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        DEPTH_Q.put((frame_rgb, depth_metric))


# ── Thread 3 — CPU Processing ─────────────────────────────────────────────────
def processing_thread():
    while True:
        item = DEPTH_Q.get()
        if item is SENTINEL:
            LOG_Q.put(SENTINEL)
            return

        frame_rgb, depth_metric = item
        points, valid_mask      = backproject(depth_metric)
        colors                  = frame_rgb[valid_mask].astype(np.uint8)

        floor_mask       = np.zeros(len(points), dtype=bool)
        floor_y_centre   = 0.0
        obstacle_mask    = np.zeros(len(points), dtype=bool)
        floor_mask_clean = floor_mask.copy()

        if len(points) > 200:
            floor_mask, floor_y_centre = detect_floor_histogram(points)

        if floor_mask.any():
            obstacle_mask, floor_mask_clean = detect_obstacles_above_floor(
                points, floor_mask, floor_y_centre
            )

        grid    = build_occupancy_grid(points, floor_mask_clean, obstacle_mask)
        path_xz = find_safe_path(grid)
        path_3d = smooth_path_spline(path_xz, floor_y_centre)

        colors[floor_mask_clean] = FLOOR_COLOR
        colors[obstacle_mask]    = OBSTACLE_COLOR

        scene_mask = ~floor_mask_clean & ~obstacle_mask
        LOG_Q.put((frame_rgb, depth_metric, points, colors, scene_mask,
                   floor_mask_clean, obstacle_mask, path_3d))


# ── Thread 4 — Rerun Logger ───────────────────────────────────────────────────
def logger_thread():
    frame_count = 0
    while True:
        item = LOG_Q.get()
        if item is SENTINEL:
            return

        frame_rgb, depth_metric, points, colors, scene_mask, \
            floor_mask_clean, obstacle_mask, path_3d = item

        # Monotonically increasing sequence → Rerun advances the timeline
        rr.set_time("frame", sequence=frame_count)
        frame_count += 1

        # Draw smoothed path overlay on the camera image
        frame_overlay = frame_rgb.copy()
        pixels = project_path_to_image(path_3d)
        if pixels is not None and len(pixels) >= 2:
            pts = pixels.reshape(-1, 1, 2)
            cv2.polylines(frame_overlay, [pts], isClosed=False,
                          color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
            # Start dot (green) and goal dot (red) in RGB space
            cv2.circle(frame_overlay, tuple(pixels[0]),  6, (0, 200, 0),  -1, cv2.LINE_AA)
            cv2.circle(frame_overlay, tuple(pixels[-1]), 6, (255, 60,  0), -1, cv2.LINE_AA)

        # Camera image (with path overlay)
        rr.log("world/camera/rgb", rr.Image(frame_overlay))

        # Depth image
        rr.log("world/camera/depth", rr.DepthImage(depth_metric, meter=1.0))

        # Point clouds
        rr.log(
            "world/point_cloud/scene",
            rr.Points3D(positions=points[scene_mask], colors=colors[scene_mask], radii=0.005),
        )
        if floor_mask_clean.any():
            rr.log(
                "world/point_cloud/floor",
                rr.Points3D(positions=points[floor_mask_clean], colors=colors[floor_mask_clean], radii=0.006),
            )
        if obstacle_mask.any():
            rr.log(
                "world/point_cloud/obstacles",
                rr.Points3D(positions=points[obstacle_mask], colors=colors[obstacle_mask], radii=0.007),
            )

        # Smoothed path as a single continuous line strip
        if path_3d is not None and len(path_3d) >= 2:
            rr.log(
                "world/path/safe_route",
                rr.LineStrips3D([path_3d], colors=[[0, 0, 255, 255]], radii=0.02),
            )


# ── Main ──────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

print("Live feed started.")
print("Green = clear floor | Red = obstacle | Blue = safe path (smoothed)")

threads = [
    threading.Thread(target=camera_thread,     args=(cap,), daemon=True),
    threading.Thread(target=inference_thread,              daemon=True),
    threading.Thread(target=processing_thread,             daemon=True),
    threading.Thread(target=logger_thread,                 daemon=True),
]

for t in threads:
    t.start()

try:
    for t in threads:
        t.join()
finally:
    cap.release()
    print("Session ended.")