import cv2
import openvino as ov
import numpy as np
import rerun as rr
import threading
import queue

# ── Setup Rerun ───────────────────────────────────────────────────────────────
rr.init("intel_igpu_3d_mapping", spawn=True)
rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

# ── Camera & iGPU ─────────────────────────────────────────────────────────────
width, height = 640, 480
focal_length  = width * 0.8

core           = ov.Core()
compiled_model = core.compile_model("Test Env\depth_anything_v2_vitb.xml", "GPU")

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

fx = fy = focal_length
cx, cy  = width / 2.0, height / 2.0

# ── Pipeline queues ───────────────────────────────────────────────────────────
# Each queue holds at most 2 items so a slow stage doesn't accumulate stale frames.
RAW_Q    = queue.Queue(maxsize=2)   # camera  → inference
DEPTH_Q  = queue.Queue(maxsize=2)   # inference → CPU processing
LOG_Q    = queue.Queue(maxsize=2)   # CPU processing → Rerun logger
SENTINEL = None                     # poison-pill to shut down threads


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


# Pre-compute the ray-scale grid once — it never changes
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
    z_corrected = depth_metric / _RAY_SCALE          # element-wise; _RAY_SCALE already cached
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
    """Return an (N,) int64 array encoding (xi, zi) pairs as a single hash key.
    Using xi * LARGE_PRIME + zi avoids building a set of tuples in Python."""
    xi = np.floor(pts[:, 0] / XZ_BIN_SIZE).astype(np.int32)
    zi = np.floor(pts[:, 2] / XZ_BIN_SIZE).astype(np.int32)
    # Pack two int32s into one int64 — collision-free for any reasonable scene
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
    floor_pts = points[floor_mask]

    # ── Vectorized XZ cell counting (no Python loops) ─────────────────────────
    cand_keys = _xz_keys(cand_pts)

    # Count occurrences per key
    unique_keys, counts = np.unique(cand_keys, return_counts=True)
    obstacle_keys = unique_keys[counts >= MIN_OBSTACLE_POINTS]   # cells that qualify

    if len(obstacle_keys) == 0:
        return np.zeros(len(points), dtype=bool), floor_mask.copy()

    # Convert to a set for O(1) lookup — but now it's a set of ints, not tuples
    obs_key_set = set(obstacle_keys.tolist())

    # ── Mark obstacle points (vectorized isin) ────────────────────────────────
    obstacle_mask = np.zeros(len(points), dtype=bool)
    cand_in_obs   = np.isin(cand_keys, obstacle_keys)            # pure numpy
    cand_indices  = np.where(candidate_mask)[0]
    obstacle_mask[cand_indices[cand_in_obs]] = True

    # ── Un-green floor points below obstacle cells (vectorized isin) ──────────
    floor_mask_clean = floor_mask.copy()
    if len(floor_pts) > 0:
        floor_keys    = _xz_keys(floor_pts)
        floor_blocked = np.isin(floor_keys, obstacle_keys)       # pure numpy
        floor_indices = np.where(floor_mask)[0]
        floor_mask_clean[floor_indices[floor_blocked]] = False

    return obstacle_mask, floor_mask_clean


# ── Thread 1 — Camera capture ─────────────────────────────────────────────────
def camera_thread(cap):
    """Grabs frames as fast as possible and pushes them to RAW_Q."""
    while True:
        ret, frame = cap.read()
        if not ret:
            RAW_Q.put(SENTINEL)
            return
        # If the inference thread is behind, drop the oldest frame rather than stall
        if RAW_Q.full():
            try:
                RAW_Q.get_nowait()
            except queue.Empty:
                pass
        RAW_Q.put(frame)


# ── Thread 2 — GPU Inference ──────────────────────────────────────────────────
def inference_thread():
    """Pulls raw frames, runs depth model on GPU, pushes (frame_rgb, depth) pairs."""
    while True:
        frame = RAW_Q.get()
        if frame is SENTINEL:
            DEPTH_Q.put(SENTINEL)
            return
        depth_metric = infer_depth(frame)
        frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        DEPTH_Q.put((frame_rgb, depth_metric))


# ── Thread 3 — CPU Processing (backproject + floor + obstacle) ─────────────
def processing_thread():
    """Back-projects depth, runs floor and obstacle detection, pushes log payload."""
    while True:
        item = DEPTH_Q.get()
        if item is SENTINEL:
            LOG_Q.put(SENTINEL)
            return

        frame_rgb, depth_metric = item
        points, valid_mask      = backproject(depth_metric)
        colors                  = frame_rgb[valid_mask].astype(np.uint8)

        floor_mask      = np.zeros(len(points), dtype=bool)
        floor_y_centre  = 0.0
        obstacle_mask   = np.zeros(len(points), dtype=bool)
        floor_mask_clean = floor_mask.copy()

        if len(points) > 200:
            floor_mask, floor_y_centre = detect_floor_histogram(points)

        if floor_mask.any():
            obstacle_mask, floor_mask_clean = detect_obstacles_above_floor(
                points, floor_mask, floor_y_centre
            )

        colors[floor_mask_clean] = FLOOR_COLOR
        colors[obstacle_mask]    = OBSTACLE_COLOR

        scene_mask = ~floor_mask_clean & ~obstacle_mask
        LOG_Q.put((frame_rgb, points, colors, scene_mask, floor_mask_clean, obstacle_mask))


# ── Thread 4 — Rerun Logger ───────────────────────────────────────────────────
def logger_thread():
    """Consumes processed payloads and sends them to Rerun — never blocks CPU."""
    while True:
        item = LOG_Q.get()
        if item is SENTINEL:
            return

        frame_rgb, points, colors, scene_mask, floor_mask_clean, obstacle_mask = item

        rr.set_time("time", duration=0.0)
        rr.log("world/camera/image", rr.Image(frame_rgb))

        rr.log(
            "world/point_cloud/scene",
            rr.Points3D(positions=points[scene_mask],  colors=colors[scene_mask],  radii=0.005),
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


# ── Main ──────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

print("Live feed started. Green = clear floor | Red = obstacle | Floor under obstacle = unmarked.")

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