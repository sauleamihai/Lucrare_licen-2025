#!/usr/bin/env python3
import math
import argparse
import sys
import numpy as np
import pyrealsense2 as rs

# ───────── Configurable Params ─────────
NUM_FRAMES = 500   # Default number of frames to process
GRID_R, GRID_A = 3, 16
EMA_ALPHA = 0.07   # EMA smoothing for histogram (not used here, but kept for consistency)
MIN_GROUND_POINTS = 50
RANSAC_IT = 60     # Number of RANSAC iterations
RANSAC_TOL = 0.10  # Distance threshold for inliers (meters)
PLANE_BLEND = 0.8  # EMA blend factor for successive plane estimates
GROUND_EPS, MAX_H = 0.02, 1.9  # Height filtering (meters)
RADIAL_EDGES = np.array([0.0, 0.5, 1.0, 4.5])  # Radial edges in meters

# ───────── Command‐Line Parsing ─────────
parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=NUM_FRAMES,
                    help="Number of depth frames to process then exit.")
args = parser.parse_args()
NUM_FRAMES = args.frames  # Override default if provided

# ───────── Helper Functions ─────────
def depth_to_points(depth_map, fx, fy, ppx, ppy, d_scale):
    """
    Convert a (H×W) depth_map of raw units → an N×3 array of (X, Y, Z) points in meters.
    X = (u - ppx)*Z/fx, Y = (v - ppy)*Z/fy.
    """
    ys, xs = np.nonzero(depth_map)
    zs = depth_map[ys, xs].astype(float) * d_scale  # Z in meters
    X = (xs - ppx) * zs / fx
    Y = (ys - ppy) * zs / fy
    return np.vstack((X, Y, zs)).T

def plane_ransac(pts, prev_plane=None):
    """
    RANSAC‐fit a plane Ax+By+Cz+D=0 to pts (N×3). Returns [A, B, C, D].
    If prev_plane is provided, does EMA blending: new_plane ← PLANE_BLEND*new + (1-PLANE_BLEND)*prev_plane.
    """
    best_plane, best_inliers = None, 0
    N = pts.shape[0]
    for _ in range(RANSAC_IT):
        if N < 3:
            break
        # Randomly sample 3 points
        idx = np.random.choice(N, 3, replace=False)
        sample = pts[idx]
        # Compute candidate normal via cross product
        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-6:
            continue  # Degenerate sample
        A, B, C = n / norm_n  # Normalize normal
        D = -np.dot(n / norm_n, sample[0])
        # Compute distances of all pts to this candidate plane
        dists = np.abs((pts @ (n / norm_n)) + D)
        inliers = np.sum(dists < RANSAC_TOL)
        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = np.array([A, B, C, D], dtype=float)
    if best_plane is None:
        return prev_plane
    if prev_plane is None:
        return best_plane
    # EMA blend with previous plane
    return PLANE_BLEND * best_plane + (1 - PLANE_BLEND) * prev_plane

def compute_votes(pts, plane, ang_edges):
    """
    Given N×3 array of pts and plane [A,B,C,D], compute a GRID_R×(GRID_A/2) histogram:
    1) Compute signed height h_i = (n·p_i + D).
    2) Keep pts with GROUND_EPS < h < MAX_H.
    3) For each remaining pt: r = sqrt(x^2 + z^2), φ = atan2(x, z).
    4) Bin (r, φ) into a 2D histogram with radial bins RADIAL_EDGES and angular bins ang_edges.
    """
    A, B, C, D = plane
    # Signed distance to plane
    h = ((pts @ np.array([A, B, C])) + D)  # Already normalized since n is unit
    live = pts[(h > GROUND_EPS) & (h < MAX_H)]
    if live.shape[0] == 0:
        return np.zeros((GRID_R, GRID_A // 2), dtype=float)
    X = live[:, 0]; Z = live[:, 2]
    r = np.hypot(X, Z)
    phi = np.clip(np.arctan2(X, Z), ang_edges[0], ang_edges[-1] - 1e-6)
    H8, _, _ = np.histogram2d(r, phi, bins=[RADIAL_EDGES, ang_edges])
    return H8.astype(float)

def duplicate_bins(H8):
    """
    Convert a GRID_R×(GRID_A/2) histogram to GRID_R×GRID_A by repeating each angular bin twice.
    """
    first8 = H8[:, : (GRID_A // 2)]
    return np.repeat(first8, 2, axis=1).astype(int)

# ───────── Main Script ─────────
import cv2

# 1) Start RealSense depth‐only pipeline
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipe.start(cfg)

sensor = profile.get_device().first_depth_sensor()
d_scale = sensor.get_depth_scale()  # Depth scale (meters per unit)

intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy
H_img, W_img = intr.height, intr.width

# Precompute angular edges for GRID_A/2 = 8 bins
FOV = 2 * math.atan((W_img / 2) / fx)
ang_edges = np.linspace(-FOV / 2, FOV / 2, (GRID_A // 2) + 1)

# Initialize variables
plane = None
frames_processed = 0

# 2) Process frames
while frames_processed < NUM_FRAMES:
    try:
        frames = pipe.wait_for_frames(timeout_ms=5000)
    except RuntimeError:
        # Timeout waiting for frame; retry
        continue

    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        continue
    depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)

    # 3) Convert to point cloud (in meters)
    pts3D = depth_to_points(depth_image, fx, fy, ppx, ppy, d_scale)

    # 4) Select candidate ground points (lowest 25% by Y)
    if pts3D.shape[0] == 0:
        # No valid depth; output zero matrix
        H16 = np.zeros((GRID_R, GRID_A), dtype=int)
    else:
        Ys = pts3D[:, 1]
        threshold = np.percentile(Ys, 25)
        ground_pts = pts3D[Ys < threshold]
        if ground_pts.shape[0] < MIN_GROUND_POINTS:
            # Too few points to fit plane
            H16 = np.zeros((GRID_R, GRID_A), dtype=int)
        else:
            # 5) Fit or update ground plane via RANSAC
            plane = plane_ransac(ground_pts, plane)
            if plane is None:
                H16 = np.zeros((GRID_R, GRID_A), dtype=int)
            else:
                # 6) Compute 3×8 histogram + duplicate → 3×16
                H8 = compute_votes(pts3D, plane, ang_edges)
                H16 = duplicate_bins(H8)

    # 7) Print 3×16 matrix
    for row in H16:
        print(",".join(str(int(v)) for v in row))
    print("---")

    frames_processed += 1

# 8) Cleanup
pipe.stop()
sys.exit(0)