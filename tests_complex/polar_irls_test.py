#!/usr/bin/env python3
import math
import argparse
import sys
import numpy as np
import pyrealsense2 as rs

# ───────── Configurable Params ─────────
NUM_FRAMES = 500   # Default; overridden via CLI
GRID_R, GRID_A = 3, 16
EMA_ALPHA = 0.07
MIN_GROUND_POINTS = 50
IRLS_IT = 10       # Number of IRLS iterations
TUKEY_C = 0.1      # Tukey biweight parameter (meters)
GROUND_EPS, MAX_H = 0.02, 1.9
RADIAL_EDGES = np.array([0.0, 0.5, 1.0, 4.5])

# ───────── Command‐Line Parsing ─────────
parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=NUM_FRAMES,
                    help="Number of depth frames to process then exit.")
args = parser.parse_args()
NUM_FRAMES = args.frames

# ───────── Helper Functions ─────────
def depth_to_points(depth_map, fx, fy, ppx, ppy, d_scale):
    """
    Convert (H×W) depth_map (raw units) → N×3 (X, Y, Z) in meters.
    """
    ys, xs = np.nonzero(depth_map)
    zs = depth_map[ys, xs].astype(float) * d_scale
    X = (xs - ppx) * zs / fx
    Y = (ys - ppy) * zs / fy
    return np.vstack((X, Y, zs)).T

def irls_plane_fit(pts):
    """
    Fit a plane to pts (N×3) using IRLS with Tukey's biweight. Returns [A, B, C, D].
    Algorithm:
      1) Initialize all weights w_i = 1.
      2) Repeat IRLS_IT times:
         a) Compute weighted centroid μ = (∑ w_i p_i) / ∑ w_i.
         b) Compute weighted covariance C = ∑ w_i (p_i - μ)(p_i - μ)^T.
         c) Use SVD to find unit normal n = eigenvector of C with smallest eigenvalue.
         d) Compute D = -n·μ.
         e) Compute residuals r_i = n·p_i + D (signed distances).
         f) Update weights w_i via Tukey’s biweight: 
            If |r_i| < TUKEY_C, w_i = (1 - (r_i/TUKEY_C)^2)^2 else w_i = 0.
      3) Return final [n_x, n_y, n_z, D].
    """
    N = pts.shape[0]
    if N < 3:
        return None
    # 1) Initialize weights
    w = np.ones(N, dtype=float)
    plane = None

    for _ in range(IRLS_IT):
        w_sum = np.sum(w)
        if w_sum < 1e-6:
            break
        # a) Weighted centroid
        mu = (w.reshape(-1, 1) * pts).sum(axis=0) / w_sum
        # b) Weighted covariance
        diffs = pts - mu
        C = (w.reshape(-1, 1) * diffs).T @ diffs
        # c) SVD to find unit normal (smallest eigenvalue)
        _, _, Vt = np.linalg.svd(C)
        n = Vt[-1, :]
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-6:
            break
        n = n / norm_n
        # d) D = -n·μ
        D = -np.dot(n, mu)
        # e) Residuals
        r = (pts @ n) + D
        # f) Tukey biweight weights
        abs_r = np.abs(r)
        mask = abs_r < TUKEY_C
        w_new = np.zeros_like(w)
        # Tukey’s: w_i = [1 - (r_i/c)^2]^2  for |r_i| < c
        w_new[mask] = (1 - (r[mask] / TUKEY_C) ** 2) ** 2
        w = w_new

        plane = np.array([n[0], n[1], n[2], D], dtype=float)

    return plane

def compute_votes(pts, plane, ang_edges):
    """
    Given pts (N×3) and plane [A,B,C,D], build a 3×8 histogram over non-ground points:
      1) r = √(x²+z²), φ = atan2(x,z)
      2) Bin into RADIAL_EDGES×ang_edges (3×8)
    """
    if plane is None:
        return np.zeros((GRID_R, GRID_A // 2), dtype=float)
    A, B, C, D = plane
    # Signed height
    h = ((pts @ np.array([A, B, C])) + D)
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
    Expand a 3×8 histogram to 3×16 by repeating each angular bin twice.
    """
    first8 = H8[:, : (GRID_A // 2)]
    return np.repeat(first8, 2, axis=1).astype(int)

# ───────── Main Script ─────────
import cv2

# 1) Start depth pipeline
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipe.start(cfg)

sensor = profile.get_device().first_depth_sensor()
d_scale = sensor.get_depth_scale()

intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy
H_img, W_img = intr.height, intr.width

# Precompute 8 angular edges
FOV = 2 * math.atan((W_img / 2) / fx)
ang_edges = np.linspace(-FOV / 2, FOV / 2, (GRID_A // 2) + 1)

frames_processed = 0
while frames_processed < NUM_FRAMES:
    try:
        frames = pipe.wait_for_frames(timeout_ms=5000)
    except RuntimeError:
        continue

    depth_frame = frames.get_depth_frame()
    if not depth_frame:
        continue
    depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)

    # 2) Convert to point cloud
    pts3D = depth_to_points(depth_image, fx, fy, ppx, ppy, d_scale)

    # 3) Select bottom 25% as candidate ground
    if pts3D.shape[0] == 0:
        H16 = np.zeros((GRID_R, GRID_A), dtype=int)
    else:
        Ys = pts3D[:, 1]
        threshold = np.percentile(Ys, 25)
        ground_pts = pts3D[Ys < threshold]
        if ground_pts.shape[0] < MIN_GROUND_POINTS:
            H16 = np.zeros((GRID_R, GRID_A), dtype=int)
        else:
            # 4) IRLS plane fitting on ground_pts
            plane = irls_plane_fit(ground_pts)
            if plane is None:
                H16 = np.zeros((GRID_R, GRID_A), dtype=int)
            else:
                H8 = compute_votes(pts3D, plane, ang_edges)
                H16 = duplicate_bins(H8)

    # 5) Print 3×16 matrix
    for row in H16:
        print(",".join(str(int(v)) for v in row))
    print("---")

    frames_processed += 1

# 6) Cleanup
pipe.stop()
sys.exit(0)