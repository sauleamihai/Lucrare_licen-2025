#!/usr/bin/env python3
import math
import time
import sys
import numpy as np
import pyrealsense2 as rs

# ───────── Configurable Params ─────────
NUM_FRAMES = 100   # default; overridden by CLI
BASELINE_M = 0.05  # stereo baseline (m)
GRID_R, GRID_A = 3, 16
EMA_ALPHA = 0.5
D_MAX = 256
HOUGH_THRESH = 20
DEPTH_MIN, DEPTH_MAX = 0.15, 4.5

# ───────── Command‐Line Parsing ─────────
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=NUM_FRAMES,
                    help="Number of depth frames to process then exit.")
args = parser.parse_args()
NUM_FRAMES = args.frames

# ───────── Helpers ─────────
def build_v_disparity(depth_image, d_scale, fx, baseline):
    H, W = depth_image.shape
    v_disp = np.zeros((H, D_MAX), dtype=np.uint32)

    # Depth→Z (m)
    Z_m = depth_image.astype(float) * d_scale
    with np.errstate(divide="ignore", invalid="ignore"):
        disp_f = (fx * baseline) / Z_m
        disp_f[~np.isfinite(disp_f)] = 0

    disp_q = np.floor(disp_f).astype(np.int32)
    disp_q = np.clip(disp_q, 0, D_MAX - 1)

    for v in range(H):
        row = disp_q[v, :]
        mask = (row > 0)
        if np.any(mask):
            hist = np.bincount(row[mask], minlength=D_MAX)
            v_disp[v, :] = hist

    maxv = v_disp.max(initial=1)
    return ((v_disp.astype(float) / maxv) * 255.0).astype(np.uint8), disp_q

def detect_ground_line(v_disp_norm):
    v_blur = cv2.GaussianBlur(v_disp_norm, (5, 5), 0)
    v_eq   = cv2.equalizeHist(v_blur)
    edges  = cv2.Canny(v_eq, 20, 60)
    min_t, max_t = -np.pi/12, np.pi/12
    lines = cv2.HoughLines(edges, 1, np.pi/180.0, HOUGH_THRESH,
                           srn=0, stn=0, min_theta=min_t,
                           max_theta=max_t)
    if lines is None:
        return None
    rho, theta = lines[0][0]
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    if abs(sin_t) < 1e-6:
        return None
    m = -cos_t / sin_t
    c = rho / sin_t
    return m, c

def mask_ground_pixels(disp_q, m, c, threshold=1):
    H, W = disp_q.shape
    line_vals = (m * np.arange(H) + c).astype(np.int32)
    line_vals = np.clip(line_vals, 0, D_MAX - 1)
    mask = np.zeros((H, W), dtype=bool)
    for v in range(H):
        dv = disp_q[v, :]
        diff = np.abs(dv - line_vals[v])
        mask[v, :] = (diff <= threshold) & (dv > 0)
    return mask

def depth_to_points_non_ground(depth_image, mask, fx, fy, ppx, ppy, d_scale):
    H, W = depth_image.shape
    pts = []
    for v in range(H):
        for u in range(W):
            if not mask[v, u]:
                z_raw = depth_image[v, u]
                if z_raw == 0:
                    continue
                Z = float(z_raw) * d_scale
                X = (u - ppx) * Z / fx
                Y = (v - ppy) * Z / fy
                pts.append([X, Y, Z])
    return np.array(pts, dtype=float) if len(pts) else np.empty((0,3), float)

def compute_votes(pts, ang_edges):
    if pts.shape[0] == 0:
        return np.zeros((GRID_R, GRID_A//2), dtype=float)
    Y = pts[:,1]
    mask = (Y > 0.02) & (Y < 1.9)
    live = pts[mask]
    if live.shape[0] == 0:
        return np.zeros((GRID_R, GRID_A//2), dtype=float)
    X = live[:,0]; Z = live[:,2]
    r = np.hypot(X, Z)
    phi = np.clip(np.arctan2(X, Z), ang_edges[0], ang_edges[-1] - 1e-6)
    rad_edges = np.array([0.0, 0.5, 1.0, 4.5])
    H8, _, _ = np.histogram2d(r, phi, bins=[rad_edges, ang_edges])
    return H8.astype(float)

def duplicate_angular_bins(H8):
    first8 = H8[:, : (GRID_A // 2)]
    return np.repeat(first8, 2, axis=1).astype(int)

# ───────── Main Loop ─────────
import cv2
pipe = rs.pipeline()
cfg  = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipe.start(cfg)

sensor   = profile.get_device().first_depth_sensor()
d_scale  = sensor.get_depth_scale()
intr     = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy
H, W    = intr.height, intr.width
FOV     = 2 * math.atan((W / 2) / fx)
ang_edges = np.linspace(-FOV/2, FOV/2, (GRID_A // 2) + 1)

frames_processed = 0
while frames_processed < NUM_FRAMES:
    frames = pipe.wait_for_frames(timeout_ms=5000)
    depth = frames.get_depth_frame()
    if not depth:
        continue
    depth_image = np.asanyarray(depth.get_data(), dtype=np.uint16)

    # 1) v-Disparity + disparity quantization
    v_disp_norm, disp_q = build_v_disparity(depth_image, d_scale, fx, BASELINE_M)

    # 2) Detect ground line
    line = detect_ground_line(v_disp_norm)
    if line is None:
        # Print zero matrix if no ground found
        H16 = np.zeros((GRID_R, GRID_A), dtype=int)
    else:
        m, c = line
        # 3) Mask out ground
        ground_mask = mask_ground_pixels(disp_q, m, c, threshold=1)
        # 4) Non-ground → 3D points
        pts3D = depth_to_points_non_ground(depth_image, ground_mask,
                                           fx, fy, ppx, ppy, d_scale)
        # 5) 3×8 histogram + duplicate → 3×16
        raw_H8 = compute_votes(pts3D, ang_edges)
        H16    = duplicate_angular_bins(raw_H8)

    # Print out 3×16 matrix (each row comma‐separated)
    for row in H16:
        print(",".join(str(int(v)) for v in row))
    print("---")

    frames_processed += 1

pipe.stop()