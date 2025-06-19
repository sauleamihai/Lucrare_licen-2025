#!/usr/bin/env python3

import math
import numpy as np
import cv2
import pyrealsense2 as rs

# ───────── polar‐grid parameters ─────────
RADIAL_EDGES = np.array([0.0, 0.5, 1.0, 4.5])  # meters
GRID_R, GRID_A = len(RADIAL_EDGES) - 1, 16
EMA_ALPHA   = 0.07    # ≈3s at 30fps
MIN_VOTES   = 500
GROUND_EPS, MAX_H = 0.02, 1.9
RANSAC_TOL, RANSAC_IT, PLANE_A = 0.10, 60, 0.8
DEPTH_MIN, DEPTH_MAX = 0.15, 4.5
COLORS = [(0, 0, 255), (0, 255, 255), (0, 255, 0)]

# ───────── RealSense setup ─────────
pipe = rs.pipeline()
cfg  = rs.config()
cfg.enable_stream(rs.stream.depth,     640, 480, rs.format.z16,  30)
cfg.enable_stream(rs.stream.color,     640, 480, rs.format.bgr8,  30)
cfg.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8,   30)
cfg.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8,   30)
profile = pipe.start(cfg)

d_scale = profile.get_device().first_depth_sensor().get_depth_scale()
intr    = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy

# Compute FOV & angular edges
FOV       = 2 * math.atan((intr.width/2) / fx)
ANG_EDGES = np.linspace(-FOV/2, FOV/2, GRID_A+1)

# Align depth → color
align = rs.align(rs.stream.color)

# Filters
dec_filter  = rs.decimation_filter(2)
thr_filter  = rs.threshold_filter(DEPTH_MIN, DEPTH_MAX)
d2d         = rs.disparity_transform(True)
spat_filter = rs.spatial_filter()
temp_filter = rs.temporal_filter()
spat_filter.set_option(rs.option.filter_magnitude,    5)
spat_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
spat_filter.set_option(rs.option.filter_smooth_delta, 20)
temp_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
temp_filter.set_option(rs.option.filter_smooth_delta, 20)
fill_holes  = rs.hole_filling_filter(2)

# ───────── helper functions ─────────
def depth_to_points(depth_map):
    ys, xs = np.nonzero(depth_map)
    zs = depth_map[ys, xs]
    X  = (xs - ppx) * zs / fx
    Y  = (ys - ppy) * zs / fy
    return np.vstack((X, Y, zs)).T

def plane_ransac(pts, prev_plane=None):
    best, best_count = None, 0
    for _ in range(RANSAC_IT):
        s = pts[np.random.choice(len(pts), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-6:
            continue
        A, B, C = n; D = -n.dot(s[0])
        dists = np.abs((pts @ n) + D) / np.linalg.norm(n)
        cnt = (dists < RANSAC_TOL).sum()
        if cnt > best_count:
            best_count, best = cnt, np.array([A, B, C, D], float)
    if best is None:
        return prev_plane
    return best if prev_plane is None else PLANE_A * best + (1 - PLANE_A) * prev_plane

def compute_votes(pts, plane):
    A, B, C, D = plane
    h = ((pts @ np.array([A, B, C])) + D) / math.sqrt(A*A + B*B + C*C)
    live = pts[(h > GROUND_EPS) & (h < MAX_H)]
    if live.size == 0:
        return np.zeros((GRID_R, GRID_A), float)
    r   = np.hypot(live[:,0], live[:,2])
    phi = np.clip(np.arctan2(live[:,0], live[:,2]), ANG_EDGES[0], ANG_EDGES[-1]-1e-6)
    H, _, _ = np.histogram2d(r, phi, bins=[RADIAL_EDGES, ANG_EDGES])
    return H.astype(float)

def duplicate_angular_bins(H):
    first8 = H[:, :GRID_A//2]
    return np.repeat(first8, 2, axis=1)

def make_heatmap(C):
    Ci = C.astype(int)
    mv = Ci.max() if Ci.max() > 0 else 1
    norm = (Ci / mv * 255).astype(np.uint8)
    img = cv2.resize(norm, (GRID_A*60, GRID_R*60), cv2.INTER_NEAREST)
    hm  = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    for i in range(GRID_R):
        for j in range(GRID_A):
            col = (255,255,255) if Ci[i,j] > 0.5*mv else (0,0,0)
            cv2.putText(hm, str(Ci[i,j]), (j*60+3, i*60+48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1)
    return hm

def draw_fan_overlay(img, C):
    h, w = img.shape[:2]
    cx, cy = w//2, h
    Rpix   = int(0.9 * min(h, w) / 2)
    ov     = img.copy()
    # radial circles
    for r in RADIAL_EDGES[1:]:
        rr = int(r / RADIAL_EDGES[-1] * Rpix)
        cv2.ellipse(ov, (cx, cy), (rr, rr), 0,
                    math.degrees(ANG_EDGES[0]),
                    math.degrees(ANG_EDGES[-1]), (200,200,200), 2)
    # angular lines
    for a in ANG_EDGES:
        x2 = int(cx + Rpix * math.sin(a))
        y2 = int(cy - Rpix * math.cos(a))
        cv2.line(ov, (cx, cy), (x2, y2), (200,200,200), 2)
    # fill sectors
    for (i, j), v in np.ndenumerate(C):
        if v < MIN_VOTES:
            continue
        r0, r1 = RADIAL_EDGES[i]/RADIAL_EDGES[-1]*Rpix, RADIAL_EDGES[i+1]/RADIAL_EDGES[-1]*Rpix
        a0, a1 = ANG_EDGES[j], ANG_EDGES[j+1]
        th = np.linspace(a0, a1, 32)
        pts = np.vstack([
            (cx + r0*np.sin(th), cy - r0*np.cos(th)),
            (cx + r1*np.sin(th[::-1]), cy - r1*np.cos(th[::-1]))
        ]).T.reshape(-1,2).astype(int)
        cv2.fillPoly(ov, [pts], COLORS[i])
    cv2.addWeighted(ov, 0.6, img, 0.4, 0, img)

# ───────── main loop ─────────
plane = None
ema   = np.zeros((GRID_R, GRID_A), float)

cv2.namedWindow('Polar Grid',        cv2.WINDOW_NORMAL)
cv2.namedWindow('Stereo IR',         cv2.WINDOW_NORMAL)
cv2.namedWindow('Intensity Heatmap', cv2.WINDOW_NORMAL)
print("Press 'q' to quit.")

try:
    while True:
        frames  = pipe.wait_for_frames()
        aligned = align.process(frames)

        d   = aligned.get_depth_frame()
        c   = aligned.get_color_frame()
        ir1 = frames.get_infrared_frame(1)
        ir2 = frames.get_infrared_frame(2)
        if not d or not c:
            continue

        # apply filters
        df = dec_filter.process(d)
        df = thr_filter.process(df)
        df = d2d.process(df)
        df = spat_filter.process(df)
        df = temp_filter.process(df)
        df = rs.disparity_transform(False).process(df)
        df = fill_holes.process(df)

        # point cloud & ground plane
        pts = depth_to_points(np.asarray(df.get_data(), float) * d_scale)
        gr  = pts[pts[:,1] < np.percentile(pts[:,1], 25)]
        if len(gr) > 50:
            plane = plane_ransac(gr, plane)
        if plane is None:
            continue

        # votes + EMA
        cnt = compute_votes(pts, plane)
        ema = (1 - EMA_ALPHA) * ema + EMA_ALPHA * cnt

        # draw overlay
        rgb = np.ascontiguousarray(np.asarray(c.get_data()))
        draw_fan_overlay(rgb, ema.astype(int))
        cv2.imshow('Polar Grid', rgb)

        # stereo IR
        stereo = np.hstack([
            cv2.cvtColor(np.asarray(ir1.get_data()), cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(np.asarray(ir2.get_data()), cv2.COLOR_GRAY2BGR)
        ])
        cv2.imshow('Stereo IR', stereo)

        # intensity heatmap
        Hdup = duplicate_angular_bins(ema.astype(int))
        cv2.imshow('Intensity Heatmap', make_heatmap(Hdup))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipe.stop()
    cv2.destroyAllWindows()
