#!/usr/bin/env python3

import math
import time
import sys
import socket
import argparse
import numpy as np
import pyrealsense2 as rs

# ───────── Default Parameters ─────────
# Network parameters
DEFAULT_ESP32_IP = "10.42.0.220"
DEFAULT_ESP32_PORT = 12345
DEFAULT_ACK_TIMEOUT = 30.0
DEFAULT_SEND_INTERVAL = 1.0

# Grid parameters
DEFAULT_GRID_R = 3
DEFAULT_GRID_A = 16
DEFAULT_EMA_ALPHA = 0.5
DEFAULT_GROUND_EPS = 0.02
DEFAULT_MAX_H = 1.9
DEFAULT_DEPTH_MIN = 0.15
DEFAULT_DEPTH_MAX = 4.5

# IRLS (Huber M-Estimator) parameters
DEFAULT_IRLS_ITER = 5
DEFAULT_HUBER_DELTA = 0.05
DEFAULT_MIN_GROUND_POINTS = 3

# Logistic intensity parameters
DEFAULT_LOGI_BASELINE = 20
DEFAULT_LOGI_MAXVAL = 255
DEFAULT_LOGI_DMID = 0.9
DEFAULT_LOGI_ALPHA = 2.0

# Ramp parameters
DEFAULT_RAMP_MAX_STEP = 10

# ───────── Command‐Line Parsing ─────────
parser = argparse.ArgumentParser(description="IRLS (Huber) ESP32 TCP Client - Fixed and Configurable")

# Network parameters
parser.add_argument("--esp-ip", type=str, default=DEFAULT_ESP32_IP, 
                    help="ESP32 IP address")
parser.add_argument("--esp-port", type=int, default=DEFAULT_ESP32_PORT, 
                    help="ESP32 port")
parser.add_argument("--ack-timeout", type=float, default=DEFAULT_ACK_TIMEOUT, 
                    help="ACK timeout in seconds")
parser.add_argument("--send-interval", type=float, default=DEFAULT_SEND_INTERVAL, 
                    help="Send interval in seconds")

# Depth processing parameters
parser.add_argument("--depth-min", type=float, default=DEFAULT_DEPTH_MIN, 
                    help="Minimum depth in meters")
parser.add_argument("--depth-max", type=float, default=DEFAULT_DEPTH_MAX, 
                    help="Maximum depth in meters")
parser.add_argument("--ground-eps", type=float, default=DEFAULT_GROUND_EPS, 
                    help="Ground epsilon threshold in meters")
parser.add_argument("--max-height", type=float, default=DEFAULT_MAX_H, 
                    help="Maximum obstacle height in meters")

# Grid parameters
parser.add_argument("--grid-radial", type=int, default=DEFAULT_GRID_R, 
                    help="Number of radial bins")
parser.add_argument("--grid-angular", type=int, default=DEFAULT_GRID_A, 
                    help="Number of angular bins")

# IRLS parameters
parser.add_argument("--irls-iterations", type=int, default=DEFAULT_IRLS_ITER, 
                    help="Number of IRLS iterations")
parser.add_argument("--huber-delta", type=float, default=DEFAULT_HUBER_DELTA, 
                    help="Huber M-estimator delta threshold in meters")
parser.add_argument("--min-ground-points", type=int, default=DEFAULT_MIN_GROUND_POINTS, 
                    help="Minimum points required for ground plane fitting")

# Logistic intensity parameters
parser.add_argument("--logi-baseline", type=int, default=DEFAULT_LOGI_BASELINE, 
                    help="Logistic intensity baseline")
parser.add_argument("--logi-maxval", type=int, default=DEFAULT_LOGI_MAXVAL, 
                    help="Logistic intensity maximum value")
parser.add_argument("--logi-dmid", type=float, default=DEFAULT_LOGI_DMID, 
                    help="Logistic intensity mid-distance")
parser.add_argument("--logi-alpha", type=float, default=DEFAULT_LOGI_ALPHA, 
                    help="Logistic intensity alpha parameter")

# EMA and ramp parameters
parser.add_argument("--ema-alpha", type=float, default=DEFAULT_EMA_ALPHA, 
                    help="EMA alpha for smoothing (0.0-1.0)")
parser.add_argument("--ramp-max-step", type=int, default=DEFAULT_RAMP_MAX_STEP, 
                    help="Maximum intensity jump per interval")

# Processing options
parser.add_argument("--no-ema", action="store_true", 
                    help="Disable EMA smoothing")
parser.add_argument("--no-ramp", action="store_true", 
                    help="Disable ramp limiting")
parser.add_argument("--use-grid-intensity", action="store_true", 
                    help="Use grid-based logistic intensity instead of histogram")
parser.add_argument("--estimator", type=str, choices=["huber", "tukey"], 
                    default="huber", help="M-estimator type (huber or tukey)")
parser.add_argument("--verbose", "-v", action="store_true", 
                    help="Enable verbose output")
parser.add_argument("--debug", action="store_true", 
                    help="Enable debug output with detailed IRLS info")

args = parser.parse_args()

# Apply parsed arguments
ESP32_IP = args.esp_ip
ESP32_PORT = args.esp_port
ACK_TIMEOUT = args.ack_timeout
SEND_INTERVAL = args.send_interval

DEPTH_MIN = args.depth_min
DEPTH_MAX = args.depth_max
GROUND_EPS = args.ground_eps
MAX_H = args.max_height

GRID_R = args.grid_radial
GRID_A = args.grid_angular

IRLS_ITER = args.irls_iterations
HUBER_DELTA = args.huber_delta
MIN_GROUND_POINTS = args.min_ground_points

LOGI_BASELINE = args.logi_baseline
LOGI_MAXVAL = args.logi_maxval
LOGI_DMID = args.logi_dmid
LOGI_ALPHA = args.logi_alpha

EMA_ALPHA = args.ema_alpha
RAMP_MAX_STEP = args.ramp_max_step

USE_EMA = not args.no_ema
USE_RAMP = not args.no_ramp
USE_GRID_INTENSITY = args.use_grid_intensity
ESTIMATOR_TYPE = args.estimator
VERBOSE = args.verbose
DEBUG = args.debug

# Validation
if not (0.0 <= EMA_ALPHA <= 1.0):
    print("Warning: EMA alpha should be between 0.0 and 1.0")
    EMA_ALPHA = np.clip(EMA_ALPHA, 0.0, 1.0)

if DEPTH_MIN >= DEPTH_MAX:
    print("Error: depth-min must be less than depth-max")
    sys.exit(1)

if IRLS_ITER <= 0:
    print("Error: IRLS iterations must be positive")
    sys.exit(1)

if HUBER_DELTA <= 0:
    print("Error: Huber delta must be positive")
    sys.exit(1)

if GRID_A % 2 != 0:
    print("Warning: Grid angular should be even, adjusting...")
    GRID_A = (GRID_A // 2) * 2

if VERBOSE or DEBUG:
    print(f" IRLS CONFIGURATION:")
    print(f"   Network: {ESP32_IP}:{ESP32_PORT}")
    print(f"   Timeouts: ACK={ACK_TIMEOUT}s, Interval={SEND_INTERVAL}s")
    print(f"   Depth range: {DEPTH_MIN:.2f}m - {DEPTH_MAX:.2f}m")
    print(f"   Obstacle height: {GROUND_EPS:.3f}m - {MAX_H:.2f}m")
    print(f"   Grid size: {GRID_R}x{GRID_A}")
    print(f"   IRLS: {IRLS_ITER} iterations, {ESTIMATOR_TYPE} estimator")
    print(f"   Huber delta: {HUBER_DELTA:.3f}m")
    print(f"   Min ground points: {MIN_GROUND_POINTS}")
    print(f"   Grid intensity: {'ON' if USE_GRID_INTENSITY else 'OFF'}")
    print(f"   EMA: {'ON' if USE_EMA else 'OFF'} (α={EMA_ALPHA:.2f})")
    print(f"   Ramp limiting: {'ON' if USE_RAMP else 'OFF'} (max_step={RAMP_MAX_STEP})")
    print()

# ═══════════════════════════════════════════════════════════════════════════
# TCP COMMUNICATION
# ═══════════════════════════════════════════════════════════════════════════

def send_matrix_with_ack(sock, matrix):
    matrix_str = ";".join([",".join(map(str, row)) for row in matrix])
    retries = 3
    
    while retries > 0:
        try:
            sock.sendall(matrix_str.encode())
            
            if DEBUG:
                print(f"[TCP] Sent matrix ({len(matrix_str)} bytes)")
                print("Matrix:")
                for row in matrix:
                    print(" ".join(f"{v:3d}" for v in row))
            
            time.sleep(0.4)  # Let the ESP32 process
            response = sock.recv(1024).decode().strip()
            
            if response == "ACK":
                if DEBUG:
                    print("[TCP] ACK received!")
                return True
            else:
                print(f"[TCP] Unexpected response: {response}")
                
        except socket.timeout:
            print("[TCP] Timeout waiting for ACK, retrying…")
        except Exception as e:
            print(f"[TCP] Error sending matrix: {e}")
            
        retries -= 1
        time.sleep(1.0)
        
    print("[TCP] Failed to receive ACK after multiple retries.")
    return False

# ═══════════════════════════════════════════════════════════════════════════
# M-ESTIMATOR WEIGHT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def huber_weight(r, delta=None):
    if delta is None:
        delta = HUBER_DELTA
    abs_r = abs(r)
    return 1.0 if abs_r <= delta else delta / abs_r

def tukey_weight(r, c=None):
    if c is None:
        c = HUBER_DELTA * 4.685  # Convert Huber delta to appropriate Tukey c
    abs_r = abs(r)
    if abs_r <= c:
        ratio = r / c
        return (1 - ratio**2)**2
    else:
        return 0.0

def get_weight_function():
    if ESTIMATOR_TYPE == "huber":
        return huber_weight
    elif ESTIMATOR_TYPE == "tukey":
        return tukey_weight
    else:
        return huber_weight

# ═══════════════════════════════════════════════════════════════════════════
# IRLS PLANE FITTING - Fixed and Robust
# ═══════════════════════════════════════════════════════════════════════════

def irls_plane_fit(pts):
    N = pts.shape[0]
    if N < MIN_GROUND_POINTS:
        if DEBUG:
            print(f"[IRLS] Too few points: {N} < {MIN_GROUND_POINTS}")
        return None
    
    # Initialize weights to 1
    w = np.ones(N, dtype=float)
    plane = None
    weight_func = get_weight_function()
    
    if DEBUG:
        print(f"[IRLS] Starting with {N} points, {IRLS_ITER} iterations, {ESTIMATOR_TYPE} estimator")
    
    for iteration in range(IRLS_ITER):
        sum_w = np.sum(w)
        if sum_w <= 1e-10:  # More robust check
            if DEBUG:
                print(f"[IRLS] Iteration {iteration}: Weight sum too small, breaking")
            break
            
        # Weighted centroid
        c = (w[:, np.newaxis] * pts).sum(axis=0) / sum_w
        
        # Centered points
        diffs = pts - c
        
        # Weighted covariance matrix
        weighted_diffs = diffs * np.sqrt(w[:, np.newaxis])
        Cmat = weighted_diffs.T @ weighted_diffs
        
        # Add small regularization to prevent singular matrix
        Cmat += np.eye(3) * 1e-10
        
        # Find plane normal (eigenvector with smallest eigenvalue)
        try:
            eigvals, eigvecs = np.linalg.eigh(Cmat)
            n = eigvecs[:, np.argmin(eigvals)]
            
            # Ensure normal has unit length
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                if DEBUG:
                    print(f"[IRLS] Iteration {iteration}: Normal too small")
                break
            n = n / norm_n
            
        except np.linalg.LinAlgError as e:
            if DEBUG:
                print(f"[IRLS] Iteration {iteration}: LinAlg error: {e}")
            break
            
        # Plane equation
        D = -np.dot(n, c)
        plane = np.array([n[0], n[1], n[2], D], dtype=float)
        
        # Compute residuals and update weights
        r = pts @ n + D  # Signed distance to plane
        w_new = np.array([weight_func(ri) for ri in r], dtype=float)
        
        # Ensure weights are valid
        w_new = np.clip(w_new, 1e-10, 1.0)
        
        if DEBUG:
            avg_weight = np.mean(w_new)
            nonzero_weights = np.count_nonzero(w_new > 1e-6)
            residual_rms = np.sqrt(np.mean(r**2))
            print(f"[IRLS] Iter {iteration}: avg_weight={avg_weight:.3f}, "
                  f"active={nonzero_weights}/{N}, RMS={residual_rms:.4f}")
        
        # Check for convergence
        if iteration > 0:
            weight_change = np.mean(np.abs(w_new - w))
            if weight_change < 1e-6:
                if DEBUG:
                    print(f"[IRLS] Converged at iteration {iteration}")
                break
        
        w = w_new
    
    if DEBUG and plane is not None:
        final_residuals = pts @ plane[:3] + plane[3]
        inliers = np.sum(np.abs(final_residuals) < HUBER_DELTA)
        print(f"[IRLS] Final: {inliers}/{N} inliers within {HUBER_DELTA:.3f}m")
    
    return plane

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS - Fixed and Robust
# ═══════════════════════════════════════════════════════════════════════════

def depth_to_points(depth_map, fx, fy, ppx, ppy, d_scale):
     ys, xs = np.nonzero(depth_map)
    if len(ys) == 0:
        return np.empty((0, 3), dtype=float)
        
    zs = depth_map[ys, xs] * d_scale
    xs_m = (xs - ppx) * zs / fx
    ys_m = (ys - ppy) * zs / fy
    return np.vstack((xs_m, ys_m, zs)).T

def compute_votes(pts, plane, ANG_EDGES):
    if pts.shape[0] == 0:
        if DEBUG:
            print("[VOTES] No input points")
        return np.zeros((GRID_R, GRID_A//2), dtype=float)
        
    A, B, C, D = plane
    norm_n = math.sqrt(A*A + B*B + C*C)
    
    if norm_n < 1e-10:
        if DEBUG:
            print("[VOTES] Zero normal vector")
        return np.zeros((GRID_R, GRID_A//2), dtype=float)
    
    # Normalized signed distance to plane
    h = (pts @ np.array([A, B, C]) + D) / norm_n
    
    # Filter points above ground
    mask = (h > GROUND_EPS) & (h < MAX_H)
    live = pts[mask]
    
    if live.shape[0] == 0:
        if DEBUG:
            print("[VOTES] No points above ground plane")
        return np.zeros((GRID_R, GRID_A//2), dtype=float)
    
    # Convert to polar coordinates
    r = np.hypot(live[:, 0], live[:, 2])
    phi = np.arctan2(live[:, 0], live[:, 2])
    phi = np.clip(phi, ANG_EDGES[0], ANG_EDGES[-1] - 1e-6)
    
    # Create radial edges based on grid size
    max_range = max(4.5, np.max(r) * 1.1)  # Adaptive max range
    RADIAL_EDGES = np.linspace(0.0, max_range, GRID_R + 1)
    
    # Create histogram
    try:
        H, _, _ = np.histogram2d(r, phi, bins=[RADIAL_EDGES, ANG_EDGES])
    except Exception as e:
        if DEBUG:
            print(f"[VOTES] Histogram error: {e}")
        return np.zeros((GRID_R, GRID_A//2), dtype=float)
    
    if DEBUG:
        total_votes = np.sum(H)
        print(f"[VOTES] {live.shape[0]} points → {total_votes} votes")
    
    return H.astype(float)

def duplicate_angular_bins(H8):
    if H8.shape[1] != GRID_A // 2:
        if DEBUG:
            print(f"[DUP] Warning: Expected {GRID_A//2} angular bins, got {H8.shape[1]}")
    
    first_half = H8[:, : (GRID_A // 2)]
    return np.repeat(first_half, 2, axis=1)

def logistic_intensity(d, baseline=None, maxVal=None, d_mid=None, alpha=None):
    if baseline is None: baseline = LOGI_BASELINE
    if maxVal is None: maxVal = LOGI_MAXVAL
    if d_mid is None: d_mid = LOGI_DMID
    if alpha is None: alpha = LOGI_ALPHA
    
    if d <= 0.10:
        return maxVal
    if d >= 2.0:
        return 0
    
    try:
        result = baseline + (maxVal - baseline) / (1.0 + math.exp(alpha * (d - d_mid)))
        return int(np.clip(result, 0, maxVal))
    except:
        return baseline

def process_depth_image_grid(depth_image, depth_scale):
    max_dist_raw = int(2.0 / depth_scale)
    depth_clipped = np.where((depth_image > 0) & (depth_image < max_dist_raw), depth_image, 0)
    
    h, w = depth_clipped.shape
    cell_h = max(1, h // GRID_R)
    cell_w = max(1, w // GRID_A)
    
    intensity_matrix = np.zeros((GRID_R, GRID_A), dtype=np.uint8)
    
    for row in range(GRID_R):
        for col in range(GRID_A):
            y1 = row * cell_h
            y2 = min((row + 1) * cell_h, h)
            x1 = col * cell_w
            x2 = min((col + 1) * cell_w, w)
            
            cell = depth_clipped[y1:y2, x1:x2]
            valid_pixels = cell[cell > 0]
            
            if valid_pixels.size > 0:
                # Count very close pixels
                raw_vals_m = valid_pixels * depth_scale
                close_count = np.count_nonzero(raw_vals_m <= 0.10)
                
                if close_count >= 0.7 * valid_pixels.size:
                    intensity = LOGI_MAXVAL
                else:
                    avg_m = np.mean(valid_pixels) * depth_scale
                    intensity = logistic_intensity(avg_m)
            else:
                intensity = 0
            
            intensity_matrix[row, col] = intensity
    
    return intensity_matrix

def apply_ramp(H_curr, H_prev, max_step=None):
    if not USE_RAMP or H_prev is None:
        return H_curr.copy()
    
    if max_step is None:
        max_step = RAMP_MAX_STEP
    
    H_new = H_curr.copy()
    rows, cols = H_curr.shape
    ramp_applied = 0
    
    for i in range(rows):
        for j in range(cols):
            prev = int(H_prev[i, j])
            curr = int(H_curr[i, j])
            if curr == 255 and prev < 255 - max_step:
                H_new[i, j] = prev + max_step
                ramp_applied += 1
    
    if DEBUG and ramp_applied > 0:
        print(f"[RAMP] Applied ramp limiting to {ramp_applied} cells")
    
    return H_new

# ═══════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=== IRLS (Huber M-Estimator) ESP32 TCP Client - Fixed ===")
    print(f"Algorithm: IRLS {ESTIMATOR_TYPE.title()} M-Estimator → Polar Histogram")
    
    if VERBOSE:
        print(f"ESP32: {ESP32_IP}:{ESP32_PORT}")
    
    # Setup TCP Socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    try:
        client_socket.connect((ESP32_IP, ESP32_PORT))
        print(f" Connected to ESP32 @ {ESP32_IP}:{ESP32_PORT}")
    except Exception as e:
        print(f"[ERROR] Cannot connect to ESP32 @ {ESP32_IP}:{ESP32_PORT} → {e}")
        sys.exit(1)
    
    client_socket.settimeout(5)
    
    # Start RealSense pipeline
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipe.start(cfg)
    
    # Read intrinsics & depth scale
    sensor = profile.get_device().first_depth_sensor()
    d_scale = sensor.get_depth_scale()
    intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy
    width, height = intr.width, intr.height
    
    if VERBOSE:
        print(f"RealSense: {width}x{height}, fx={fx:.1f}, depth_scale={d_scale:.6f}")
    
    # Compute FOV & angular edges
    FOV = 2 * math.atan((width / 2) / fx)
    ANG_EDGES = np.linspace(-FOV/2, FOV/2, (GRID_A // 2) + 1)
    
    if VERBOSE:
        print(f"FOV: {math.degrees(FOV):.1f}°, Angular bins: {len(ANG_EDGES)-1}")
    
    # Initialize variables
    ema_hist = np.zeros((GRID_R, GRID_A//2), dtype=float) if USE_EMA else None
    H_prev = None
    last_send = 0.0
    last_ack = 0.0
    
    # Statistics
    frames_processed = 0
    successful_sends = 0
    plane_failures = 0
    total_irls_time = 0
    
    print("Starting IRLS processing... Press Ctrl+C to stop.\n")
    
    try:
        while True:
            now = time.time()
            if now - last_send < SEND_INTERVAL:
                time.sleep(0.01)
                continue
            
            # Grab depth frame with retry
            depth_frame = None
            while depth_frame is None:
                try:
                    frames = pipe.wait_for_frames(timeout_ms=5000)
                    depth_frame = frames.get_depth_frame()
                    if not depth_frame:
                        depth_frame = None
                        if VERBOSE:
                            print("[WARN] No valid depth frame; retrying…")
                except RuntimeError:
                    if VERBOSE:
                        print("[WARN] Timeout waiting for frame; retrying…")
                    continue
            
            # Convert depth frame to numpy
            depth_image = np.asanyarray(depth_frame.get_data(), dtype=float)
            
            # Build 3D point cloud from depth image
            pts_all = depth_to_points(depth_image, fx, fy, ppx, ppy, d_scale)
            
            # Select bottom 25% of points by Y as ground candidates
            if pts_all.shape[0] > 0:
                ys = pts_all[:, 1]
                thresh = np.percentile(ys, 25)
                ground_pts = pts_all[ys < thresh]
            else:
                ground_pts = np.empty((0, 3), dtype=float)
            
            # IRLS plane fitting
            plane = None
            if ground_pts.shape[0] >= MIN_GROUND_POINTS:
                irls_start = time.time()
                plane = irls_plane_fit(ground_pts)
                irls_time = (time.time() - irls_start) * 1000
                total_irls_time += irls_time
                
                if DEBUG and plane is not None:
                    print(f"[IRLS] Plane fit completed in {irls_time:.2f}ms")
            
            # Generate output matrix
            if plane is None:
                plane_failures += 1
                if VERBOSE:
                    print(f"[WARN] IRLS plane fit failed (total failures: {plane_failures})")
                
                if USE_GRID_INTENSITY:
                    # Use grid-based logistic intensity processing
                    H16 = process_depth_image_grid(depth_image.astype(np.uint16), d_scale)
                else:
                    # Use previous EMA or zeros
                    if USE_EMA and ema_hist is not None:
                        H8 = ema_hist
                        H16 = duplicate_angular_bins(H8.astype(int))
                    else:
                        H16 = np.zeros((GRID_R, GRID_A), dtype=int)
            else:
                if USE_GRID_INTENSITY:
                    # Use grid-based logistic intensity processing
                    H16 = process_depth_image_grid(depth_image.astype(np.uint16), d_scale)
                else:
                    # Compute raw histogram from IRLS plane
                    raw_hist = compute_votes(pts_all, plane, ANG_EDGES)
                    
                    # Apply EMA smoothing if enabled
                    if USE_EMA:
                        if ema_hist is None:
                            ema_hist = raw_hist.copy()
                        else:
                            ema_hist = (1 - EMA_ALPHA) * ema_hist + EMA_ALPHA * raw_hist
                        H8 = ema_hist
                    else:
                        H8 = raw_hist
                    
                    # Duplicate angular bins
                    H16 = duplicate_angular_bins(H8.astype(int))
            
            # Ensure correct matrix dimensions
            if H16.shape != (GRID_R, GRID_A):
                if DEBUG:
                    print(f"[WARN] Matrix shape mismatch: {H16.shape} != ({GRID_R}, {GRID_A})")
                H16 = np.zeros((GRID_R, GRID_A), dtype=int)
            
            # Clip to [0,255] and apply ramp
            H16 = np.clip(H16, 0, 255).astype(np.uint8)
            H16_ramped = apply_ramp(H16, H_prev)
            
            # Print matrix
            if VERBOSE or DEBUG:
                matrix_sum = np.sum(H16_ramped)
                matrix_max = np.max(H16_ramped)
                avg_irls_time = total_irls_time / max(1, frames_processed - plane_failures)
                print(f"# Frame {frames_processed}: sum={matrix_sum}, max={matrix_max}, "
                      f"avg_IRLS={avg_irls_time:.1f}ms")
            
            for row in H16_ramped:
                print(",".join(str(int(v)) for v in row))
            print("---")
            
            # Send over TCP
            success = send_matrix_with_ack(client_socket, H16_ramped)
            
            if success:
                last_send = now
                last_ack = now
                H_prev = H16_ramped.copy()
                successful_sends += 1
                if VERBOSE:
                    print(f"[TCP {time.strftime('%H:%M:%S')}] Matrix sent, ACK OK "
                          f"(#{successful_sends})")
            else:
                if last_ack == 0.0:
                    last_ack = now
                elapsed = now - last_ack
                print(f"[TCP] No ACK, elapsed since last ACK: {elapsed:.1f}s")
                if elapsed > ACK_TIMEOUT:
                    print(f"[ERROR] No ACK from ESP32 for {ACK_TIMEOUT:.0f}s → exiting.")
                    break
                else:
                    last_send = now
            
            frames_processed += 1
            
            # Stats every 10 successful sends
            if DEBUG and successful_sends > 0 and successful_sends % 10 == 0:
                success_rate = successful_sends / frames_processed * 100
                failure_rate = plane_failures / frames_processed * 100
                avg_irls_time = total_irls_time / max(1, frames_processed - plane_failures)
                
                print(f"\n IRLS STATS after {successful_sends} sends:")
                print(f"   Frames processed: {frames_processed}")
                print(f"   Success rate: {success_rate:.1f}%")
                print(f"   Plane failure rate: {failure_rate:.1f}%")
                print(f"   Avg IRLS time: {avg_irls_time:.1f}ms")
                print(f"   Estimator: {ESTIMATOR_TYPE}, delta={HUBER_DELTA:.3f}")
                print()
    
    except KeyboardInterrupt:
        print("\n Interrupted by user. Exiting…")
    
    finally:
        # Final statistics
        print(f"\n FINAL IRLS STATS:")
        print(f"   Frames processed: {frames_processed}")
        print(f"   Successful sends: {successful_sends}")
        print(f"   Plane failures: {plane_failures}")
        
        if frames_processed > 0:
            success_rate = successful_sends / frames_processed * 100
            failure_rate = plane_failures / frames_processed * 100
            avg_irls_time = total_irls_time / max(1, frames_processed - plane_failures)
            
            print(f"   Success rate: {success_rate:.1f}%")
            print(f"   Plane failure rate: {failure_rate:.1f}%")
            print(f"   Avg IRLS time: {avg_irls_time:.1f}ms")
            print(f"   Total IRLS time: {total_irls_time:.0f}ms")
            print(f"   Estimator used: {ESTIMATOR_TYPE}")
        
        pipe.stop()
        client_socket.close()
        print("Cleanup complete.")
        sys.exit(0)

if __name__ == "__main__":
    main()