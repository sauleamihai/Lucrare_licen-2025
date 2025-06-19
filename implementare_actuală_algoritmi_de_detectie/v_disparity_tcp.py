#!/usr/bin/env python3


import math
import time
import sys
import socket
import argparse
import numpy as np
import pyrealsense2 as rs
import cv2

# ───────── Default Parameters ─────────
# Network parameters
DEFAULT_ESP32_IP = "10.42.0.220"
DEFAULT_ESP32_PORT = 12345
DEFAULT_ACK_TIMEOUT = 30.0
DEFAULT_SEND_INTERVAL = 1.0

# Grid parameters
DEFAULT_RADIAL_EDGES = "0.0,0.5,1.0,4.5"
DEFAULT_EMA_ALPHA = 0.5
DEFAULT_DEPTH_MIN = 0.15
DEFAULT_DEPTH_MAX = 4.5

# V-Disparity specific parameters
DEFAULT_D_MAX = 256
DEFAULT_HOUGH_THRESH = 20
DEFAULT_BASELINE_M = 0.05
DEFAULT_GROUND_MASK_THRESHOLD = 1

# Logistic intensity parameters
DEFAULT_LOGI_BASELINE = 20
DEFAULT_LOGI_MAXVAL = 255
DEFAULT_LOGI_DMID = 0.9
DEFAULT_LOGI_ALPHA = 2.0

# Ramp parameters
DEFAULT_RAMP_MAX_STEP = 10

# Image processing parameters
DEFAULT_GAUSSIAN_KERNEL = 5
DEFAULT_CANNY_LOW = 20
DEFAULT_CANNY_HIGH = 60
DEFAULT_HOUGH_MIN_THETA = -15  # degrees
DEFAULT_HOUGH_MAX_THETA = 15   # degrees

# ───────── Command‐Line Parsing ─────────
parser = argparse.ArgumentParser(description="V-Disparity ESP32 TCP Client - All parameters configurable")

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

# Radial edges
parser.add_argument("--radial-edges", type=str, default=DEFAULT_RADIAL_EDGES, 
                    help="Radial edges as comma-separated values")

# V-Disparity parameters
parser.add_argument("--d-max", type=int, default=DEFAULT_D_MAX, 
                    help="Maximum disparity value")
parser.add_argument("--hough-threshold", type=int, default=DEFAULT_HOUGH_THRESH, 
                    help="Hough lines threshold")
parser.add_argument("--baseline", type=float, default=DEFAULT_BASELINE_M, 
                    help="Stereo baseline in meters")
parser.add_argument("--ground-mask-threshold", type=int, default=DEFAULT_GROUND_MASK_THRESHOLD, 
                    help="Ground masking threshold")

# Image processing parameters
parser.add_argument("--gaussian-kernel", type=int, default=DEFAULT_GAUSSIAN_KERNEL, 
                    help="Gaussian blur kernel size (odd number)")
parser.add_argument("--canny-low", type=int, default=DEFAULT_CANNY_LOW, 
                    help="Canny edge detection low threshold")
parser.add_argument("--canny-high", type=int, default=DEFAULT_CANNY_HIGH, 
                    help="Canny edge detection high threshold")
parser.add_argument("--hough-min-theta", type=float, default=DEFAULT_HOUGH_MIN_THETA, 
                    help="Hough lines minimum theta in degrees")
parser.add_argument("--hough-max-theta", type=float, default=DEFAULT_HOUGH_MAX_THETA, 
                    help="Hough lines maximum theta in degrees")

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
parser.add_argument("--no-histogram-eq", action="store_true", 
                    help="Disable histogram equalization")
parser.add_argument("--no-gaussian-blur", action="store_true", 
                    help="Disable Gaussian blur")
parser.add_argument("--verbose", "-v", action="store_true", 
                    help="Enable verbose output")
parser.add_argument("--debug", action="store_true", 
                    help="Enable debug output with v-disparity visualization")

args = parser.parse_args()

# Apply parsed arguments
ESP32_IP = args.esp_ip
ESP32_PORT = args.esp_port
ACK_TIMEOUT = args.ack_timeout
SEND_INTERVAL = args.send_interval

DEPTH_MIN = args.depth_min
DEPTH_MAX = args.depth_max

# Parse radial edges
try:
    RADIAL_EDGES = np.array([float(x.strip()) for x in args.radial_edges.split(',')])
    GRID_R = len(RADIAL_EDGES) - 1
    GRID_A = 16
except:
    print(f"Error parsing radial edges: {args.radial_edges}")
    RADIAL_EDGES = np.array([0.0, 0.5, 1.0, 4.5])
    GRID_R = 3
    GRID_A = 16

D_MAX = args.d_max
HOUGH_THRESH = args.hough_threshold
BASELINE_M = args.baseline
GROUND_MASK_THRESHOLD = args.ground_mask_threshold

GAUSSIAN_KERNEL = args.gaussian_kernel
CANNY_LOW = args.canny_low
CANNY_HIGH = args.canny_high
HOUGH_MIN_THETA = math.radians(args.hough_min_theta)
HOUGH_MAX_THETA = math.radians(args.hough_max_theta)

LOGI_BASELINE = args.logi_baseline
LOGI_MAXVAL = args.logi_maxval
LOGI_DMID = args.logi_dmid
LOGI_ALPHA = args.logi_alpha

EMA_ALPHA = args.ema_alpha
RAMP_MAX_STEP = args.ramp_max_step

USE_EMA = not args.no_ema
USE_RAMP = not args.no_ramp
USE_HISTOGRAM_EQ = not args.no_histogram_eq
USE_GAUSSIAN_BLUR = not args.no_gaussian_blur
VERBOSE = args.verbose
DEBUG = args.debug

# Validation
if GAUSSIAN_KERNEL % 2 == 0:
    print("Warning: Gaussian kernel should be odd, adjusting...")
    GAUSSIAN_KERNEL += 1

if not (0.0 <= EMA_ALPHA <= 1.0):
    print("Warning: EMA alpha should be between 0.0 and 1.0")

if DEPTH_MIN >= DEPTH_MAX:
    print("Error: depth-min must be less than depth-max")
    sys.exit(1)

if D_MAX <= 0:
    print("Error: d-max must be positive")
    sys.exit(1)

if VERBOSE or DEBUG:
    print(f"   V-DISPARITY CONFIGURATION:")
    print(f"   Network: {ESP32_IP}:{ESP32_PORT}")
    print(f"   Depth range: {DEPTH_MIN:.2f}m - {DEPTH_MAX:.2f}m")
    print(f"   Radial edges: {RADIAL_EDGES}")
    print(f"   V-Disparity: D_max={D_MAX}, baseline={BASELINE_M:.3f}m")
    print(f"   Hough: threshold={HOUGH_THRESH}, θ=[{math.degrees(HOUGH_MIN_THETA):.0f}°, {math.degrees(HOUGH_MAX_THETA):.0f}°]")
    print(f"   Canny: low={CANNY_LOW}, high={CANNY_HIGH}")
    print(f"   Gaussian: kernel={GAUSSIAN_KERNEL}, enabled={USE_GAUSSIAN_BLUR}")
    print(f"   Histogram EQ: {USE_HISTOGRAM_EQ}")
    print(f"   Logistic: baseline={LOGI_BASELINE}, max={LOGI_MAXVAL}, mid={LOGI_DMID:.2f}, α={LOGI_ALPHA:.1f}")
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
                
            time.sleep(0.4)  # Allow ESP32 time to process
            response = sock.recv(1024).decode().strip()
            
            if response == "ACK":
                if DEBUG:
                    print("[TCP] ACK received!")
                return True
            else:
                print(f"[TCP] Unexpected response: {response}")
                
        except socket.timeout:
            print("[TCP] Timeout waiting for ACK; retrying…")
        except Exception as e:
            print(f"[TCP] Error sending matrix: {e}")
            
        retries -= 1
        time.sleep(1.0)
        
    print("[TCP] Failed to receive ACK after multiple retries.")
    return False

# ═══════════════════════════════════════════════════════════════════════════
# V-DISPARITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def build_v_disparity(depth_image, depth_scale, fx, fy, ppx, ppy):
    H, W = depth_image.shape
    v_disp = np.zeros((H, D_MAX), dtype=np.uint32)
    
    # Convert depth to Z_m and compute disparity
    Z_m = depth_image.astype(float) * depth_scale
    with np.errstate(divide="ignore", invalid="ignore"):
        disp_f = (fx * BASELINE_M) / Z_m
        disp_f[~np.isfinite(disp_f)] = 0
    
    # Quantize disparity
    disp_q = np.floor(disp_f).astype(np.int32)
    disp_q = np.clip(disp_q, 0, D_MAX - 1)
    
    # Build per-row histograms
    for v in range(H):
        row_disp = disp_q[v, :]
        valid_mask = (row_disp > 0)
        if np.any(valid_mask):
            hist = np.bincount(row_disp[valid_mask], minlength=D_MAX)
            v_disp[v, :] = hist
    
    # Normalize for processing
    max_val = v_disp.max(initial=1)
    v_disp_norm = ((v_disp.astype(float) / max_val) * 255.0).astype(np.uint8)
    
    if DEBUG:
        print(f"[V-DISP] Built {H}x{D_MAX} v-disparity, max={max_val}")
    
    return v_disp_norm, disp_q

def detect_ground_line(v_disp_norm):
    
    # Apply Gaussian blur if enabled
    if USE_GAUSSIAN_BLUR:
        v_blur = cv2.GaussianBlur(v_disp_norm, (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL), 0)
    else:
        v_blur = v_disp_norm.copy()
    
    # Apply histogram equalization if enabled
    if USE_HISTOGRAM_EQ:
        v_eq = cv2.equalizeHist(v_blur)
    else:
        v_eq = v_blur
    
    # Canny edge detection
    edges = cv2.Canny(v_eq, CANNY_LOW, CANNY_HIGH)
    
    if DEBUG:
        edge_count = np.count_nonzero(edges)
        print(f"[CANNY] Found {edge_count} edge pixels with thresholds [{CANNY_LOW}, {CANNY_HIGH}]")
    
    # HoughLines with configurable parameters
    lines = cv2.HoughLines(
        edges,
        rho=1,
        theta=np.pi / 180.0,
        threshold=HOUGH_THRESH,
        srn=0,
        stn=0,
        min_theta=HOUGH_MIN_THETA,
        max_theta=HOUGH_MAX_THETA
    )
    
    if lines is None:
        if DEBUG:
            print(f"[HOUGH] No lines found with threshold {HOUGH_THRESH}")
        return None
    
    if DEBUG:
        print(f"[HOUGH] Found {len(lines)} lines")
    
    # Use the first (strongest) line
    rho, theta = lines[0][0]
    sin_t, cos_t = math.sin(theta), math.cos(theta)
    
    if abs(sin_t) < 1e-6:
        if DEBUG:
            print("[HOUGH] Line too horizontal, rejecting")
        return None
    
    # Convert to slope-intercept form
    m = -cos_t / sin_t
    c = rho / sin_t
    
    if DEBUG:
        print(f"[HOUGH] Ground line: disparity = {m:.3f} * v + {c:.1f}")
    
    return m, c

def mask_ground_pixels(disp_q, m, c, threshold=None):
    if threshold is None:
        threshold = GROUND_MASK_THRESHOLD
        
    H, W = disp_q.shape
    line_vals = (m * np.arange(H) + c).astype(np.int32)
    line_vals = np.clip(line_vals, 0, D_MAX - 1)
    
    ground_mask = np.zeros((H, W), dtype=bool)
    
    for v in range(H):
        dv = disp_q[v, :]
        diff = np.abs(dv - line_vals[v])
        ground_mask[v, :] = (diff <= threshold) & (dv > 0)
    
    if DEBUG:
        ground_pixels = np.count_nonzero(ground_mask)
        total_pixels = H * W
        print(f"[MASK] {ground_pixels}/{total_pixels} ({ground_pixels/total_pixels*100:.1f}%) pixels marked as ground")
    
    return ground_mask

def depth_to_points_non_ground(depth_image, ground_mask, fx, fy, ppx, ppy, depth_scale):
    H, W = depth_image.shape
    points = []
    
    for v in range(H):
        for u in range(W):
            if not ground_mask[v, u]:
                z_raw = depth_image[v, u]
                if z_raw == 0:
                    continue
                Z = float(z_raw) * depth_scale
                X = (u - ppx) * Z / fx
                Y = (v - ppy) * Z / fy
                points.append([X, Y, Z])
    
    if len(points) == 0:
        return np.empty((0, 3), dtype=float)
    
    result = np.array(points, dtype=float)
    
    if DEBUG:
        print(f"[POINTS] Generated {len(result)} non-ground 3D points")
    
    return result

def compute_votes(pts, ANG_EDGES):
    if pts.shape[0] == 0:
        return np.zeros((GRID_R, GRID_A // 2), dtype=float)
    
    # Filter by height
    Y = pts[:, 1]
    mask = (Y > 0.02) & (Y < 1.9)
    live = pts[mask]
    
    if live.shape[0] == 0:
        return np.zeros((GRID_R, GRID_A // 2), dtype=float)
    
    # Convert to polar coordinates
    X = live[:, 0]
    Z = live[:, 2]
    r = np.hypot(X, Z)
    phi = np.clip(np.arctan2(X, Z), ANG_EDGES[0], ANG_EDGES[-1] - 1e-6)
    
    # Create histogram
    H8, _, _ = np.histogram2d(r, phi, bins=[RADIAL_EDGES, ANG_EDGES])
    
    if DEBUG:
        total_votes = np.sum(H8)
        print(f"[VOTES] {live.shape[0]} points → {total_votes} votes")
    
    return H8.astype(float)

def duplicate_angular_bins(H8):
    first8 = H8[:, : (GRID_A // 2)]
    return np.repeat(first8, 2, axis=1).astype(int)

def logistical_intensity(d, baseline=None, maxVal=None, d_mid=None, alpha=None):
    if baseline is None: baseline = LOGI_BASELINE
    if maxVal is None: maxVal = LOGI_MAXVAL
    if d_mid is None: d_mid = LOGI_DMID
    if alpha is None: alpha = LOGI_ALPHA
    
    if d <= 0.10:
        return maxVal
    if d >= 2.0:
        return 0
    
    return int(baseline + (maxVal - baseline) / (1.0 + math.exp(alpha * (d - d_mid))))

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
    print("=== V-Disparity ESP32 TCP Client ===")
    print("Algorithm: V-Disparity → Hough Lines → Ground Masking → Polar Histogram")
    
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
    
    # Initialize RealSense
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipe.start(cfg)
    
    # Read intrinsics & depth scale
    sensor = profile.get_device().first_depth_sensor()
    d_scale = sensor.get_depth_scale()
    intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy
    H, W = intr.height, intr.width
    
    if VERBOSE:
        print(f"RealSense: {W}x{H}, fx={fx:.1f}, depth_scale={d_scale:.6f}")
    
    # Compute FOV & angular edges
    FOV = 2 * math.atan((W / 2) / fx)
    ANG_EDGES = np.linspace(-FOV / 2, FOV / 2, (GRID_A // 2) + 1)
    
    if VERBOSE:
        print(f"FOV: {math.degrees(FOV):.1f}°, Angular bins: {len(ANG_EDGES)-1}")
    
    # Initialize variables
    H_prev = None
    last_send = 0.0
    last_ack = 0.0
    
    # Statistics
    frames_processed = 0
    successful_sends = 0
    ground_line_failures = 0
    
    print("Starting v-disparity processing... Press Ctrl+C to stop.\n")
    
    try:
        while True:
            now = time.time()
            if now - last_send < SEND_INTERVAL:
                time.sleep(0.01)
                continue
            
            # Grab depth frame
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
            
            depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
            
            # Build v-disparity map
            v_disp_norm, disp_q = build_v_disparity(depth_image, d_scale, fx, fy, ppx, ppy)
            
            # Detect ground line
            ground_line = detect_ground_line(v_disp_norm)
            
            if ground_line is None:
                ground_line_failures += 1
                if VERBOSE:
                    print(f"[WARN] No ground line detected (total failures: {ground_line_failures})")
                # Send zero matrix
                zero_mat = np.zeros((GRID_R, GRID_A), dtype=int)
                for row in zero_mat:
                    print(",".join(str(int(v)) for v in row))
                print("---")
                last_send = now
                frames_processed += 1
                continue
            
            m, c = ground_line
            
            # Mask ground pixels
            ground_mask = mask_ground_pixels(disp_q, m, c)
            
            # Convert non-ground to 3D points
            pts3D = depth_to_points_non_ground(depth_image, ground_mask, fx, fy, ppx, ppy, d_scale)
            
            # Compute histogram
            raw_H8 = compute_votes(pts3D, ANG_EDGES)
            
            # Apply EMA if enabled
            if USE_EMA:
                # Note: In original, EMA was not properly implemented - just copying raw
                ema8 = raw_H8.copy()
            else:
                ema8 = raw_H8
            
            # Duplicate angular bins
            H16 = duplicate_angular_bins(ema8)
            
            # Clip and apply ramp
            H16 = np.clip(H16, 0, 255).astype(np.uint8)
            H16_ramped = apply_ramp(H16, H_prev)
            
            # Print matrix
            if VERBOSE or DEBUG:
                matrix_sum = np.sum(H16_ramped)
                matrix_max = np.max(H16_ramped)
                print(f"# Frame {frames_processed}: sum={matrix_sum}, max={matrix_max}")
            
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
                failure_rate = ground_line_failures / frames_processed * 100
                
                print(f"\n V-DISPARITY STATS after {successful_sends} sends:")
                print(f"   Frames processed: {frames_processed}")
                print(f"   Success rate: {success_rate:.1f}%")
                print(f"   Ground line failure rate: {failure_rate:.1f}%")
                print()
    
    except KeyboardInterrupt:
        print("\n Interrupted by user. Exiting…")
    
    finally:
        # Final statistics
        print(f"\n FINAL V-DISPARITY STATS:")
        print(f"   Frames processed: {frames_processed}")
        print(f"   Successful sends: {successful_sends}")
        print(f"   Ground line failures: {ground_line_failures}")
        
        if frames_processed > 0:
            success_rate = successful_sends / frames_processed * 100
            failure_rate = ground_line_failures / frames_processed * 100
            print(f"   Success rate: {success_rate:.1f}%")
            print(f"   Ground line failure rate: {failure_rate:.1f}%")
        
        pipe.stop()
        client_socket.close()
        print(" Cleanup complete.")
        sys.exit(0)

if __name__ == "__main__":
    main()