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

# Polar grid parameters
DEFAULT_RADIAL_EDGES = "0.0,0.5,1.0,4.5"
DEFAULT_EMA_ALPHA = 0.07
DEFAULT_GROUND_EPS = 0.02
DEFAULT_MAX_H = 1.9
DEFAULT_DEPTH_MIN = 0.15
DEFAULT_DEPTH_MAX = 4.5

# RANSAC parameters
DEFAULT_RANSAC_TOL = 0.10
DEFAULT_RANSAC_IT = 60
DEFAULT_PLANE_A = 0.8
DEFAULT_MIN_GROUND_POINTS = 50

# ───────── Command‐Line Parsing ─────────
parser = argparse.ArgumentParser(description="Polar RANSAC ESP32 TCP Client - All parameters configurable")

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

# Radial edges
parser.add_argument("--radial-edges", type=str, default=DEFAULT_RADIAL_EDGES, 
                    help="Radial edges as comma-separated values (e.g., '0.0,0.5,1.0,4.5')")

# RANSAC parameters
parser.add_argument("--ransac-tolerance", type=float, default=DEFAULT_RANSAC_TOL, 
                    help="RANSAC distance tolerance in meters")
parser.add_argument("--ransac-iterations", type=int, default=DEFAULT_RANSAC_IT, 
                    help="Number of RANSAC iterations")
parser.add_argument("--plane-alpha", type=float, default=DEFAULT_PLANE_A, 
                    help="Plane blending factor (0.0-1.0)")
parser.add_argument("--min-ground-points", type=int, default=DEFAULT_MIN_GROUND_POINTS, 
                    help="Minimum points required for ground plane fitting")

# EMA parameters
parser.add_argument("--ema-alpha", type=float, default=DEFAULT_EMA_ALPHA, 
                    help="EMA alpha for smoothing (0.0-1.0)")

# Processing options
parser.add_argument("--no-ema", action="store_true", 
                    help="Disable EMA smoothing")
parser.add_argument("--no-normalization", action="store_true", 
                    help="Skip normalization to [0,255]")
parser.add_argument("--verbose", "-v", action="store_true", 
                    help="Enable verbose output")
parser.add_argument("--debug", action="store_true", 
                    help="Enable debug output with detailed statistics")

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

# Parse radial edges
try:
    RADIAL_EDGES = np.array([float(x.strip()) for x in args.radial_edges.split(',')])
    GRID_R = len(RADIAL_EDGES) - 1
    GRID_A = 16  # Fixed angular bins
except:
    print(f"Error parsing radial edges: {args.radial_edges}")
    print("Using default: 0.0,0.5,1.0,4.5")
    RADIAL_EDGES = np.array([0.0, 0.5, 1.0, 4.5])
    GRID_R = 3
    GRID_A = 16

RANSAC_TOL = args.ransac_tolerance
RANSAC_IT = args.ransac_iterations
PLANE_A = args.plane_alpha
MIN_GROUND_POINTS = args.min_ground_points

EMA_ALPHA = args.ema_alpha
USE_EMA = not args.no_ema
USE_NORMALIZATION = not args.no_normalization
VERBOSE = args.verbose
DEBUG = args.debug

# Validation
if not (0.0 <= EMA_ALPHA <= 1.0):
    print("Warning: EMA alpha should be between 0.0 and 1.0")
if not (0.0 <= PLANE_A <= 1.0):
    print("Warning: Plane alpha should be between 0.0 and 1.0")
if DEPTH_MIN >= DEPTH_MAX:
    print("Error: depth-min must be less than depth-max")
    sys.exit(1)
if RANSAC_TOL <= 0:
    print("Error: RANSAC tolerance must be positive")
    sys.exit(1)
if RANSAC_IT <= 0:
    print("Error: RANSAC iterations must be positive")
    sys.exit(1)

if VERBOSE or DEBUG:
    print(f"   CONFIGURATION:")
    print(f"   Network: {ESP32_IP}:{ESP32_PORT}")
    print(f"   Timeouts: ACK={ACK_TIMEOUT}s, Interval={SEND_INTERVAL}s")
    print(f"   Depth range: {DEPTH_MIN:.2f}m - {DEPTH_MAX:.2f}m")
    print(f"   Obstacle height: {GROUND_EPS:.3f}m - {MAX_H:.2f}m")
    print(f"   Radial edges: {RADIAL_EDGES}")
    print(f"   Grid size: {GRID_R}x{GRID_A}")
    print(f"   RANSAC: {RANSAC_IT} iterations, tolerance={RANSAC_TOL:.3f}m")
    print(f"   Plane blending: α={PLANE_A:.2f}")
    print(f"   EMA: {'ON' if USE_EMA else 'OFF'} (α={EMA_ALPHA:.3f})")
    print(f"   Normalization: {'ON' if USE_NORMALIZATION else 'OFF'}")
    print(f"   Min ground points: {MIN_GROUND_POINTS}")
    print()

# ═══════════════════════════════════════════════════════════════════════════
# TCP COMMUNICATION
# ═══════════════════════════════════════════════════════════════════════════

def send_matrix_with_ack(sock, matrix, retries=3):
    matrix_str = ";".join(",".join(map(str, row)) for row in matrix)
    
    for attempt in range(retries):
        try:
            sock.sendall(matrix_str.encode())
            if DEBUG:
                print(f"[TCP] Attempt {attempt+1}: Sent {len(matrix_str)} bytes")
            
            time.sleep(0.4)  # pause before reading response
            response = sock.recv(1024).decode().strip()
            
            if response == "ACK":
                if DEBUG:
                    print(f"[TCP] ACK received on attempt {attempt+1}")
                return True
            else:
                print(f"[TCP] Unexpected response: {response}")
                
        except socket.timeout:
            print(f"[TCP] Timeout waiting for ACK, attempt {attempt+1}/{retries}")
        except Exception as e:
            print(f"[TCP] Error sending matrix (attempt {attempt+1}): {e}")
            
        if attempt < retries - 1:  # Don't sleep after last attempt
            time.sleep(1.0)
    
    print(f"[TCP] Failed to receive ACK after {retries} retries")
    return False

# ═══════════════════════════════════════════════════════════════════════════
# PLANE DETECTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def depth_to_points(depth_map, fx, fy, ppx, ppy, d_scale):
    ys, xs = np.nonzero(depth_map)
    zs = depth_map[ys, xs] * d_scale
    X = (xs - ppx) * zs / fx
    Y = (ys - ppy) * zs / fy
    return np.vstack((X, Y, zs)).T

def plane_ransac(pts, prev_plane=None):
    best, best_count = None, 0
    
    if DEBUG:
        print(f"[RANSAC] Processing {len(pts)} points, {RANSAC_IT} iterations")
    
    for iteration in range(RANSAC_IT):
        if pts.shape[0] < 3:
            break
            
        # Sample 3 random points
        idx = np.random.choice(pts.shape[0], 3, replace=False)
        s = pts[idx]
        
        # Compute plane normal
        n = np.cross(s[1] - s[0], s[2] - s[0])
        if np.linalg.norm(n) < 1e-6:
            continue
            
        # Plane equation Ax + By + Cz + D = 0
        A, B, C = n
        D = -n.dot(s[0])
        
        # Count inliers
        dists = np.abs((pts @ n) + D) / np.linalg.norm(n)
        inliers = (dists < RANSAC_TOL).sum()
        
        if inliers > best_count:
            best_count, best = inliers, np.array([A, B, C, D], float)
            
            if DEBUG and iteration % 10 == 0:
                print(f"[RANSAC] Iteration {iteration}: {inliers} inliers (best: {best_count})")
    
    if best is None:
        if DEBUG:
            print("[RANSAC] No valid plane found")
        return prev_plane
    
    if prev_plane is None:
        if DEBUG:
            print(f"[RANSAC] New plane: {best_count} inliers")
        return best
    
    # Blend with previous plane
    blended = PLANE_A * best + (1 - PLANE_A) * prev_plane
    
    if DEBUG:
        print(f"[RANSAC] Blended plane: {best_count} inliers, α={PLANE_A:.2f}")
    
    return blended

def compute_votes(pts, plane, ANG_EDGES):
    A, B, C, D = plane
    
    # Normalized distance to plane
    h = ((pts @ np.array([A, B, C])) + D) / math.sqrt(A*A + B*B + C*C)
    
    # Filter points above ground plane
    live = pts[(h > GROUND_EPS) & (h < MAX_H)]
    
    if live.shape[0] == 0:
        if DEBUG:
            print("[VOTES] No points above ground plane")
        return np.zeros((GRID_R, GRID_A//2), dtype=float)
    
    # Convert to polar coordinates
    r = np.hypot(live[:, 0], live[:, 2])
    phi = np.clip(np.arctan2(live[:, 0], live[:, 2]),
                  ANG_EDGES[0], ANG_EDGES[-1] - 1e-6)
    
    # Create 2D histogram
    H, _, _ = np.histogram2d(r, phi, bins=[RADIAL_EDGES, ANG_EDGES])
    
    if DEBUG:
        total_votes = np.sum(H)
        print(f"[VOTES] {live.shape[0]} points → {total_votes} votes in {GRID_R}x{GRID_A//2} grid")
    
    return H.astype(float)

def duplicate_angular_bins(H8):
    first8 = H8[:, : (GRID_A // 2)]
    return np.repeat(first8, 2, axis=1)

# ═══════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=== Polar RANSAC ESP32 TCP Client ===")
    print("Algorithm: RANSAC Ground Plane + Polar Histogram + EMA Smoothing")
    
    if VERBOSE:
        print(f"ESP32: {ESP32_IP}:{ESP32_PORT}, Protocol: Semicolon-separated CSV + ACK")
    
    # ─── Set up TCP Socket ───
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    
    try:
        client_socket.connect((ESP32_IP, ESP32_PORT))
        print(f" Connected to ESP32 @ {ESP32_IP}:{ESP32_PORT}")
    except Exception as e:
        print(f"[ERROR] Cannot connect to ESP32 @ {ESP32_IP}:{ESP32_PORT} → {e}")
        sys.exit(1)
    
    client_socket.settimeout(5)
    
    # ─── Start RealSense pipeline ───
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipe.start(cfg)
    
    # ─── Get intrinsics and scale ───
    sensor = profile.get_device().first_depth_sensor()
    d_scale = sensor.get_depth_scale()
    intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy
    width, height = intr.width, intr.height
    
    if VERBOSE:
        print(f"RealSense: {width}x{height}, fx={fx:.1f}, depth_scale={d_scale:.6f}")
    
    # ─── Compute angular edges for 8 bins (duplicate to 16) ───
    FOV = 2 * math.atan((width / 2) / fx)
    ANG_EDGES = np.linspace(-FOV/2, FOV/2, GRID_A//2 + 1)
    
    if VERBOSE:
        print(f"FOV: {math.degrees(FOV):.1f}°, Angular bins: {len(ANG_EDGES)-1}")
    
    # ─── Initialize variables ───
    ema_hist = np.zeros((GRID_R, GRID_A//2), dtype=float)
    prev_plane = None
    last_send = 0.0
    last_ack = 0.0
    
    # Statistics
    frames_processed = 0
    successful_sends = 0
    plane_failures = 0
    total_points_processed = 0
    
    print("Capturing frames... Press Ctrl+C to stop.\n")
    
    try:
        while True:
            now = time.time()
            if now - last_send < SEND_INTERVAL:
                time.sleep(0.01)
                continue
            
            # 1) Grab a valid depth frame
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
            
            # 2) Convert to NumPy & mask out-of-range depths
            depth_image = np.asanyarray(depth_frame.get_data(), dtype=float)
            Z_m = depth_image * d_scale
            mask = (Z_m >= DEPTH_MIN) & (Z_m <= DEPTH_MAX)
            depth_image = np.where(mask, depth_image, 0.0)
            
            # 3) Build 3D point cloud
            ys, xs = np.nonzero(depth_image)
            if len(ys) == 0:
                if VERBOSE:
                    print("[WARN] No valid depth pixels in range")
                continue
                
            zs = depth_image[ys, xs] * d_scale
            pts = np.vstack(((xs - ppx) * zs / fx,
                            (ys - ppy) * zs / fy,
                            zs)).T
            
            total_points_processed += len(pts)
            
            # 4) Select ground candidate points (bottom 25% by Y)
            if pts.shape[0] > 0:
                y_vals = pts[:, 1]
                thresh = np.percentile(y_vals, 25)
                ground_pts = pts[y_vals < thresh]
            else:
                ground_pts = np.empty((0, 3), dtype=float)
            
            # 5) Fit ground plane via RANSAC
            plane = None
            if ground_pts.shape[0] >= MIN_GROUND_POINTS:
                plane = plane_ransac(ground_pts, prev_plane)
                if plane is not None:
                    prev_plane = plane
            
            if plane is None:
                plane_failures += 1
                if VERBOSE:
                    print(f"[WARN] Plane fit failed (total failures: {plane_failures})")
                # Reuse previous EMA
                H8 = ema_hist
            else:
                # 6) Compute raw histogram
                raw_hist = compute_votes(pts, plane, ANG_EDGES)
                
                # 7) Apply EMA smoothing
                if USE_EMA:
                    ema_hist = (1 - EMA_ALPHA) * ema_hist + EMA_ALPHA * raw_hist
                    H8 = ema_hist
                else:
                    H8 = raw_hist
            
            # 8) Duplicate 8→16 angular bins
            H16 = duplicate_angular_bins(H8.astype(int))
            
            # 9) Normalize to [0..255] if enabled
            if USE_NORMALIZATION:
                flat = H16.flatten()
                max_val = flat.max() if flat.size > 0 else 0
                if max_val <= 0:
                    Hnorm = np.zeros_like(H16, dtype=int)
                else:
                    Hnorm = np.floor((H16.astype(float) * 255) / max_val).astype(int)
            else:
                Hnorm = H16.astype(int)
            
            # 10) Print matrix
            if VERBOSE or DEBUG:
                matrix_sum = np.sum(Hnorm)
                matrix_max = np.max(Hnorm)
                print(f"# Frame {frames_processed}: {len(pts)} points, "
                      f"sum={matrix_sum}, max={matrix_max}")
            
            for row in Hnorm:
                print(",".join(str(int(v)) for v in row))
            print("---")
            
            # 11) Send over TCP with ACK
            success = send_matrix_with_ack(client_socket, Hnorm)
            if success:
                last_send = now
                last_ack = now
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
                    print(f"[ERROR] No ACK for {ACK_TIMEOUT:.0f}s. Exiting.")
                    break
                else:
                    last_send = now
            
            frames_processed += 1
            
            # Print stats every 10 successful sends
            if DEBUG and successful_sends > 0 and successful_sends % 10 == 0:
                avg_points = total_points_processed / frames_processed
                success_rate = successful_sends / frames_processed * 100
                failure_rate = plane_failures / frames_processed * 100
                
                print(f"\n STATS after {successful_sends} successful sends:")
                print(f"   Frames processed: {frames_processed}")
                print(f"   Success rate: {success_rate:.1f}%")
                print(f"   Plane failure rate: {failure_rate:.1f}%")
                print(f"   Avg points per frame: {avg_points:.0f}")
                print(f"   Total points processed: {total_points_processed}")
                print()
    
    except KeyboardInterrupt:
        print("\n Interrupted by user. Exiting…")
    
    finally:
        # Final statistics
        elapsed_time = time.time() - (last_ack if last_ack > 0 else time.time())
        
        print(f"\n FINAL STATS:")
        print(f"   Frames processed: {frames_processed}")
        print(f"   Successful sends: {successful_sends}")
        print(f"   Plane failures: {plane_failures}")
        print(f"   Total points: {total_points_processed}")
        
        if frames_processed > 0:
            success_rate = successful_sends / frames_processed * 100
            failure_rate = plane_failures / frames_processed * 100
            avg_points = total_points_processed / frames_processed
            
            print(f"   Success rate: {success_rate:.1f}%")
            print(f"   Plane failure rate: {failure_rate:.1f}%")
            print(f"   Avg points per frame: {avg_points:.0f}")
        
        pipe.stop()
        client_socket.close()
        print(" Cleanup complete.")
        sys.exit(0)

if __name__ == "__main__":
    main()
