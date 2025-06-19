#!/usr/bin/env python3


import math
import argparse
import sys
import time
import socket
import numpy as np
import pyrealsense2 as rs
import cv2

# ───────── Network Parameters (ESP32 Compatible) ─────────
ESP32_IP = "10.42.0.220"        # Change to your ESP32's IP
ESP32_PORT = 12345              # Change to your ESP32's listening port
ACK_TIMEOUT = 30.0              # Seconds before giving up on ACK
SEND_INTERVAL = 1.0             # Seconds between each depth capture + send

# ───────── Configurable Params ─────────
GRID_R, GRID_A = 3, 16
MIN_GROUND_POINTS = 50
GROUND_EPS, MAX_H = 0.02, 1.9
RADIAL_EDGES = np.array([0.0, 0.5, 1.0, 4.5])
DEPTH_MIN, DEPTH_MAX = 0.15, 4.5

# EMA Parameters (like original script)
EMA_ALPHA = 0.07                # Exponential smoothing factor

# Temporal filtering params (reduced for ESP32 compatibility)
TEMPORAL_DECAY = 0.90           # Slower decay for stability
PRESENCE_THRESHOLD = 30         # Lower threshold
RAPID_DECAY = 0.7              # Less aggressive decay

# Fast Hybrid specific params (optimized for ESP32)
FAST_SAMPLE_SIZE = 1500         # Reduced for faster processing
HOUGH_RESOLUTION = 0.04         # Slightly lower resolution
PROSAC_ITERATIONS = 20          # Reduced iterations
STABILITY_THRESHOLD = 0.20      # More lenient for real-world

# ───────── Command‐Line Parsing ─────────
parser = argparse.ArgumentParser(description="Fast Hybrid ESP32 TCP Client - All parameters configurable")

# TCP/Network parameters
parser.add_argument("--esp-ip", type=str, default=ESP32_IP, help="ESP32 IP address")
parser.add_argument("--esp-port", type=int, default=ESP32_PORT, help="ESP32 port")
parser.add_argument("--interval", type=float, default=SEND_INTERVAL, help="Send interval seconds")
parser.add_argument("--ack-timeout", type=float, default=ACK_TIMEOUT, help="ACK timeout seconds")

# Depth processing parameters
parser.add_argument("--depth-min", type=float, default=DEPTH_MIN, help="Minimum depth in meters")
parser.add_argument("--depth-max", type=float, default=DEPTH_MAX, help="Maximum depth in meters")
parser.add_argument("--ground-eps", type=float, default=GROUND_EPS, help="Ground epsilon threshold")
parser.add_argument("--max-height", type=float, default=MAX_H, help="Maximum obstacle height")

# Radial edges (as comma-separated values)
parser.add_argument("--radial-edges", type=str, default="0.0,0.5,1.0,4.5", 
                    help="Radial edges as comma-separated values (e.g., '0.0,0.5,1.0,4.5')")

# Algorithm parameters
parser.add_argument("--min-ground-points", type=int, default=MIN_GROUND_POINTS, 
                    help="Minimum points required for ground plane fitting")
parser.add_argument("--sample-size", type=int, default=FAST_SAMPLE_SIZE, 
                    help="Maximum points to use for plane detection")
parser.add_argument("--prosac-iterations", type=int, default=PROSAC_ITERATIONS, 
                    help="Number of PROSAC iterations")
parser.add_argument("--hough-resolution", type=float, default=HOUGH_RESOLUTION, 
                    help="Hough transform resolution")
parser.add_argument("--stability-threshold", type=float, default=STABILITY_THRESHOLD, 
                    help="Plane stability threshold in radians")

# Temporal filtering parameters
parser.add_argument("--ema-alpha", type=float, default=EMA_ALPHA, 
                    help="EMA alpha for smoothing (0.0-1.0)")
parser.add_argument("--temporal-decay", type=float, default=TEMPORAL_DECAY, 
                    help="Temporal decay factor (0.0-1.0)")
parser.add_argument("--presence-threshold", type=int, default=PRESENCE_THRESHOLD, 
                    help="Minimum value to consider obstacle present")
parser.add_argument("--rapid-decay", type=float, default=RAPID_DECAY, 
                    help="Rapid decay factor for old detections")

# Processing options
parser.add_argument("--no-temporal", action="store_true", 
                    help="Disable temporal filtering")
parser.add_argument("--method", type=str, choices=["auto", "refinement", "hough", "prosac"], 
                    default="auto", help="Force specific detection method")
parser.add_argument("--verbose", "-v", action="store_true", 
                    help="Enable verbose output")

args = parser.parse_args()

# Apply parsed arguments
ESP32_IP = args.esp_ip
ESP32_PORT = args.esp_port
SEND_INTERVAL = args.interval
ACK_TIMEOUT = args.ack_timeout

DEPTH_MIN = args.depth_min
DEPTH_MAX = args.depth_max
GROUND_EPS = args.ground_eps
MAX_H = args.max_height

# Parse radial edges
try:
    RADIAL_EDGES = np.array([float(x.strip()) for x in args.radial_edges.split(',')])
    GRID_R = len(RADIAL_EDGES) - 1
except:
    print(f"Error parsing radial edges: {args.radial_edges}")
    print("Using default: 0.0,0.5,1.0,4.5")
    RADIAL_EDGES = np.array([0.0, 0.5, 1.0, 4.5])
    GRID_R = 3

MIN_GROUND_POINTS = args.min_ground_points
FAST_SAMPLE_SIZE = args.sample_size
PROSAC_ITERATIONS = args.prosac_iterations
HOUGH_RESOLUTION = args.hough_resolution
STABILITY_THRESHOLD = args.stability_threshold

EMA_ALPHA = args.ema_alpha
TEMPORAL_DECAY = args.temporal_decay
PRESENCE_THRESHOLD = args.presence_threshold
RAPID_DECAY = args.rapid_decay

USE_TEMPORAL = not args.no_temporal
FORCED_METHOD = args.method
VERBOSE = args.verbose

# Validation
if not (0.0 <= EMA_ALPHA <= 1.0):
    print("Warning: EMA alpha should be between 0.0 and 1.0")
if not (0.0 <= TEMPORAL_DECAY <= 1.0):
    print("Warning: Temporal decay should be between 0.0 and 1.0")
if DEPTH_MIN >= DEPTH_MAX:
    print("Error: depth-min must be less than depth-max")
    sys.exit(1)

if VERBOSE:
    print(f" CONFIGURATION:")
    print(f"   Network: {ESP32_IP}:{ESP32_PORT}, interval: {SEND_INTERVAL}s")
    print(f"   Depth range: {DEPTH_MIN:.2f}m - {DEPTH_MAX:.2f}m")
    print(f"   Radial edges: {RADIAL_EDGES}")
    print(f"   Sample size: {FAST_SAMPLE_SIZE}, PROSAC iter: {PROSAC_ITERATIONS}")
    print(f"   Temporal filter: {'ON' if USE_TEMPORAL else 'OFF'}")
    print(f"   Forced method: {FORCED_METHOD}")
    print()

# ═══════════════════════════════════════════════════════════════════════════
# ESP32 TCP COMMUNICATION (Exact same protocol as original)
# ═══════════════════════════════════════════════════════════════════════════

def send_matrix_with_ack(sock, matrix, retries=3):
   
    matrix_str = ";".join(",".join(map(str, row)) for row in matrix)
    for attempt in range(retries):
        try:
            sock.sendall(matrix_str.encode())
            time.sleep(0.4)  # pause before reading response
            response = sock.recv(1024).decode().strip()
            if response == "ACK":
                return True
            else:
                print(f"[TCP] Unexpected response: {response}")
        except socket.timeout:
            print("[TCP] Timeout waiting for ACK, retrying…")
        except Exception as e:
            print(f"[TCP] Error sending matrix: {e}")
        time.sleep(1.0)
    print("[TCP] Failed to receive ACK after retries")
    return False

# ═══════════════════════════════════════════════════════════════════════════
# TEMPORAL OBSTACLE FILTER (Optimized for ESP32)
# ═══════════════════════════════════════════════════════════════════════════

class LightweightTemporalFilter:
    
    def __init__(self, grid_shape=(GRID_R, GRID_A)):
        self.grid_shape = grid_shape
        self.accumulator = np.zeros(grid_shape, dtype=float)
        self.last_detection = np.zeros(grid_shape, dtype=int)
        self.frame_count = 0
        
    def update(self, current_histogram):
        self.frame_count += 1
        current_float = current_histogram.astype(float)
        
        # Track detections
        has_detection = current_float > PRESENCE_THRESHOLD
        self.last_detection[has_detection] = self.frame_count
        
        # Apply decay
        self._apply_temporal_decay()
        
        # Update accumulator
        self.accumulator += current_float
        self.accumulator = np.clip(self.accumulator, 0, 10000)  # Prevent overflow
        
        # Generate output
        return self._generate_filtered_output()
    
    def _apply_temporal_decay(self):
        frames_since_detection = self.frame_count - self.last_detection
        
        # Normal decay for recent detections
        recent_mask = frames_since_detection <= 3
        self.accumulator[recent_mask] *= TEMPORAL_DECAY
        
        # Faster decay for old detections
        old_mask = frames_since_detection > 3
        self.accumulator[old_mask] *= RAPID_DECAY
        
        # Clear very old detections
        very_old_mask = frames_since_detection > 10
        self.accumulator[very_old_mask] = 0
        
    def _generate_filtered_output(self):
        filtered = self.accumulator.copy()
        
        # Simple noise filtering
        filtered[filtered < 5] = 0
        
        # Gentle smoothing
        filtered = self._light_smoothing(filtered)
        
        return np.round(filtered).astype(int)
    
    def _light_smoothing(self, matrix):
        smoothed = matrix.copy()
        
        # Only smooth angular direction with simple averaging
        for r in range(matrix.shape[0]):
            row = matrix[r, :]
            for i in range(1, len(row)-1):
                # Simple 3-point average
                smoothed[r, i] = (row[i-1] + row[i] + row[i+1]) / 3
                
        return smoothed

# ═══════════════════════════════════════════════════════════════════════════
# FAST HYBRID PLANE DETECTOR (ESP32 Optimized)
# ═══════════════════════════════════════════════════════════════════════════

class ESP32FastHybridDetector:
    def __init__(self):
        self.last_plane = None
        self.plane_history = []
        self.method_used = "none"
        
    def detect_plane(self, points):
        if len(points) < MIN_GROUND_POINTS:
            return None, 0.0
            
        # Smart subsample pentru ESP32
        sampled_points = self._esp32_subsample(points)
        
        # Check for forced method
        if FORCED_METHOD != "auto":
            if FORCED_METHOD == "refinement" and self.last_plane is not None:
                refined_plane = self._quick_refinement(sampled_points)
                if refined_plane is not None:
                    self.method_used = "refinement"
                    self._update_history(refined_plane)
                    confidence = self._compute_confidence(points, refined_plane)
                    return refined_plane, confidence
            elif FORCED_METHOD == "hough":
                hough_plane = self._lightweight_hough(sampled_points)
                if hough_plane is not None:
                    self.method_used = "hough"
                    self._update_history(hough_plane)
                    confidence = self._compute_confidence(points, hough_plane)
                    return hough_plane, confidence
            elif FORCED_METHOD == "prosac":
                prosac_plane = self._fast_prosac(sampled_points)
                if prosac_plane is not None:
                    self.method_used = "prosac"
                    self._update_history(prosac_plane)
                    confidence = self._compute_confidence(points, prosac_plane)
                    return prosac_plane, confidence
        
        # Auto method selection (original logic)
        # Strategy 1: Quick refinement (most common)
        if self.last_plane is not None:
            refined_plane = self._quick_refinement(sampled_points)
            if refined_plane is not None:
                self.method_used = "refinement"
                self._update_history(refined_plane)
                confidence = self._compute_confidence(points, refined_plane)
                return refined_plane, confidence
        
        # Strategy 2: Lightweight Hough 3D
        hough_plane = self._lightweight_hough(sampled_points)
        if hough_plane is not None:
            confidence = self._compute_confidence(points, hough_plane)
            if confidence > 0.5:  # Lower threshold for ESP32
                self.method_used = "hough"
                self._update_history(hough_plane)
                return hough_plane, confidence
        
        # Strategy 3: Fast PROSAC
        prosac_plane = self._fast_prosac(sampled_points)
        if prosac_plane is not None:
            self.method_used = "prosac"
            self._update_history(prosac_plane)
            confidence = self._compute_confidence(points, prosac_plane)
            return prosac_plane, confidence
        
        # Fallback to temporal
        if len(self.plane_history) > 0:
            self.method_used = "temporal"
            return self.plane_history[-1], 0.4
        
        return None, 0.0
    
    def _esp32_subsample(self, points):
        if len(points) <= FAST_SAMPLE_SIZE:
            return points
        
        # Simple uniform sampling for speed
        indices = np.random.choice(len(points), FAST_SAMPLE_SIZE, replace=False)
        return points[indices]
    
    def _quick_refinement(self, points):
        distances = np.abs(points @ self.last_plane[:3] + self.last_plane[3])
        inlier_mask = distances < 0.12  # Slightly relaxed
        inliers = points[inlier_mask]
        
        if len(inliers) < 12:  # Lower threshold
            return None
        
        # Quick least squares
        centroid = np.mean(inliers, axis=0)
        centered = inliers - centroid
        
        # Subsample aggressive pentru viteză
        if len(centered) > 100:
            idx = np.random.choice(len(centered), 100, replace=False)
            centered = centered[idx]
        
        try:
            _, _, V = np.linalg.svd(centered)
            normal = V[-1]
            
            # Ensure consistent orientation
            if np.dot(normal, self.last_plane[:3]) < 0:
                normal = -normal
                
            D = -np.dot(normal, centroid)
            refined_plane = np.array([normal[0], normal[1], normal[2], D])
            
            # More lenient stability check
            angle_diff = np.arccos(np.clip(
                abs(np.dot(self.last_plane[:3], refined_plane[:3])), -1, 1))
            
            if angle_diff < STABILITY_THRESHOLD:
                return refined_plane
                
        except:
            pass
            
        return None
    
    def _lightweight_hough(self, points):
        if len(points) < 30:
            return None
        
        # Very aggressive subsampling
        if len(points) > 300:
            idx = np.random.choice(len(points), 300, replace=False)
            hough_points = points[idx]
        else:
            hough_points = points
        
        # Minimal parameter space
        max_dist = np.max(np.linalg.norm(hough_points, axis=1))
        rho_bins = np.arange(-max_dist, max_dist, HOUGH_RESOLUTION * 3)  # Coarser
        theta_bins = np.linspace(0, np.pi, 10)      # Very reduced
        phi_bins = np.linspace(0, 2*np.pi, 10)     # Very reduced
        
        accumulator = np.zeros((len(rho_bins), len(theta_bins), len(phi_bins)))
        
        # Sparse voting for speed
        for point in hough_points[::4]:  # Skip every 4th point
            x, y, z = point
            
            for i, theta in enumerate(theta_bins[::2]):  # Skip half
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                
                for j, phi in enumerate(phi_bins[::2]):  # Skip half
                    nx = sin_theta * np.cos(phi)
                    ny = sin_theta * np.sin(phi)
                    nz = cos_theta
                    
                    rho = x * nx + y * ny + z * nz
                    rho_idx = np.searchsorted(rho_bins, rho)
                    
                    if 0 <= rho_idx < len(rho_bins):
                        accumulator[rho_idx, i*2, j*2] += 1
        
        max_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
        
        if accumulator[max_idx] < 3:  # Very low threshold
            return None
        
        rho_idx, theta_idx, phi_idx = max_idx
        rho = rho_bins[rho_idx] if rho_idx < len(rho_bins) else 0
        theta = theta_bins[theta_idx*2] if theta_idx*2 < len(theta_bins) else 0
        phi = phi_bins[phi_idx*2] if phi_idx*2 < len(phi_bins) else 0
        
        A = np.sin(theta) * np.cos(phi)
        B = np.sin(theta) * np.sin(phi)
        C = np.cos(theta)
        D = -rho
        
        return np.array([A, B, C, D])
    
    def _fast_prosac(self, points):
        if len(points) < 10:
            return None
        
        best_plane = None
        best_inliers = 0
        
        # Very reduced iterations
        for iteration in range(10):  # Minimal iterations
            if len(points) < 3:
                break
                
            sample_idx = np.random.choice(len(points), 3, replace=False)
            sample = points[sample_idx]
            
            plane = self._fit_plane_sample(sample)
            if plane is None:
                continue
            
            # Quick inlier check
            distances = np.abs(points @ plane[:3] + plane[3])
            inliers = np.sum(distances < 0.08)  # Relaxed threshold
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = plane
                
                if inliers > 0.5 * len(points):  # Lower threshold
                    break
        
        return best_plane
    
    def _fit_plane_sample(self, sample):
        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        normal = np.cross(v1, v2)
        
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            return None
        
        normal = normal / norm
        d = -np.dot(normal, sample[0])
        return np.array([normal[0], normal[1], normal[2], d])
    
    def _compute_confidence(self, points, plane):
        distances = np.abs(points @ plane[:3] + plane[3])
        return np.sum(distances < 0.06) / len(points)
    
    def _update_history(self, plane):
        self.last_plane = plane
        self.plane_history.append(plane)
        
        if len(self.plane_history) > 2:  # Very short history
            self.plane_history.pop(0)

# ───────── Helper Functions (Compatible with original) ─────────
def depth_to_points(depth_map, fx, fy, ppx, ppy, d_scale):
    
    ys, xs = np.nonzero(depth_map)
    zs = depth_map[ys, xs] * d_scale
    X = (xs - ppx) * zs / fx
    Y = (ys - ppy) * zs / fy
    return np.vstack((X, Y, zs)).T

def compute_votes(pts, plane, ang_edges):
   
    if plane is None:
        return np.zeros((GRID_R, GRID_A // 2), dtype=float)
    
    A, B, C, D = plane
    # Normalized distance to plane
    h = ((pts @ np.array([A, B, C])) + D) / math.sqrt(A*A + B*B + C*C)
    live = pts[(h > GROUND_EPS) & (h < MAX_H)]
    
    if live.shape[0] == 0:
        return np.zeros((GRID_R, GRID_A // 2), dtype=float)
    
    X = live[:, 0]; Z = live[:, 2]
    r = np.hypot(X, Z)
    phi = np.clip(np.arctan2(X, Z), ang_edges[0], ang_edges[-1] - 1e-6)
    H8, _, _ = np.histogram2d(r, phi, bins=[RADIAL_EDGES, ang_edges])
    return H8.astype(float)

def duplicate_angular_bins(H8):
    """Duplicate bins - SAME AS ORIGINAL"""
    first8 = H8[:, : (GRID_A // 2)]
    return np.repeat(first8, 2, axis=1)

# ───────── Main Script (ESP32 Protocol Compatible) ─────────
def main():
    print("=== Fast Hybrid ESP32 TCP Client ===")
    print("Algorithm: Quick Refinement → Hough 3D → PROSAC + Temporal Filter")
    print(f"ESP32: {ESP32_IP}:{ESP32_PORT}, Interval: {SEND_INTERVAL}s")
    print("Protocol: Semicolon-separated CSV with ACK confirmation\n")
    
    # ─── Set up TCP Socket (EXACT SAME AS ORIGINAL) ───
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        client_socket.connect((ESP32_IP, ESP32_PORT))
        print(f" Connected to ESP32 @ {ESP32_IP}:{ESP32_PORT}")
    except Exception as e:
        print(f"[ERROR] Cannot connect to ESP32 @ {ESP32_IP}:{ESP32_PORT} → {e}")
        sys.exit(1)
    client_socket.settimeout(5)
    
    # ─── Start RealSense pipeline (SAME AS ORIGINAL) ───
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipe.start(cfg)
    
    # ─── Get intrinsics and scale (SAME AS ORIGINAL) ───
    sensor = profile.get_device().first_depth_sensor()
    d_scale = sensor.get_depth_scale()
    intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy
    width, height = intr.width, intr.height
    
    # ─── Compute angular edges (SAME AS ORIGINAL) ───
    FOV = 2 * math.atan((width / 2) / fx)
    ANG_EDGES = np.linspace(-FOV/2, FOV/2, GRID_A//2 + 1)
    
    # ─── Initialize components ───
    detector = ESP32FastHybridDetector()
    temporal_filter = LightweightTemporalFilter()
    
    # ─── Prepare EMA storage (SAME AS ORIGINAL) ───
    ema_hist = np.zeros((GRID_R, GRID_A//2), dtype=float)
    last_send = 0.0
    last_ack = 0.0
    
    # Statistics
    frames_processed = 0
    method_counts = {"refinement": 0, "hough": 0, "prosac": 0, "temporal": 0}
    successful_sends = 0
    
    print("Capturing one depth frame per second (Fast Hybrid + EMA + ESP32 TCP).")
    print("Press Ctrl+C to stop.\n")
    
    try:
        while True:
            now = time.time()
            if now - last_send < SEND_INTERVAL:
                time.sleep(0.01)
                continue
            
            # 1) Grab a valid depth frame (SAME AS ORIGINAL)
            depth_frame = None
            while depth_frame is None:
                try:
                    frames = pipe.wait_for_frames(timeout_ms=5000)
                    depth_frame = frames.get_depth_frame()
                    if not depth_frame:
                        depth_frame = None
                        print("[WARN] No valid depth frame; retrying…")
                except RuntimeError:
                    print("[WARN] Timeout waiting for frame; retrying…")
                    continue
            
            # 2) Convert and mask (SAME AS ORIGINAL)
            depth_image = np.asanyarray(depth_frame.get_data(), dtype=float)
            Z_m = depth_image * d_scale
            mask = (Z_m >= DEPTH_MIN) & (Z_m <= DEPTH_MAX)
            depth_image = np.where(mask, depth_image, 0.0)
            
            # 3) Build 3D point cloud (SAME AS ORIGINAL)
            ys, xs = np.nonzero(depth_image)
            zs = depth_image[ys, xs] * d_scale
            pts = np.vstack(((xs - ppx) * zs / fx,
                            (ys - ppy) * zs / fy,
                            zs)).T
            
            # 4) Fast Hybrid plane detection
            plane = None
            confidence = 0.0
            if pts.shape[0] > 0:
                y_vals = pts[:, 1]
                thresh = np.percentile(y_vals, 25)
                ground_pts = pts[y_vals < thresh]
                if ground_pts.shape[0] > MIN_GROUND_POINTS:
                    plane, confidence = detector.detect_plane(ground_pts)
                    method_counts[detector.method_used] += 1
            
            if plane is None:
                print(f"[WARN] Plane fit failed; reusing previous EMA. (Method: {detector.method_used})")
                H8 = ema_hist
            else:
                # 5) Compute raw histogram
                raw_hist = compute_votes(pts, plane, ANG_EDGES)
                # 6) Update EMA (SAME AS ORIGINAL)
                ema_hist = (1 - EMA_ALPHA) * ema_hist + EMA_ALPHA * raw_hist
                H8 = ema_hist
            
            # 7) Duplicate 8→16 angular bins (SAME AS ORIGINAL)
            H16 = duplicate_angular_bins(H8.astype(int))
            
            # 8) Apply temporal filtering
            if USE_TEMPORAL:
                H16_filtered = temporal_filter.update(H16)
            else:
                H16_filtered = H16
                if VERBOSE:
                    print("[INFO] Temporal filtering disabled")
            
            # 9) Normalize to [0..255] (SAME AS ORIGINAL)
            flat = H16_filtered.flatten()
            max_val = flat.max() if flat.size > 0 else 0
            if max_val <= 0:
                Hnorm = np.zeros_like(H16_filtered, dtype=int)
            else:
                Hnorm = np.floor((H16_filtered.astype(float) * 255) / max_val).astype(int)
            
            # 10) Print normalized matrix
            if VERBOSE:
                print(f"# Method: {detector.method_used}, Confidence: {confidence:.2f}, "
                      f"Max: {max_val}, Points: {pts.shape[0]}")
            else:
                print(f"# Method: {detector.method_used}, Confidence: {confidence:.2f}")
            
            for row in Hnorm:
                print(",".join(str(int(v)) for v in row))
            print("---")
            
            # 11) Send over TCP with ACK (EXACT SAME PROTOCOL)
            success = send_matrix_with_ack(client_socket, Hnorm)
            if success:
                last_send = now
                last_ack = now
                successful_sends += 1
                print(f"[TCP {time.strftime('%H:%M:%S')}] Sent matrix, ACK OK. "
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
            if successful_sends > 0 and successful_sends % 10 == 0:
                total_methods = sum(method_counts.values())
                print(f"\n STATS after {successful_sends} successful sends:")
                print(f"   Frames processed: {frames_processed}")
                print(f"   Success rate: {successful_sends/frames_processed*100:.1f}%")
                for method, count in method_counts.items():
                    if total_methods > 0:
                        percentage = (count / total_methods) * 100
                        print(f"   {method:12}: {count:3} ({percentage:5.1f}%)")
                print()
    
    except KeyboardInterrupt:
        print("\n Interrupted by user. Exiting…")
    
    finally:
        # Final statistics
        total_methods = sum(method_counts.values())
        print(f"\n FINAL STATS:")
        print(f"   Frames processed: {frames_processed}")
        print(f"   Successful sends: {successful_sends}")
        print(f"   Success rate: {successful_sends/max(1,frames_processed)*100:.1f}%")
        print(f" METHOD DISTRIBUTION:")
        for method, count in sorted(method_counts.items(), key=lambda x: x[1], reverse=True):
            if total_methods > 0:
                percentage = (count / total_methods) * 100
                print(f"   {method:12}: {count:3} ({percentage:5.1f}%)")
        
        pipe.stop()
        client_socket.close()
        print(" Cleanup complete.")
        sys.exit(0)

if __name__ == "__main__":
    main()