#!/usr/bin/env python3
"""
Intel RealSense Live Algorithm Test Suite
Real-world performance analysis using actual RealSense camera data
Provides comprehensive testing with live camera feeds and recorded datasets
"""

import numpy as np
import cv2
import time
import json
import threading
import queue
from collections import deque
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import pandas as pd
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import psutil
import gc
import argparse
import warnings
warnings.filterwarnings('ignore')

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("PyRealSense2 not available. Install with: pip install pyrealsense2")

# Set plotting style for professional documentation
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

@dataclass
class FrameMetrics:
    """Metrics for a single frame processing"""
    frame_id: int
    timestamp: float
    algorithm: str
    processing_time_ms: float
    plane_found: bool
    plane_equation: Optional[np.ndarray]
    num_points: int
    num_inliers: int
    convergence_iterations: int
    memory_usage_mb: float
    cpu_percent: float
    ground_coverage_percent: float
    obstacle_points: int
    
@dataclass
class TestSession:
    """Complete test session results"""
    session_id: str
    start_time: float
    end_time: float
    total_frames: int
    camera_config: Dict
    test_scenarios: List[str]
    algorithm_results: Dict[str, List[FrameMetrics]]

class RealSenseDataCapture:
    """Handles Intel RealSense camera data capture and management"""
    
    def __init__(self, width=640, height=480, fps=30, enable_rgb=True):
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_rgb = enable_rgb
        self.pipeline = None
        self.profile = None
        self.depth_scale = None
        self.intrinsics = None
        self.is_recording = False
        self.recorded_frames = []
        
    def initialize_camera(self):
        """Initialize RealSense camera"""
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("PyRealSense2 not available")
            
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Configure streams
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            if self.enable_rgb:
                config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
            
            # Start pipeline
            self.profile = self.pipeline.start(config)
            
            # Get depth sensor details
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            # Get camera intrinsics
            depth_stream = self.profile.get_stream(rs.stream.depth)
            self.intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
            
            print(f"RealSense camera initialized: {self.width}x{self.height} @ {self.fps}FPS")
            print(f"   Depth scale: {self.depth_scale}")
            print(f"   Intrinsics: fx={self.intrinsics.fx:.1f}, fy={self.intrinsics.fy:.1f}")
            
            return True
            
        except Exception as e:
            print(f"Failed to initialize RealSense camera: {e}")
            return False
    
    def get_camera_info(self):
        """Get camera configuration info"""
        if not self.intrinsics:
            return {}
        
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'fx': self.intrinsics.fx,
            'fy': self.intrinsics.fy,
            'ppx': self.intrinsics.ppx,
            'ppy': self.intrinsics.ppy,
            'depth_scale': self.depth_scale,
            'model': self.intrinsics.model.name if hasattr(self.intrinsics.model, 'name') else 'unknown'
        }
    
    def capture_frame(self, timeout_ms=5000):
        """Capture a single frame from camera"""
        if not self.pipeline:
            return None, None
            
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame() if self.enable_rgb else None
            
            if not depth_frame:
                return None, None
            
            # Convert to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
            color_image = np.asanyarray(color_frame.get_data()) if color_frame else None
            
            frame_data = {
                'depth': depth_image,
                'color': color_image,
                'timestamp': time.time(),
                'frame_number': depth_frame.get_frame_number()
            }
            
            # Store for recording if enabled
            if self.is_recording:
                self.recorded_frames.append(frame_data.copy())
            
            return frame_data, depth_frame.get_frame_number()
            
        except Exception as e:
            print(f"Frame capture failed: {e}")
            return None, None
    
    def start_recording(self):
        """Start recording frames"""
        self.is_recording = True
        self.recorded_frames = []
        print("Started recording frames")
    
    def stop_recording(self):
        """Stop recording frames"""
        self.is_recording = False
        print(f"Stopped recording. Captured {len(self.recorded_frames)} frames")
        return self.recorded_frames.copy()
    
    def save_recording(self, filename):
        """Save recorded frames to file"""
        if not self.recorded_frames:
            print("No frames to save")
            return
        
        # Save as compressed numpy archive
        np.savez_compressed(filename, 
                          frames=self.recorded_frames,
                          camera_info=self.get_camera_info())
        print(f"Saved {len(self.recorded_frames)} frames to {filename}")
    
    def load_recording(self, filename):
        """Load recorded frames from file"""
        try:
            data = np.load(filename, allow_pickle=True)
            self.recorded_frames = data['frames'].tolist()
            camera_info = data['camera_info'].item()
            
            # Update camera settings from loaded data
            self.width = camera_info['width']
            self.height = camera_info['height']
            self.depth_scale = camera_info['depth_scale']
            
            print(f"Loaded {len(self.recorded_frames)} frames from {filename}")
            return True
        except Exception as e:
            print(f"Failed to load recording: {e}")
            return False
    
    def cleanup(self):
        """Cleanup camera resources"""
        if self.pipeline:
            self.pipeline.stop()
            print("🔌 Camera disconnected")

class RealTimeAlgorithmTester:
    """Tests algorithms using live RealSense data"""
    
    def __init__(self, camera_capture):
        self.camera = camera_capture
        self.algorithms = {
            'IRLS': self.test_irls_algorithm,
            'RANSAC': self.test_ransac_algorithm,
            'V-Disparity': self.test_vdisparity_algorithm
        }
        self.test_results = {}
        self.live_metrics = {alg: deque(maxlen=100) for alg in self.algorithms}
        
    def depth_to_points(self, depth_image):
        """Convert depth image to 3D point cloud"""
        if not self.camera.intrinsics:
            return np.empty((0, 3))
        
        # Get valid depth pixels
        ys, xs = np.nonzero(depth_image)
        if len(xs) == 0:
            return np.empty((0, 3))
        
        # Convert to metric coordinates
        zs = depth_image[ys, xs].astype(float) * self.camera.depth_scale
        
        # Filter reasonable depth range
        valid_mask = (zs > 0.1) & (zs < 10.0)
        xs, ys, zs = xs[valid_mask], ys[valid_mask], zs[valid_mask]
        
        if len(xs) == 0:
            return np.empty((0, 3))
        
        # Project to 3D
        fx, fy = self.camera.intrinsics.fx, self.camera.intrinsics.fy
        ppx, ppy = self.camera.intrinsics.ppx, self.camera.intrinsics.ppy
        
        X = (xs - ppx) * zs / fx
        Y = (ys - ppy) * zs / fy
        
        return np.column_stack([X, Y, zs])
    
    def evaluate_ground_plane(self, points, plane):
        """Evaluate ground plane detection results with stricter criteria"""
        if plane is None or len(points) == 0:
            return {
                'ground_coverage': 0.0,
                'obstacle_points': 0,
                'inliers': 0,
                'mean_height': 0.0,
                'height_std': 0.0,
                'plane_valid': False
            }
        
        # Check if plane is reasonable (not vertical, not upside down)
        A, B, C, D = plane
        normal = np.array([A, B, C])
        normal_mag = np.linalg.norm(normal)
        
        if normal_mag < 1e-6:
            return {
                'ground_coverage': 0.0,
                'obstacle_points': 0,
                'inliers': 0,
                'mean_height': 0.0,
                'height_std': 0.0,
                'plane_valid': False
            }
        
        normal = normal / normal_mag
        
        # Check if plane is roughly horizontal (Y component should be dominant)
        # Ground plane should have normal pointing up (positive Y)
        if abs(normal[1]) < 0.5 or normal[1] < 0:  # Not horizontal enough or pointing down
            return {
                'ground_coverage': 0.0,
                'obstacle_points': 0,
                'inliers': 0,
                'mean_height': 0.0,
                'height_std': 0.0,
                'plane_valid': False
            }
        
        # Calculate signed distances to plane
        distances = (points @ normal) + D
        
        # Filter points that are reasonable for ground plane detection
        # Ground should be below camera (positive Y distances when normal points up)
        valid_ground_mask = (
            (distances > -0.1) &  # Close to or above plane
            (distances < 0.1) &   # But not too far above
            (points[:, 1] > -2.0) & # Not too far below camera
            (points[:, 1] < 0.5)    # Not above camera level
        )
        
        ground_points = np.sum(valid_ground_mask)
        
        # Obstacle points (clearly above ground plane)
        obstacle_mask = (
            (distances > 0.1) & 
            (distances < 2.0) &     # Reasonable obstacle height
            (points[:, 1] > -1.5) & 
            (points[:, 1] < 1.5)
        )
        obstacle_points = np.sum(obstacle_mask)
        
        # Coverage metrics
        total_valid_points = len(points)
        ground_coverage = (ground_points / total_valid_points) * 100 if total_valid_points > 0 else 0
        
        # Additional validation: require minimum ground coverage for valid plane
        min_coverage_threshold = 10.0  # At least 10% ground coverage
        min_inlier_threshold = 200     # At least 200 inlier points
        
        plane_valid = (
            ground_coverage >= min_coverage_threshold and 
            ground_points >= min_inlier_threshold and
            abs(normal[1]) >= 0.7  # Strong vertical component
        )
        
        # Height statistics for ground points
        if ground_points > 0:
            ground_heights = distances[valid_ground_mask]
            mean_height = np.mean(ground_heights)
            height_std = np.std(ground_heights)
        else:
            mean_height = 0.0
            height_std = 0.0
        
        return {
            'ground_coverage': ground_coverage,
            'obstacle_points': obstacle_points,
            'inliers': ground_points,
            'mean_height': mean_height,
            'height_std': height_std,
            'plane_valid': plane_valid
        }
    
    def test_irls_algorithm(self, points):
        """Test IRLS algorithm with RealSense data"""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # IRLS parameters optimized for speed and RealSense data
        MAX_ITERATIONS = 5  # Reduced for speed
        TUKEY_C = 0.08
        MIN_POINTS = 100
        
        if len(points) < MIN_POINTS:
            return None, time.perf_counter() - start_time, {
                'iterations': 0,
                'converged': False,
                'inliers': 0
            }
        
        # Subsample points for speed if too many
        if len(points) > 2000:
            indices = np.random.choice(len(points), 2000, replace=False)
            points = points[indices]
        
        # Filter points for ground plane estimation (bottom 25%)
        y_threshold = np.percentile(points[:, 1], 25)
        candidate_points = points[points[:, 1] < y_threshold]
        
        if len(candidate_points) < MIN_POINTS:
            candidate_points = points
        
        N = len(candidate_points)
        w = np.ones(N, dtype=float)
        plane = None
        converged = False
        
        for iteration in range(MAX_ITERATIONS):
            w_sum = np.sum(w)
            if w_sum < 1e-6:
                break
            
            # Weighted centroid
            mu = (w.reshape(-1, 1) * candidate_points).sum(axis=0) / w_sum
            
            # Weighted covariance
            diffs = candidate_points - mu
            C = (w.reshape(-1, 1) * diffs).T @ diffs
            
            try:
                _, _, Vt = np.linalg.svd(C)
                n = Vt[-1, :]
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-6:
                    break
                n = n / norm_n
                
                # Ensure normal points up (positive Y)
                if n[1] < 0:
                    n = -n
                    
            except np.linalg.LinAlgError:
                break
            
            D = -np.dot(n, mu)
            r = (candidate_points @ n) + D
            
            # Update weights with Tukey biweight
            abs_r = np.abs(r)
            mask = abs_r < TUKEY_C
            w_new = np.zeros_like(w)
            w_new[mask] = (1 - (r[mask] / TUKEY_C) ** 2) ** 2
            
            # Check convergence
            if np.allclose(w, w_new, atol=1e-3):
                converged = True
                w = w_new
                break
                
            w = w_new
            plane = np.array([n[0], n[1], n[2], D], dtype=float)
        
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Evaluate results
        evaluation = self.evaluate_ground_plane(points, plane)
        
        # Only consider successful if plane is valid
        if not evaluation.get('plane_valid', False):
            plane = None
        
        metadata = {
            'iterations': iteration + 1 if 'iteration' in locals() else 0,
            'converged': converged and plane is not None,
            'inliers': evaluation['inliers'],
            'memory_used': end_memory - start_memory
        }
        
        return plane, end_time - start_time, metadata
    
    def test_ransac_algorithm(self, points):
        """Test RANSAC algorithm with RealSense data"""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # RANSAC parameters optimized for speed and RealSense
        MAX_ITERATIONS = 30  # Reduced for speed
        TOLERANCE = 0.08
        MIN_POINTS = 100
        
        if len(points) < MIN_POINTS:
            return None, time.perf_counter() - start_time, {
                'iterations': 0,
                'converged': False,
                'inliers': 0
            }
        
        # Subsample points for speed if too many
        if len(points) > 2000:
            indices = np.random.choice(len(points), 2000, replace=False)
            points = points[indices]
        
        # Filter potential ground points (bottom 30%)
        y_threshold = np.percentile(points[:, 1], 30)
        candidate_points = points[points[:, 1] < y_threshold]
        
        if len(candidate_points) < MIN_POINTS:
            candidate_points = points
        
        best_plane = None
        best_inliers = 0
        N = len(candidate_points)
        min_inliers_threshold = max(N * 0.1, 100)  # At least 10% or 100 points
        
        for iteration in range(MAX_ITERATIONS):
            # Random sample
            try:
                idx = np.random.choice(N, 3, replace=False)
                sample = candidate_points[idx]
            except ValueError:
                break
            
            # Compute plane
            v1 = sample[1] - sample[0]
            v2 = sample[2] - sample[0]
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            
            if norm_n < 1e-6:
                continue
            
            A, B, C = n / norm_n
            
            # Ensure normal points up
            if B < 0:
                A, B, C = -A, -B, -C
            
            D = -np.dot([A, B, C], sample[0])
            
            # Count inliers in all points
            dists = np.abs((points @ np.array([A, B, C])) + D)
            inliers = np.sum(dists < TOLERANCE)
            
            if inliers > best_inliers and inliers >= min_inliers_threshold:
                best_inliers = inliers
                best_plane = np.array([A, B, C, D], dtype=float)
        
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Evaluate results
        evaluation = self.evaluate_ground_plane(points, best_plane)
        
        # Only consider successful if plane is valid
        if not evaluation.get('plane_valid', False):
            best_plane = None
        
        metadata = {
            'iterations': MAX_ITERATIONS,
            'converged': best_plane is not None,
            'inliers': evaluation['inliers'],
            'memory_used': end_memory - start_memory
        }
        
        return best_plane, end_time - start_time, metadata
    
    def test_vdisparity_algorithm(self, points):
        """Test V-Disparity algorithm with RealSense data"""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # V-Disparity parameters
        BASELINE = 0.05
        MIN_POINTS = 50
        
        if len(points) < MIN_POINTS:
            return None, time.perf_counter() - start_time, {
                'iterations': 1,
                'converged': False,
                'inliers': 0
            }
        
        # Subsample points for speed
        if len(points) > 1500:
            indices = np.random.choice(len(points), 1500, replace=False)
            points = points[indices]
        
        # Use camera intrinsics
        fx = self.camera.intrinsics.fx
        
        # Extract coordinates
        Z = points[:, 2]
        Y = points[:, 1]
        
        # Filter valid depths
        valid_mask = (Z > 0.3) & (Z < 6.0) & (Y > -1.5) & (Y < 0.3)
        if not np.any(valid_mask):
            return None, time.perf_counter() - start_time, {
                'iterations': 1,
                'converged': False,
                'inliers': 0
            }
        
        valid_points = points[valid_mask]
        Z_valid = valid_points[:, 2]
        Y_valid = valid_points[:, 1]
        
        # Compute disparity
        with np.errstate(divide='ignore', invalid='ignore'):
            disparity = (fx * BASELINE) / Z_valid
            disparity = disparity[np.isfinite(disparity)]
        
        if len(disparity) < MIN_POINTS:
            return None, time.perf_counter() - start_time, {
                'iterations': 1,
                'converged': False,
                'inliers': 0
            }
        
        try:
            # Robust line fitting in disparity-height space
            valid_y = Y_valid[np.isfinite(disparity)]
            
            if len(valid_y) < MIN_POINTS:
                return None, time.perf_counter() - start_time, {
                    'iterations': 1,
                    'converged': False,
                    'inliers': 0
                }
            
            # Use simplified RANSAC for line fitting (faster)
            best_inliers = 0
            best_line = None
            
            for _ in range(15):  # Reduced iterations for speed
                if len(disparity) < 2:
                    break
                    
                # Sample two points
                idx = np.random.choice(len(disparity), 2, replace=False)
                d1, d2 = disparity[idx]
                y1, y2 = valid_y[idx]
                
                if abs(d2 - d1) < 1e-6:
                    continue
                
                # Fit line
                m = (y2 - y1) / (d2 - d1)
                c = y1 - m * d1
                
                # Count inliers
                predicted_y = m * disparity + c
                errors = np.abs(valid_y - predicted_y)
                inliers = np.sum(errors < 0.15)
                
                if inliers > best_inliers and inliers >= len(disparity) * 0.2:
                    best_inliers = inliers
                    best_line = (m, c)
            
            if best_line is None:
                return None, time.perf_counter() - start_time, {
                    'iterations': 1,
                    'converged': False,
                    'inliers': 0
                }
            
            # Convert to plane equation
            m, c = best_line
            # Simple conversion - this is approximate for v-disparity
            plane = np.array([0, 1, 0, -np.mean(Y_valid)], dtype=float)
            
        except Exception:
            return None, time.perf_counter() - start_time, {
                'iterations': 1,
                'converged': False,
                'inliers': 0
            }
        
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Evaluate results
        evaluation = self.evaluate_ground_plane(points, plane)
        
        # V-Disparity has different validation criteria
        if evaluation['inliers'] < 100:
            plane = None
        
        metadata = {
            'iterations': 1,
            'converged': plane is not None,
            'inliers': evaluation['inliers'],
            'memory_used': end_memory - start_memory
        }
        
        return plane, end_time - start_time, metadata
    
    def run_single_frame_test(self, frame_data, frame_id):
        """Test all algorithms on a single frame"""
        depth_image = frame_data['depth']
        timestamp = frame_data['timestamp']
        
        # Convert to point cloud
        points = self.depth_to_points(depth_image)
        
        if len(points) == 0:
            return {}
        
        results = {}
        
        for alg_name, alg_func in self.algorithms.items():
            # Measure CPU before test
            cpu_before = psutil.cpu_percent(interval=None)
            
            # Run algorithm
            plane, exec_time, metadata = alg_func(points)
            
            # Measure CPU after test
            cpu_after = psutil.cpu_percent(interval=None)
            cpu_usage = (cpu_before + cpu_after) / 2
            
            # Evaluate plane
            evaluation = self.evaluate_ground_plane(points, plane)
            
            # Create frame metrics
            metrics = FrameMetrics(
                frame_id=frame_id,
                timestamp=timestamp,
                algorithm=alg_name,
                processing_time_ms=exec_time * 1000,
                plane_found=plane is not None and evaluation.get('plane_valid', False),
                plane_equation=plane,
                num_points=len(points),
                num_inliers=evaluation['inliers'],
                convergence_iterations=metadata.get('iterations', 0),
                memory_usage_mb=metadata.get('memory_used', 0),
                cpu_percent=cpu_usage,
                ground_coverage_percent=evaluation['ground_coverage'],
                obstacle_points=evaluation['obstacle_points']
            )
            
            results[alg_name] = metrics
            
            # Update live metrics
            self.live_metrics[alg_name].append(metrics)
        
        return results
            
    
    def run_live_test_session(self, duration_seconds=60, target_fps=30, enable_live_plots=False):
        """Run live testing session with camera"""
        if not self.camera.pipeline:
            raise RuntimeError("Camera not initialized")
        
        print(f"Starting live test session: {duration_seconds}s @ {target_fps} FPS target")
        if not enable_live_plots:
            print("   Live plotting disabled for maximum speed")
        
        session_id = f"live_test_{int(time.time())}"
        start_time = time.time()
        frame_interval = 1.0 / target_fps
        
        session_results = {alg: [] for alg in self.algorithms}
        frame_count = 0
        last_progress_time = start_time
        
        # Real-time plotting setup (only if enabled)
        if enable_live_plots:
            plt.ion()
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Real-Time Algorithm Performance', fontsize=14)
        
        try:
            while (time.time() - start_time) < duration_seconds:
                loop_start = time.time()
                
                # Capture frame
                frame_data, frame_number = self.camera.capture_frame(timeout_ms=1000)
                if frame_data is None:
                    continue
                
                # Test algorithms
                frame_results = self.run_single_frame_test(frame_data, frame_count)
                
                # Store results
                for alg_name, metrics in frame_results.items():
                    session_results[alg_name].append(metrics)
                
                frame_count += 1
                
                # Update real-time plots less frequently for speed
                if enable_live_plots and frame_count % 20 == 0:
                    self.update_live_plots(axes)
                    plt.pause(0.01)
                
                # Progress update every 5 seconds
                current_time = time.time()
                if current_time - last_progress_time >= 5.0:
                    elapsed_time = current_time - start_time
                    fps = frame_count / elapsed_time
                    print(f"  Frame {frame_count}, {elapsed_time:.1f}s elapsed, {fps:.1f} FPS")
                    last_progress_time = current_time
                
                # Maintain target frame rate (but don't slow down unnecessarily)
                elapsed = time.time() - loop_start
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0.001:  # Only sleep if meaningful
                    time.sleep(sleep_time)
        
        except KeyboardInterrupt:
            print("\nTest session stopped by user")
        
        end_time = time.time()
        
        if enable_live_plots:
            plt.ioff()
            plt.close(fig)
        
        # Create test session object
        session = TestSession(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            total_frames=frame_count,
            camera_config=self.camera.get_camera_info(),
            test_scenarios=['live_capture'],
            algorithm_results=session_results
        )
        
        print(f"Test session completed: {frame_count} frames in {end_time - start_time:.1f}s")
        
        return session
    
    def update_live_plots(self, axes):
        """Update real-time performance plots"""
        if not self.live_metrics:
            return
        
        # Clear axes
        for ax in axes.flat:
            ax.clear()
        
        colors = {'IRLS': '#E74C3C', 'RANSAC': '#3498DB', 'V-Disparity': '#2ECC71'}
        
        # 1. Processing Time
        ax1 = axes[0, 0]
        for alg_name, metrics_deque in self.live_metrics.items():
            if metrics_deque:
                times = [m.processing_time_ms for m in metrics_deque]
                frames = list(range(len(times)))
                ax1.plot(frames, times, label=alg_name, color=colors[alg_name])
        ax1.set_title('Processing Time (ms)')
        ax1.set_ylabel('Time (ms)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Success Rate
        ax2 = axes[0, 1]
        success_rates = {}
        for alg_name, metrics_deque in self.live_metrics.items():
            if metrics_deque:
                successes = [m.plane_found for m in metrics_deque]
                success_rates[alg_name] = np.mean(successes) * 100
        
        if success_rates:
            bars = ax2.bar(success_rates.keys(), success_rates.values(), 
                          color=[colors[alg] for alg in success_rates.keys()])
            ax2.set_title('Success Rate (%)')
            ax2.set_ylabel('Success %')
            ax2.set_ylim(0, 100)
        
        # 3. Ground Coverage
        ax3 = axes[1, 0]
        for alg_name, metrics_deque in self.live_metrics.items():
            if metrics_deque:
                coverage = [m.ground_coverage_percent for m in metrics_deque if m.plane_found]
                if coverage:
                    frames = list(range(len(coverage)))
                    ax3.plot(frames, coverage, label=alg_name, color=colors[alg_name])
        ax3.set_title('Ground Coverage (%)')
        ax3.set_ylabel('Coverage %')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. FPS Performance
        ax4 = axes[1, 1]
        fps_data = {}
        for alg_name, metrics_deque in self.live_metrics.items():
            if metrics_deque:
                times = [m.processing_time_ms for m in metrics_deque]
                if times:
                    avg_time = np.mean(times)
                    fps_data[alg_name] = 1000 / avg_time if avg_time > 0 else 0
        
        if fps_data:
            bars = ax4.bar(fps_data.keys(), fps_data.values(),
                          color=[colors[alg] for alg in fps_data.keys()])
            ax4.set_title('Theoretical FPS')
            ax4.set_ylabel('FPS')
            ax4.axhline(y=30, color='red', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
    
    def analyze_session_results(self, session: TestSession):
        """Analyze and visualize session results"""
        print(f"\nAnalyzing session: {session.session_id}")
        print(f"Duration: {session.end_time - session.start_time:.1f}s")
        print(f"Total frames: {session.total_frames}")
        
        # Calculate summary statistics
        summary_stats = {}
        
        for alg_name, metrics_list in session.algorithm_results.items():
            if not metrics_list:
                continue
            
            # Extract data
            processing_times = [m.processing_time_ms for m in metrics_list]
            success_rate = np.mean([m.plane_found for m in metrics_list])
            ground_coverage = [m.ground_coverage_percent for m in metrics_list if m.plane_found]
            inliers = [m.num_inliers for m in metrics_list if m.plane_found]
            memory_usage = [m.memory_usage_mb for m in metrics_list]
            
            # Calculate statistics
            avg_time = np.mean(processing_times)
            std_time = np.std(processing_times)
            max_time = np.max(processing_times)
            theoretical_fps = 1000 / avg_time if avg_time > 0 else 0
            
            summary_stats[alg_name] = {
                'avg_processing_time_ms': avg_time,
                'std_processing_time_ms': std_time,
                'max_processing_time_ms': max_time,
                'theoretical_fps': theoretical_fps,
                'success_rate': success_rate,
                'avg_ground_coverage': np.mean(ground_coverage) if ground_coverage else 0,
                'avg_inliers': np.mean(inliers) if inliers else 0,
                'avg_memory_usage_mb': np.mean(memory_usage),
                'frame_drops': np.sum([t > 33.33 for t in processing_times]),  # >30 FPS
                'consistency_score': 1.0 / (std_time + 0.1)  # Lower std = higher consistency
            }
        
        return summary_stats
    
    def create_comprehensive_report(self, session: TestSession, summary_stats: Dict):
        """Create detailed performance report"""
        report = []
        report.append("=" * 80)
        report.append("INTEL REALSENSE LIVE ALGORITHM PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Session info
        duration = session.end_time - session.start_time
        actual_fps = session.total_frames / duration
        
        report.append("SESSION INFORMATION:")
        report.append(f"  Session ID: {session.session_id}")
        report.append(f"  Duration: {duration:.2f} seconds")
        report.append(f"  Total Frames Processed: {session.total_frames}")
        report.append(f"  Actual Capture FPS: {actual_fps:.2f}")
        report.append("")
        
        # Camera configuration
        camera_config = session.camera_config
        report.append("CAMERA CONFIGURATION:")
        report.append(f"  Resolution: {camera_config.get('width', 'N/A')}x{camera_config.get('height', 'N/A')}")
        report.append(f"  Target FPS: {camera_config.get('fps', 'N/A')}")
        report.append(f"  Focal Length: fx={camera_config.get('fx', 'N/A'):.1f}, fy={camera_config.get('fy', 'N/A'):.1f}")
        report.append(f"  Principal Point: ({camera_config.get('ppx', 'N/A'):.1f}, {camera_config.get('ppy', 'N/A'):.1f})")
        report.append(f"  Depth Scale: {camera_config.get('depth_scale', 'N/A')}")
        report.append("")
        
        # Algorithm performance comparison
        report.append("ALGORITHM PERFORMANCE SUMMARY:")
        report.append("-" * 50)
        
        # Sort algorithms by overall performance score
        def performance_score(stats):
            fps_score = min(stats['theoretical_fps'] / 30, 1.0) * 0.4
            success_score = stats['success_rate'] * 0.3
            consistency_score = min(stats['consistency_score'] / 10, 1.0) * 0.2
            coverage_score = stats['avg_ground_coverage'] / 100 * 0.1
            return fps_score + success_score + consistency_score + coverage_score
        
        sorted_algorithms = sorted(summary_stats.items(), 
                                 key=lambda x: performance_score(x[1]), 
                                 reverse=True)
        
        for rank, (alg_name, stats) in enumerate(sorted_algorithms, 1):
            report.append(f"{rank}. {alg_name.upper()} ALGORITHM")
            report.append(f"   Performance Score: {performance_score(stats):.3f}")
            report.append(f"   Average Processing Time: {stats['avg_processing_time_ms']:.2f} +/- {stats['std_processing_time_ms']:.2f} ms")
            report.append(f"   Theoretical Max FPS: {stats['theoretical_fps']:.1f}")
            report.append(f"   Success Rate: {stats['success_rate']:.1%}")
            report.append(f"   Ground Coverage: {stats['avg_ground_coverage']:.1f}%")
            report.append(f"   Frame Drops (>33ms): {stats['frame_drops']}")
            report.append(f"   Memory Usage: {stats['avg_memory_usage_mb']:.1f} MB")
            
            # Real-time assessment
            if stats['theoretical_fps'] >= 30 and stats['success_rate'] >= 0.9:
                assessment = "[EXCELLENT] for real-time applications"
            elif stats['theoretical_fps'] >= 20 and stats['success_rate'] >= 0.8:
                assessment = "[GOOD] for real-time applications"
            elif stats['theoretical_fps'] >= 15:
                assessment = "[ACCEPTABLE] for some real-time applications"
            else:
                assessment = "[NOT SUITABLE] for real-time applications"
            
            report.append(f"   Real-time Assessment: {assessment}")
            report.append("")
        
        # Detailed analysis
        report.append("DETAILED ANALYSIS:")
        report.append("-" * 30)
        report.append("")
        
        best_fps = max(summary_stats.items(), key=lambda x: x[1]['theoretical_fps'])
        best_success = max(summary_stats.items(), key=lambda x: x[1]['success_rate'])
        most_consistent = max(summary_stats.items(), key=lambda x: x[1]['consistency_score'])
        lowest_memory = min(summary_stats.items(), key=lambda x: x[1]['avg_memory_usage_mb'])
        
        report.append(f"Fastest Algorithm: {best_fps[0]} ({best_fps[1]['theoretical_fps']:.1f} FPS)")
        report.append(f"Most Reliable: {best_success[0]} ({best_success[1]['success_rate']:.1%} success)")
        report.append(f"Most Consistent: {most_consistent[0]} (consistency: {most_consistent[1]['consistency_score']:.2f})")
        report.append(f"Lowest Memory Usage: {lowest_memory[0]} ({lowest_memory[1]['avg_memory_usage_mb']:.1f} MB)")
        report.append("")
        
        # Environment-specific insights
        report.append("REALSENSE-SPECIFIC INSIGHTS:")
        report.append("-" * 40)
        report.append("")
        
        # Analyze noise characteristics
        avg_coverage = np.mean([stats['avg_ground_coverage'] for stats in summary_stats.values()])
        avg_inliers = np.mean([stats['avg_inliers'] for stats in summary_stats.values()])
        
        report.append(f"Environment Characteristics:")
        report.append(f"  Average Ground Coverage: {avg_coverage:.1f}%")
        report.append(f"  Average Inlier Points: {avg_inliers:.0f}")
        
        if avg_coverage < 30:
            report.append("  WARNING: Low ground coverage detected - may indicate complex environment")
        elif avg_coverage > 70:
            report.append("  GOOD: High ground coverage - good for ground plane detection")
        
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        report.append("-" * 20)
        report.append("")
        
        best_overall = sorted_algorithms[0][0]
        report.append(f"BEST OVERALL CHOICE: {best_overall}")
        report.append(f"   Recommended for balanced performance with RealSense cameras")
        report.append("")
        
        report.append("Specific Use Cases:")
        
        # Real-time robotics
        real_time_candidates = [alg for alg, stats in summary_stats.items() 
                               if stats['theoretical_fps'] >= 30 and stats['success_rate'] >= 0.8]
        if real_time_candidates:
            report.append(f"  Real-time Robotics: {', '.join(real_time_candidates)}")
        else:
            report.append("  Real-time Robotics: Consider optimizing parameters or reducing resolution")
        
        # High accuracy applications
        high_accuracy = max(summary_stats.items(), key=lambda x: x[1]['success_rate'])[0]
        report.append(f"  High Accuracy Applications: {high_accuracy}")
        
        # Resource-constrained systems
        low_resource = min(summary_stats.items(), 
                          key=lambda x: x[1]['avg_processing_time_ms'] + x[1]['avg_memory_usage_mb'])[0]
        report.append(f"  Resource-Constrained Systems: {low_resource}")
        
        report.append("")
        
        # Parameter tuning suggestions
        report.append("PARAMETER TUNING SUGGESTIONS:")
        report.append("-" * 40)
        report.append("")
        
        for alg_name, stats in summary_stats.items():
            report.append(f"{alg_name}:")
            
            if alg_name == 'IRLS':
                if stats['avg_processing_time_ms'] > 33:
                    report.append("  - Reduce max_iterations (try 6-8 for real-time)")
                if stats['success_rate'] < 0.8:
                    report.append("  - Adjust Tukey constant (try 0.06-0.12 range)")
                report.append("  - Consider filtering input points to bottom 25% by height")
                
            elif alg_name == 'RANSAC':
                if stats['avg_processing_time_ms'] > 33:
                    report.append("  - Reduce max_iterations (try 30-40 for real-time)")
                if stats['success_rate'] < 0.8:
                    report.append("  - Increase tolerance (try 0.08-0.12m for RealSense noise)")
                report.append("  - Pre-filter candidate points to improve convergence")
                
            elif alg_name == 'V-Disparity':
                if stats['success_rate'] < 0.8:
                    report.append("  - Verify stereo baseline parameter matches camera")
                    report.append("  - Check depth range filtering (0.2-8.0m recommended)")
                report.append("  - Ensure proper camera calibration for best results")
            
            report.append("")
        
        return "\n".join(report)

class RealSenseTestVisualization:
    """Create comprehensive visualizations for RealSense test results"""
    
    def __init__(self):
        self.colors = {'IRLS': '#E74C3C', 'RANSAC': '#3498DB', 'V-Disparity': '#2ECC71'}
    
    def create_performance_dashboard(self, session: TestSession, summary_stats: Dict):
        """Create comprehensive performance dashboard"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        algorithms = list(summary_stats.keys())
        
        # 1. Processing Time Distribution
        ax1 = fig.add_subplot(gs[0, :2])
        for alg in algorithms:
            metrics = session.algorithm_results[alg]
            times = [m.processing_time_ms for m in metrics]
            ax1.hist(times, bins=30, alpha=0.7, label=alg, color=self.colors[alg])
        
        ax1.axvline(x=33.33, color='red', linestyle='--', label='30 FPS limit')
        ax1.set_xlabel('Processing Time (ms)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Processing Time Distribution (RealSense Data)', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Real-time Performance
        ax2 = fig.add_subplot(gs[0, 2:])
        fps_data = [summary_stats[alg]['theoretical_fps'] for alg in algorithms]
        success_data = [summary_stats[alg]['success_rate'] * 100 for alg in algorithms]
        
        for i, alg in enumerate(algorithms):
            ax2.scatter(fps_data[i], success_data[i], s=200, 
                       color=self.colors[alg], label=alg, alpha=0.8)
            ax2.annotate(alg, (fps_data[i], success_data[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax2.axvline(x=30, color='red', linestyle='--', alpha=0.7, label='30 FPS target')
        ax2.axhline(y=90, color='green', linestyle='--', alpha=0.7, label='90% success target')
        ax2.set_xlabel('Theoretical FPS')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Real-time Suitability (RealSense)', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Performance over time
        ax3 = fig.add_subplot(gs[1, :2])
        window_size = 20  # Moving average window
        
        for alg in algorithms:
            metrics = session.algorithm_results[alg]
            times = [m.processing_time_ms for m in metrics]
            
            if len(times) > window_size:
                # Calculate moving average
                moving_avg = []
                for i in range(window_size, len(times)):
                    moving_avg.append(np.mean(times[i-window_size:i]))
                
                x_vals = list(range(window_size, len(times)))
                ax3.plot(x_vals, moving_avg, label=f'{alg} (moving avg)', 
                        color=self.colors[alg], linewidth=2)
        
        ax3.axhline(y=33.33, color='red', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Frame Number')
        ax3.set_ylabel('Processing Time (ms)')
        ax3.set_title('Performance Stability Over Time', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Ground Coverage Analysis
        ax4 = fig.add_subplot(gs[1, 2:])
        coverage_data = []
        labels = []
        
        for alg in algorithms:
            metrics = session.algorithm_results[alg]
            coverage = [m.ground_coverage_percent for m in metrics if m.plane_found]
            if coverage:
                coverage_data.append(coverage)
                labels.append(alg)
        
        if coverage_data:
            bp = ax4.boxplot(coverage_data, labels=labels, patch_artist=True)
            for patch, alg in zip(bp['boxes'], labels):
                patch.set_facecolor(self.colors[alg])
                patch.set_alpha(0.7)
        
        ax4.set_ylabel('Ground Coverage (%)')
        ax4.set_title('Ground Coverage Distribution', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. Memory and CPU Usage
        ax5 = fig.add_subplot(gs[2, :2])
        memory_data = [summary_stats[alg]['avg_memory_usage_mb'] for alg in algorithms]
        
        bars = ax5.bar(algorithms, memory_data, color=[self.colors[alg] for alg in algorithms])
        ax5.set_ylabel('Memory Usage (MB)')
        ax5.set_title('Average Memory Usage', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, memory_data):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # 6. Success Rate and Consistency
        ax6 = fig.add_subplot(gs[2, 2:])
        success_rates = [summary_stats[alg]['success_rate'] * 100 for alg in algorithms]
        consistency_scores = [summary_stats[alg]['consistency_score'] for alg in algorithms]
        
        # Normalize consistency scores for better visualization
        max_consistency = max(consistency_scores)
        norm_consistency = [(c/max_consistency) * 100 for c in consistency_scores]
        
        x = np.arange(len(algorithms))
        width = 0.35
        
        bars1 = ax6.bar(x - width/2, success_rates, width, label='Success Rate (%)', 
                       color=[self.colors[alg] for alg in algorithms], alpha=0.8)
        bars2 = ax6.bar(x + width/2, norm_consistency, width, label='Consistency (normalized)', 
                       color=[self.colors[alg] for alg in algorithms], alpha=0.5)
        
        ax6.set_xlabel('Algorithm')
        ax6.set_ylabel('Score (%)')
        ax6.set_title('Success Rate vs Consistency', fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels(algorithms)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Intel RealSense Algorithm Performance Dashboard', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        return fig
    
    def create_frame_analysis_plot(self, session: TestSession):
        """Create frame-by-frame analysis visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Frame-by-Frame Analysis (RealSense Data)', fontsize=14, fontweight='bold')
        
        algorithms = list(session.algorithm_results.keys())
        
        # 1. Processing time over frames
        ax1 = axes[0, 0]
        for alg in algorithms:
            metrics = session.algorithm_results[alg]
            frame_ids = [m.frame_id for m in metrics]
            times = [m.processing_time_ms for m in metrics]
            ax1.plot(frame_ids, times, label=alg, color=self.colors[alg], alpha=0.7)
        
        ax1.axhline(y=33.33, color='red', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Frame ID')
        ax1.set_ylabel('Processing Time (ms)')
        ax1.set_title('Processing Time per Frame')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Point cloud size variation
        ax2 = axes[0, 1]
        for alg in algorithms:
            metrics = session.algorithm_results[alg]
            frame_ids = [m.frame_id for m in metrics]
            point_counts = [m.num_points for m in metrics]
            ax2.plot(frame_ids, point_counts, label=alg, color=self.colors[alg], alpha=0.7)
        
        ax2.set_xlabel('Frame ID')
        ax2.set_ylabel('Number of Points')
        ax2.set_title('Point Cloud Size Variation')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Success rate over time (sliding window)
        ax3 = axes[1, 0]
        window_size = 10
        
        for alg in algorithms:
            metrics = session.algorithm_results[alg]
            if len(metrics) > window_size:
                success_window = []
                frame_window = []
                
                for i in range(window_size, len(metrics)):
                    window_metrics = metrics[i-window_size:i]
                    success_rate = np.mean([m.plane_found for m in window_metrics])
                    success_window.append(success_rate * 100)
                    frame_window.append(metrics[i].frame_id)
                
                ax3.plot(frame_window, success_window, label=alg, 
                        color=self.colors[alg], linewidth=2)
        
        ax3.set_xlabel('Frame ID')
        ax3.set_ylabel('Success Rate (%) - 10 frame window')
        ax3.set_title('Success Rate Stability')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 100)
        
        # 4. Inlier count distribution
        ax4 = axes[1, 1]
        for alg in algorithms:
            metrics = session.algorithm_results[alg]
            inliers = [m.num_inliers for m in metrics if m.plane_found]
            if inliers:
                ax4.hist(inliers, bins=20, alpha=0.6, label=alg, 
                        color=self.colors[alg], density=True)
        
        ax4.set_xlabel('Number of Inliers')
        ax4.set_ylabel('Density')
        ax4.set_title('Inlier Count Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

def main():
    """Main function to run RealSense live testing"""
    parser = argparse.ArgumentParser(description='Intel RealSense Algorithm Performance Test')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Test duration in seconds (default: 60)')
    parser.add_argument('--fps', type=int, default=30, 
                       help='Target FPS (default: 30)')
    parser.add_argument('--width', type=int, default=640, 
                       help='Image width (default: 640)')
    parser.add_argument('--height', type=int, default=480, 
                       help='Image height (default: 480)')
    parser.add_argument('--no-rgb', action='store_true', 
                       help='Disable RGB stream (depth only)')
    parser.add_argument('--record', type=str, 
                       help='Record session to file (e.g., --record session.npz)')
    parser.add_argument('--load', type=str, 
                       help='Load recorded session from file')
    parser.add_argument('--output-dir', type=str, default='realsense_test_results',
                       help='Output directory for results')
    parser.add_argument('--no-live-plots', action='store_true',
                       help='Disable live plotting for maximum speed')
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode: reduced resolution and iterations for speed')
    
    args = parser.parse_args()
    
    # Check RealSense availability
    if not REALSENSE_AVAILABLE and not args.load:
        print("PyRealSense2 not available and no recorded data specified")
        print("Install with: pip install pyrealsense2")
        print("Or use --load to analyze recorded data")
        return
    
    print("=" * 80)
    print("INTEL REALSENSE LIVE ALGORITHM PERFORMANCE TEST")
    print("=" * 80)
    print("")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Apply fast mode settings
    if args.fast:
        if args.width == 640:  # Default resolution
            args.width = 424
            args.height = 240
        print(f"[FAST MODE] Using {args.width}x{args.height} for speed")
    
    # Initialize camera capture
    camera = RealSenseDataCapture(
        width=args.width, 
        height=args.height, 
        fps=args.fps,
        enable_rgb=not args.no_rgb
    )
    
    # Initialize tester
    tester = RealTimeAlgorithmTester(camera)
    
    try:
        if args.load:
            # Load recorded session
            print(f"Loading recorded session: {args.load}")
            if not camera.load_recording(args.load):
                return
            
            # Process recorded frames
            print("Processing recorded frames...")
            session_results = {alg: [] for alg in tester.algorithms}
            
            for frame_id, frame_data in enumerate(camera.recorded_frames):
                frame_results = tester.run_single_frame_test(frame_data, frame_id)
                for alg_name, metrics in frame_results.items():
                    session_results[alg_name].append(metrics)
                
                if (frame_id + 1) % 50 == 0:
                    print(f"  Processed {frame_id + 1}/{len(camera.recorded_frames)} frames")
            
            # Create session object
            session = TestSession(
                session_id=f"recorded_{int(time.time())}",
                start_time=camera.recorded_frames[0]['timestamp'],
                end_time=camera.recorded_frames[-1]['timestamp'],
                total_frames=len(camera.recorded_frames),
                camera_config=camera.get_camera_info(),
                test_scenarios=['recorded_playback'],
                algorithm_results=session_results
            )
            
        else:
            # Live camera testing
            if not camera.initialize_camera():
                print("Failed to initialize camera")
                return
            
            print(f"Camera initialized successfully")
            print(f"   Resolution: {args.width}x{args.height}")
            print(f"   Target FPS: {args.fps}")
            print(f"   Test Duration: {args.duration}s")
            print("")
            
            if args.record:
                camera.start_recording()
            
            # Run live test session
            session = tester.run_live_test_session(
                duration_seconds=args.duration,
                target_fps=args.fps,
                enable_live_plots=not args.no_live_plots
            )
            
            if args.record:
                recorded_frames = camera.stop_recording()
                camera.save_recording(args.record)
        
        # Analyze results
        print("\nAnalyzing results...")
        summary_stats = tester.analyze_session_results(session)
        
        # Create visualizations
        print("Creating visualizations...")
        visualizer = RealSenseTestVisualization()
        
        # Performance dashboard
        dashboard_fig = visualizer.create_performance_dashboard(session, summary_stats)
        dashboard_fig.savefig(output_dir / 'performance_dashboard.png', 
                             dpi=300, bbox_inches='tight')
        
        # Frame analysis
        frame_fig = visualizer.create_frame_analysis_plot(session)
        frame_fig.savefig(output_dir / 'frame_analysis.png', 
                         dpi=300, bbox_inches='tight')
        
        # Generate comprehensive report
        print("Generating report...")
        report = tester.create_comprehensive_report(session, summary_stats)
        
        # Save results
        report_file = output_dir / 'realsense_performance_report.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Save session data
        session_file = output_dir / 'session_data.json'
        session_dict = asdict(session)
        # Convert numpy arrays to lists for JSON serialization
        for alg_results in session_dict['algorithm_results'].values():
            for metrics in alg_results:
                if metrics['plane_equation'] is not None:
                    metrics['plane_equation'] = metrics['plane_equation'].tolist()
        
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(session_dict, f, indent=2, default=str)
        
        # Save summary statistics
        stats_file = output_dir / 'summary_statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(summary_stats, f, indent=2)
        
        # Print summary
        print("\n" + "="*80)
        print("TEST RESULTS SUMMARY")
        print("="*80)
        
        for alg_name, stats in summary_stats.items():
            print(f"\n{alg_name}:")
            print(f"  Theoretical FPS: {stats['theoretical_fps']:.1f}")
            print(f"  Success Rate: {stats['success_rate']:.1%}")
            print(f"  Avg Processing Time: {stats['avg_processing_time_ms']:.2f} ms")
            print(f"  Frame Drops: {stats['frame_drops']}")
            print(f"  Ground Coverage: {stats['avg_ground_coverage']:.1f}%")
        
        # Best algorithm recommendation
        best_alg = max(summary_stats.items(), 
                      key=lambda x: x[1]['theoretical_fps'] * x[1]['success_rate'])
        
        print(f"\n[WINNER] RECOMMENDED: {best_alg[0]}")
        print(f"   Best balance of speed ({best_alg[1]['theoretical_fps']:.1f} FPS) and reliability ({best_alg[1]['success_rate']:.1%})")
        
        print(f"\n[SUCCESS] Results saved to: {output_dir}")
        print("[CHART] performance_dashboard.png - Comprehensive performance analysis")
        print("[CHART] frame_analysis.png - Frame-by-frame detailed analysis")
        print("[REPORT] realsense_performance_report.txt - Detailed text report")
        print("[DATA] session_data.json - Complete session data")
        print("[STATS] summary_statistics.json - Summary statistics")
        
        # Show plots
        plt.show()
        
    except KeyboardInterrupt:
        print("\n[STOP] Test interrupted by user")
    
    except Exception as e:
        print(f"[ERROR] Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        camera.cleanup()
        print("[CLEANUP] Cleanup completed")

if __name__ == "__main__":
    main()