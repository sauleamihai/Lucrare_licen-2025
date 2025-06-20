#!/usr/bin/env python3
"""
Fast Hybrid Plane Detector cu Filtrare Temporală și IMU Calibration
Rezolvă problema persistenței obstacolelor și îmbunătățește detectarea planului cu IMU
"""

import math
import argparse
import sys
import time
import numpy as np
import pyrealsense2 as rs
import cv2
from collections import deque
from scipy.spatial.transform import Rotation

# ───────── Configurable Params ─────────
NUM_FRAMES = 500   # Default; overridden via CLI
GRID_R, GRID_A = 3, 16
MIN_GROUND_POINTS = 50
GROUND_EPS, MAX_H = 0.02, 1.9
RADIAL_EDGES = np.array([0.0, 0.5, 1.0, 4.5])

# Temporal filtering params
TEMPORAL_DECAY = 0.85        # Decay factor pentru obstacole (0.85 = scade cu 15% per frame)
MIN_PERSISTENCE = 3          # Frame-uri minime pentru a considera un obstacol valid
PRESENCE_THRESHOLD = 50      # Threshold minim pentru a considera că un obstacol există
RAPID_DECAY = 0.5           # Decay rapid pentru zone fără detectări noi

# Fast Hybrid specific params
FAST_SAMPLE_SIZE = 2000
HOUGH_RESOLUTION = 0.03
PROSAC_ITERATIONS = 25
STABILITY_THRESHOLD = 0.15

# IMU calibration params
IMU_CALIBRATION_FRAMES = 100    # Frame-uri pentru calibrarea IMU
IMU_FILTER_ALPHA = 0.98        # Complementary filter alpha
IMU_GYRO_WEIGHT = 0.02         # Weightul gyroscopului în filtru
IMU_GRAVITY_THRESHOLD = 0.1    # Threshold pentru stabilitatea gravitației
IMU_UPDATE_RATE = 0.95         # Rate pentru actualizarea orientării
MAX_TILT_ANGLE = 30.0          # Unghi maxim de înclinare (grade)

# ───────── Command‐Line Parsing ─────────
parser = argparse.ArgumentParser()
parser.add_argument("--frames", type=int, default=NUM_FRAMES,
                    help="Number of depth frames to process then exit.")
parser.add_argument("--calibrate-imu", action="store_true",
                    help="Perform IMU calibration at startup")
parser.add_argument("--use-imu", action="store_true", default=True,
                    help="Use IMU for enhanced plane detection")
args = parser.parse_args()
NUM_FRAMES = args.frames

# ═══════════════════════════════════════════════════════════════════════════
# IMU CALIBRATION AND ORIENTATION FILTER
# ═══════════════════════════════════════════════════════════════════════════

class IMUCalibration:
    """
    Calibrează și filtrează datele IMU pentru detectarea planului de sol
    """
    
    def __init__(self, enable_imu=True):
        self.enable_imu = enable_imu
        self.is_calibrated = False
        self.calibration_complete = False
        
        # Calibration data
        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.gravity_vector = np.array([0, 0, -1])  # Default down
        
        # Filter state
        self.orientation = np.eye(3)  # Rotation matrix
        self.gravity_world = np.array([0, 0, -1])
        self.last_timestamp = None
        
        # Calibration buffers
        self.accel_buffer = deque(maxlen=IMU_CALIBRATION_FRAMES)
        self.gyro_buffer = deque(maxlen=IMU_CALIBRATION_FRAMES)
        
        # Stability tracking
        self.gravity_stability = deque(maxlen=20)
        self.orientation_confidence = 0.0
        
        print("IMU Calibration initialized" + (" (enabled)" if enable_imu else " (disabled)"))
    
    def add_calibration_sample(self, accel, gyro):
        """Adaugă sample pentru calibrare"""
        if not self.enable_imu or self.calibration_complete:
            return
            
        self.accel_buffer.append(accel.copy())
        self.gyro_buffer.append(gyro.copy())
        
        if len(self.accel_buffer) >= IMU_CALIBRATION_FRAMES:
            self._compute_calibration()
    
    def _compute_calibration(self):
        """Calculează parametrii de calibrare"""
        if len(self.accel_buffer) < IMU_CALIBRATION_FRAMES:
            return
        
        # Compute biases
        accel_samples = np.array(self.accel_buffer)
        gyro_samples = np.array(self.gyro_buffer)
        
        # Gyro bias (should be zero when stationary)
        self.gyro_bias = np.mean(gyro_samples, axis=0)
        
        # Accel bias and gravity estimation
        accel_mean = np.mean(accel_samples, axis=0)
        accel_std = np.std(accel_samples, axis=0)
        
        # Check if device was stationary during calibration
        if np.max(accel_std) < 0.5:  # Low variance indicates stationary
            # Device was stationary, use mean as gravity reference
            gravity_magnitude = np.linalg.norm(accel_mean)
            
            if 8.0 < gravity_magnitude < 12.0:  # Reasonable gravity range
                self.gravity_vector = accel_mean / gravity_magnitude
                self.accel_bias = accel_mean - self.gravity_vector * 9.81
                self.is_calibrated = True
                
                print(f"IMU Calibration successful!")
                print(f"  Gravity vector: {self.gravity_vector}")
                print(f"  Gravity magnitude: {gravity_magnitude:.2f} m/s²")
                print(f"  Gyro bias: {self.gyro_bias}")
            else:
                print(f"Warning: Unusual gravity magnitude: {gravity_magnitude:.2f}")
        else:
            print(f"Warning: Device not stationary during calibration (std: {accel_std})")
        
        self.calibration_complete = True
    
    def update_orientation(self, accel, gyro, timestamp):
        """Actualizează orientarea cu complementary filter"""
        if not self.enable_imu or not self.is_calibrated:
            return False
        
        # Correct for biases
        accel_corrected = accel - self.accel_bias
        gyro_corrected = gyro - self.gyro_bias
        
        # Time delta
        if self.last_timestamp is not None:
            dt = (timestamp - self.last_timestamp) / 1000.0  # Convert to seconds
            dt = max(0.001, min(dt, 0.1))  # Clamp dt to reasonable range
        else:
            dt = 0.033  # Assume ~30 FPS
        
        self.last_timestamp = timestamp
        
        # Gyroscope integration (predict)
        if np.linalg.norm(gyro_corrected) > 0.01:  # Only if significant rotation
            gyro_magnitude = np.linalg.norm(gyro_corrected)
            gyro_axis = gyro_corrected / gyro_magnitude
            angle = gyro_magnitude * dt
            
            # Create rotation matrix from axis-angle
            rotation_delta = Rotation.from_rotvec(gyro_axis * angle).as_matrix()
            self.orientation = self.orientation @ rotation_delta
        
        # Accelerometer correction (update)
        accel_magnitude = np.linalg.norm(accel_corrected)
        
        if 8.0 < accel_magnitude < 12.0:  # Valid gravity reading
            # Normalize accelerometer reading
            accel_normalized = accel_corrected / accel_magnitude
            
            # Current gravity direction in body frame
            gravity_body_predicted = self.orientation.T @ self.gravity_world
            
            # Correction vector (cross product gives rotation axis)
            correction_axis = np.cross(gravity_body_predicted, accel_normalized)
            correction_magnitude = np.linalg.norm(correction_axis)
            
            if correction_magnitude > 0.01:  # Significant correction needed
                correction_axis = correction_axis / correction_magnitude
                correction_angle = np.arcsin(min(correction_magnitude, 1.0))
                
                # Apply correction with complementary filter
                correction_angle *= (1.0 - IMU_FILTER_ALPHA)
                
                if correction_angle > 0.001:  # Apply only significant corrections
                    correction_rotation = Rotation.from_rotvec(
                        correction_axis * correction_angle).as_matrix()
                    self.orientation = correction_rotation @ self.orientation
            
            # Update stability metric
            stability = 1.0 - correction_magnitude
            self.gravity_stability.append(stability)
            self.orientation_confidence = np.mean(self.gravity_stability)
        
        # Normalize orientation matrix to prevent drift
        U, _, Vt = np.linalg.svd(self.orientation)
        self.orientation = U @ Vt
        
        return True
    
    def get_ground_plane_estimate(self):
        """Returnează estimarea planului de sol bazată pe IMU"""
        if not self.enable_imu or not self.is_calibrated:
            return None
        
        # Transform gravity to world coordinates
        gravity_world_current = self.orientation @ self.gravity_vector
        
        # Ground plane normal is opposite to gravity
        normal = -gravity_world_current
        normal = normal / np.linalg.norm(normal)
        
        # Plane equation: normal · (x - point) = 0
        # Assume ground is at y = 0 (can be adjusted)
        d = 0.0  # Distance from origin
        
        return np.array([normal[0], normal[1], normal[2], d])
    
    def get_orientation_confidence(self):
        """Returnează confidence-ul orientării (0-1)"""
        if not self.enable_imu or not self.is_calibrated:
            return 0.0
        return self.orientation_confidence
    
    def get_tilt_angle(self):
        """Returnează unghiul de înclinare în grade"""
        if not self.enable_imu or not self.is_calibrated:
            return 0.0
        
        # Current up direction
        up_current = self.orientation @ np.array([0, 1, 0])
        up_reference = np.array([0, 1, 0])
        
        # Angle between current and reference up
        cos_angle = np.dot(up_current, up_reference)
        angle_rad = np.arccos(np.clip(cos_angle, -1, 1))
        return np.degrees(angle_rad)
    
    def is_ready(self):
        """Verifică dacă IMU este gata pentru utilizare"""
        return self.enable_imu and self.is_calibrated and self.orientation_confidence > 0.7

# ═══════════════════════════════════════════════════════════════════════════
# TEMPORAL OBSTACLE FILTER (Enhanced with IMU)
# ═══════════════════════════════════════════════════════════════════════════

class TemporalObstacleFilter:
    """
    Filtrează temporally obstacolele pentru a elimina persistența falsă
    Enhanced cu informații IMU
    """
    
    def __init__(self, grid_shape=(GRID_R, GRID_A)):
        self.grid_shape = grid_shape
        self.accumulator = np.zeros(grid_shape, dtype=float)
        self.presence_map = np.zeros(grid_shape, dtype=int)
        self.last_detection = np.zeros(grid_shape, dtype=int)
        self.frame_count = 0
        
        # IMU-enhanced features
        self.tilt_compensation = True
        self.dynamic_threshold = True
        
    def update(self, current_histogram, imu_confidence=0.0, tilt_angle=0.0):
        """
        Actualizează filtrul temporal cu histograma curentă și info IMU
        """
        self.frame_count += 1
        
        # Adjust processing based on IMU data
        if self.tilt_compensation and tilt_angle > 15.0:
            # Reduce sensitivity when tilted
            current_histogram = current_histogram * 0.8
        
        # Dynamic threshold based on IMU confidence
        if self.dynamic_threshold and imu_confidence > 0.8:
            # Higher confidence allows lower thresholds
            presence_threshold = PRESENCE_THRESHOLD * 0.8
        else:
            presence_threshold = PRESENCE_THRESHOLD
        
        # Convert to float for calculations
        current_float = current_histogram.astype(float)
        
        # Update presence tracking
        has_detection = current_float > presence_threshold
        self.presence_map[has_detection] += 1
        self.presence_map[~has_detection] = 0
        self.last_detection[has_detection] = self.frame_count
        
        # Apply temporal decay
        self._apply_temporal_decay()
        
        # Update accumulator
        self._update_accumulator(current_float)
        
        # Generate filtered output
        return self._generate_filtered_output()
    
    def _apply_temporal_decay(self):
        """Aplică decay temporal pe accumulator"""
        frames_since_detection = self.frame_count - self.last_detection
        
        # Normal decay for recent detections
        recent_mask = frames_since_detection <= 5
        self.accumulator[recent_mask] *= TEMPORAL_DECAY
        
        # Rapid decay for old detections
        old_mask = frames_since_detection > 5
        self.accumulator[old_mask] *= RAPID_DECAY
        
        # Complete elimination for very old detections
        very_old_mask = frames_since_detection > 15
        self.accumulator[very_old_mask] = 0
    
    def _update_accumulator(self, current_detection):
        """Actualizează accumulator-ul cu detectările curente"""
        self.accumulator += current_detection
        max_val = 50000
        self.accumulator = np.clip(self.accumulator, 0, max_val)
    
    def _generate_filtered_output(self):
        """Generează output-ul filtrat final"""
        filtered = self.accumulator.copy()
        
        # Remove noise
        noise_threshold = 10
        filtered[filtered < noise_threshold] = 0
        
        # Spatial smoothing
        filtered = self._spatial_smoothing(filtered)
        
        return np.round(filtered).astype(int)
    
    def _spatial_smoothing(self, matrix):
        """Smoothing spatial pentru consistență"""
        smoothed = matrix.copy()
        
        for r in range(matrix.shape[0]):
            row = matrix[r, :]
            padded_row = np.concatenate([row[-2:], row, row[:2]])
            
            for i in range(len(row)):
                window = padded_row[i:i+5]
                smoothed[r, i] = np.median(window)
        
        return smoothed
    
    def get_debug_info(self):
        """Returnează informații de debug"""
        return {
            'frame_count': self.frame_count,
            'accumulator_max': np.max(self.accumulator),
            'accumulator_mean': np.mean(self.accumulator),
            'active_cells': np.sum(self.accumulator > PRESENCE_THRESHOLD)
        }

# ═══════════════════════════════════════════════════════════════════════════
# FAST HYBRID PLANE DETECTOR (Enhanced with IMU)
# ═══════════════════════════════════════════════════════════════════════════

class FastHybridPlaneDetector:
    """Versiune optimizată cu IMU integration"""
    
    def __init__(self, imu_calibration=None):
        self.imu_calibration = imu_calibration
        self.last_plane = None
        self.plane_history = []
        self.confidence_history = []
        self.method_used = "none"
        self.imu_plane_weight = 0.3  # Weight for IMU plane estimate
        
    def detect_plane(self, points):
        """Detectare hibridă optimizată cu IMU"""
        if len(points) < MIN_GROUND_POINTS:
            return None
        
        # Get IMU plane estimate if available
        imu_plane = None
        imu_confidence = 0.0
        
        if self.imu_calibration and self.imu_calibration.is_ready():
            imu_plane = self.imu_calibration.get_ground_plane_estimate()
            imu_confidence = self.imu_calibration.get_orientation_confidence()
        
        # Smart subsample
        sampled_points = self._smart_subsample(points)
        
        # Strategy 1: IMU-guided refinement
        if imu_plane is not None and imu_confidence > 0.8:
            imu_refined_plane = self._imu_guided_refinement(sampled_points, imu_plane)
            if imu_refined_plane is not None:
                self.method_used = "imu_guided"
                self._update_history(imu_refined_plane)
                return imu_refined_plane
        
        # Strategy 2: Quick refinement (existing)
        if self.last_plane is not None:
            refined_plane = self._quick_refinement(sampled_points)
            if refined_plane is not None:
                # Blend with IMU if available
                if imu_plane is not None and imu_confidence > 0.5:
                    blended_plane = self._blend_planes(refined_plane, imu_plane, imu_confidence)
                    self.method_used = "refinement_imu"
                    self._update_history(blended_plane)
                    return blended_plane
                else:
                    self.method_used = "refinement"
                    self._update_history(refined_plane)
                    return refined_plane
        
        # Strategy 3: Hough 3D with IMU bias
        if imu_plane is not None:
            hough_plane = self._imu_biased_hough(sampled_points, imu_plane)
        else:
            hough_plane = self._hough_3d_detect(sampled_points)
        
        if hough_plane is not None:
            confidence = self._compute_confidence(points, hough_plane)
            if confidence > 0.6:
                self.method_used = "hough_imu" if imu_plane is not None else "hough"
                self._update_history(hough_plane)
                return hough_plane
        
        # Strategy 4: PROSAC fallback
        prosac_plane = self._prosac_detect(sampled_points)
        if prosac_plane is not None:
            self.method_used = "prosac"
            self._update_history(prosac_plane)
            return prosac_plane
        
        # Strategy 5: Pure IMU fallback
        if imu_plane is not None and imu_confidence > 0.6:
            self.method_used = "imu_only"
            self._update_history(imu_plane)
            return imu_plane
        
        # Final fallback to history
        if len(self.plane_history) > 0:
            self.method_used = "temporal"
            return self.plane_history[-1]
        
        return None
    
    def _imu_guided_refinement(self, points, imu_plane):
        """Refinement ghidat de IMU"""
        # Use IMU plane as initial estimate
        distances = np.abs(points @ imu_plane[:3] + imu_plane[3])
        inlier_mask = distances < 0.15  # Slightly more tolerance for IMU
        inliers = points[inlier_mask]
        
        if len(inliers) < 20:
            return None
        
        # Least squares refinement
        centroid = np.mean(inliers, axis=0)
        centered = inliers - centroid
        
        if len(centered) > 200:
            idx = np.random.choice(len(centered), 200, replace=False)
            centered = centered[idx]
        
        try:
            _, _, V = np.linalg.svd(centered)
            normal = V[-1]
            
            # Ensure consistency with IMU normal direction
            if np.dot(normal, imu_plane[:3]) < 0:
                normal = -normal
            
            # Blend with IMU normal (70% refined, 30% IMU)
            blended_normal = 0.7 * normal + 0.3 * imu_plane[:3]
            blended_normal = blended_normal / np.linalg.norm(blended_normal)
            
            D = -np.dot(blended_normal, centroid)
            return np.array([blended_normal[0], blended_normal[1], blended_normal[2], D])
            
        except:
            return imu_plane
    
    def _blend_planes(self, plane1, plane2, imu_weight):
        """Blend două plane estimates"""
        # Normalize normals
        n1 = plane1[:3] / np.linalg.norm(plane1[:3])
        n2 = plane2[:3] / np.linalg.norm(plane2[:3])
        
        # Ensure same orientation
        if np.dot(n1, n2) < 0:
            n2 = -n2
            plane2 = np.array([-n2[0], -n2[1], -n2[2], -plane2[3]])
        
        # Weighted blend
        weight = min(imu_weight, 0.5)  # Limit IMU influence
        blended_normal = (1 - weight) * n1 + weight * n2
        blended_normal = blended_normal / np.linalg.norm(blended_normal)
        
        blended_d = (1 - weight) * plane1[3] + weight * plane2[3]
        
        return np.array([blended_normal[0], blended_normal[1], blended_normal[2], blended_d])
    
    def _imu_biased_hough(self, points, imu_plane):
        """Hough 3D cu bias către IMU plane"""
        if len(points) < 50:
            return None
        
        # Subsample for speed
        if len(points) > 400:
            idx = np.random.choice(len(points), 400, replace=False)
            hough_points = points[idx]
        else:
            hough_points = points
        
        # Bias theta and phi ranges around IMU normal
        imu_normal = imu_plane[:3]
        
        # Convert IMU normal to spherical coordinates
        imu_theta = np.arccos(np.clip(imu_normal[2], -1, 1))
        imu_phi = np.arctan2(imu_normal[1], imu_normal[0])
        
        # Create biased ranges
        theta_range = 0.3  # ±0.3 radians around IMU
        phi_range = 0.5    # ±0.5 radians around IMU
        
        theta_bins = np.linspace(max(0, imu_theta - theta_range), 
                                min(np.pi, imu_theta + theta_range), 10)
        phi_bins = np.linspace(imu_phi - phi_range, imu_phi + phi_range, 10)
        
        # Regular Hough voting with biased ranges
        max_dist = np.max(np.linalg.norm(hough_points, axis=1))
        rho_bins = np.arange(-max_dist, max_dist, HOUGH_RESOLUTION * 2)
        
        accumulator = np.zeros((len(rho_bins), len(theta_bins), len(phi_bins)))
        
        for point in hough_points[::2]:  # Skip for speed
            x, y, z = point
            
            for i, theta in enumerate(theta_bins):
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                
                for j, phi in enumerate(phi_bins):
                    nx = sin_theta * np.cos(phi)
                    ny = sin_theta * np.sin(phi)
                    nz = cos_theta
                    
                    rho = x * nx + y * ny + z * nz
                    rho_idx = np.searchsorted(rho_bins, rho)
                    
                    if 0 <= rho_idx < len(rho_bins):
                        accumulator[rho_idx, i, j] += 1
        
        max_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
        
        if accumulator[max_idx] < 5:
            return None
        
        rho_idx, theta_idx, phi_idx = max_idx
        rho = rho_bins[rho_idx] if rho_idx < len(rho_bins) else 0
        theta = theta_bins[theta_idx] if theta_idx < len(theta_bins) else 0
        phi = phi_bins[phi_idx] if phi_idx < len(phi_bins) else 0
        
        A = np.sin(theta) * np.cos(phi)
        B = np.sin(theta) * np.sin(phi)
        C = np.cos(theta)
        D = -rho
        
        return np.array([A, B, C, D])
    
    # Keep existing methods
    def _smart_subsample(self, points):
        """Subsample inteligent"""
        if len(points) <= FAST_SAMPLE_SIZE:
            return points
        
        n_uniform = int(FAST_SAMPLE_SIZE * 0.7)
        n_density = FAST_SAMPLE_SIZE - n_uniform
        
        uniform_indices = np.random.choice(len(points), n_uniform, replace=False)
        remaining_indices = np.setdiff1d(np.arange(len(points)), uniform_indices)
        
        if len(remaining_indices) >= n_density:
            density_indices = np.random.choice(remaining_indices, n_density, replace=False)
        else:
            density_indices = remaining_indices
        
        final_indices = np.concatenate([uniform_indices, density_indices])
        return points[final_indices]
    
    def _quick_refinement(self, points):
        """Refinement ultra-rapid"""
        distances = np.abs(points @ self.last_plane[:3] + self.last_plane[3])
        inlier_mask = distances < 0.1
        inliers = points[inlier_mask]
        
        if len(inliers) < 15:
            return None
        
        centroid = np.mean(inliers, axis=0)
        centered = inliers - centroid
        
        if len(centered) > 150:
            idx = np.random.choice(len(centered), 150, replace=False)
            centered = centered[idx]
        
        try:
            _, _, V = np.linalg.svd(centered)
            normal = V[-1]
            
            if np.dot(normal, self.last_plane[:3]) < 0:
                normal = -normal
                
            D = -np.dot(normal, centroid)
            refined_plane = np.array([normal[0], normal[1], normal[2], D])
            
            angle_diff = np.arccos(np.clip(
                abs(np.dot(self.last_plane[:3], refined_plane[:3])), -1, 1))
            
            if angle_diff < STABILITY_THRESHOLD:
                return refined_plane
                
        except:
            pass
            
        return None
    
    def _hough_3d_detect(self, points):
        """Hough 3D ultra-rapid"""
        if len(points) < 50:
            return None
        
        if len(points) > 500:
            idx = np.random.choice(len(points), 500, replace=False)
            hough_points = points[idx]
        else:
            hough_points = points
        
        max_dist = np.max(np.linalg.norm(hough_points, axis=1))
        rho_bins = np.arange(-max_dist, max_dist, HOUGH_RESOLUTION * 2)
        theta_bins = np.linspace(0, np.pi, 15)
        phi_bins = np.linspace(0, 2*np.pi, 15)
        
        accumulator = np.zeros((len(rho_bins), len(theta_bins), len(phi_bins)))
        
        for point in hough_points[::3]:
            x, y, z = point
            
            for i, theta in enumerate(theta_bins[::2]):
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                
                for j, phi in enumerate(phi_bins[::2]):
                    nx = sin_theta * np.cos(phi)
                    ny = sin_theta * np.sin(phi)
                    nz = cos_theta
                    
                    rho = x * nx + y * ny + z * nz
                    rho_idx = np.searchsorted(rho_bins, rho)
                    
                    if 0 <= rho_idx < len(rho_bins):
                        accumulator[rho_idx, i*2, j*2] += 1
        
        max_idx = np.unravel_index(np.argmax(accumulator), accumulator.shape)
        
        if accumulator[max_idx] < 5:
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
    
    def _prosac_detect(self, points):
        """PROSAC ultra-rapid"""
        if len(points) < 10:
            return None
        
        best_plane = None
        best_inliers = 0
        
        for iteration in range(15):
            if len(points) < 3:
                break
                
            sample_idx = np.random.choice(len(points), 3, replace=False)
            sample = points[sample_idx]
            
            plane = self._fit_plane_sample(sample)
            if plane is None:
                continue
            
            distances = np.abs(points @ plane[:3] + plane[3])
            inliers = np.sum(distances < 0.06)
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = plane
                
                if inliers > 0.6 * len(points):
                    break
        
        return best_plane
    
    def _fit_plane_sample(self, sample):
        """Fit rapid"""
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
        """Confidence rapid"""
        distances = np.abs(points @ plane[:3] + plane[3])
        return np.sum(distances < 0.05) / len(points)
    
    def _update_history(self, plane):
        """Update istoric"""
        self.last_plane = plane
        self.plane_history.append(plane)
        
        if len(self.plane_history) > 3:
            self.plane_history.pop(0)

# ───────── Helper Functions ─────────
def depth_to_points(depth_map, fx, fy, ppx, ppy, d_scale):
    """Convert depth map to 3D points"""
    ys, xs = np.nonzero(depth_map)
    zs = depth_map[ys, xs].astype(float) * d_scale
    X = (xs - ppx) * zs / fx
    Y = (ys - ppy) * zs / fy
    return np.vstack((X, Y, zs)).T

def compute_votes(pts, plane, ang_edges):
    """Compute votes for polar matrix"""
    if plane is None:
        return np.zeros((GRID_R, GRID_A // 2), dtype=float)
    
    A, B, C, D = plane
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
    """Expand 3×8 to 3×16"""
    first8 = H8[:, : (GRID_A // 2)]
    return np.repeat(first8, 2, axis=1).astype(int)

def setup_realsense_with_imu():
    """Configure RealSense pipeline with IMU"""
    pipe = rs.pipeline()
    cfg = rs.config()
    
    # Enable depth stream
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # Enable IMU streams
    try:
        cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)  # 250 Hz
        cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)   # 400 Hz
        imu_available = True
        print("IMU streams enabled successfully")
    except Exception as e:
        print(f"Warning: Could not enable IMU streams: {e}")
        print("Continuing with depth-only mode")
        imu_available = False
    
    profile = pipe.start(cfg)
    
    return pipe, profile, imu_available

# ───────── Main Script ─────────
def main():
    print("=== Fast Hybrid with IMU Calibration and Temporal Filtering ===")
    print(f"Processing {NUM_FRAMES} frames with IMU-enhanced obstacle detection...\n")
    
    # Setup RealSense with IMU
    pipe, profile, imu_available = setup_realsense_with_imu()
    
    # Initialize IMU calibration
    imu_calibration = IMUCalibration(enable_imu=imu_available and args.use_imu)
    
    # Initialize components
    detector = FastHybridPlaneDetector(imu_calibration)
    temporal_filter = TemporalObstacleFilter()
    
    # Get camera intrinsics
    sensor = profile.get_device().first_depth_sensor()
    d_scale = sensor.get_depth_scale()
    
    intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy
    H_img, W_img = intr.height, intr.width
    
    # Precompute angular edges
    FOV = 2 * math.atan((W_img / 2) / fx)
    ang_edges = np.linspace(-FOV / 2, FOV / 2, (GRID_A // 2) + 1)
    
    # Statistics tracking
    frames_processed = 0
    total_detection_time = 0
    method_counts = {"refinement": 0, "hough": 0, "prosac": 0, "temporal": 0, 
                    "imu_guided": 0, "refinement_imu": 0, "hough_imu": 0, "imu_only": 0}
    
    calibration_phase = args.calibrate_imu and imu_available
    if calibration_phase:
        print("Starting IMU calibration phase...")
        print("Please keep the camera stationary for optimal calibration.")
    
    while frames_processed < NUM_FRAMES:
        try:
            frames = pipe.wait_for_frames(timeout_ms=5000)
        except RuntimeError:
            continue
        
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue
        
        # Process IMU data if available
        current_timestamp = time.time() * 1000  # milliseconds
        accel_data = None
        gyro_data = None
        
        if imu_available and args.use_imu:
            # Get IMU frames
            try:
                if frames.first_or_default(rs.stream.accel):
                    accel_frame = frames.first_or_default(rs.stream.accel)
                    accel_data = np.array([accel_frame.as_motion_frame().get_motion_data().x,
                                          accel_frame.as_motion_frame().get_motion_data().y,
                                          accel_frame.as_motion_frame().get_motion_data().z])
                
                if frames.first_or_default(rs.stream.gyro):
                    gyro_frame = frames.first_or_default(rs.stream.gyro)
                    gyro_data = np.array([gyro_frame.as_motion_frame().get_motion_data().x,
                                         gyro_frame.as_motion_frame().get_motion_data().y,
                                         gyro_frame.as_motion_frame().get_motion_data().z])
                
                # Update IMU calibration/orientation
                if accel_data is not None and gyro_data is not None:
                    if calibration_phase and not imu_calibration.calibration_complete:
                        imu_calibration.add_calibration_sample(accel_data, gyro_data)
                        
                        # Show calibration progress
                        if frames_processed % 10 == 0:
                            progress = len(imu_calibration.accel_buffer) / IMU_CALIBRATION_FRAMES
                            print(f"Calibration progress: {progress*100:.1f}%", file=sys.stderr)
                        
                        if imu_calibration.calibration_complete:
                            calibration_phase = False
                            if imu_calibration.is_calibrated:
                                print("IMU calibration completed successfully!", file=sys.stderr)
                            else:
                                print("IMU calibration failed. Continuing without IMU.", file=sys.stderr)
                    
                    elif imu_calibration.is_calibrated:
                        imu_calibration.update_orientation(accel_data, gyro_data, current_timestamp)
                        
            except Exception as e:
                if frames_processed % 100 == 0:  # Avoid spam
                    print(f"IMU data processing error: {e}", file=sys.stderr)
        
        # Skip depth processing during initial calibration phase
        if calibration_phase:
            frames_processed += 1
            continue
        
        # Process depth frame
        depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
        pts3D = depth_to_points(depth_image, fx, fy, ppx, ppy, d_scale)
        
        if pts3D.shape[0] == 0:
            raw_H16 = np.zeros((GRID_R, GRID_A), dtype=int)
        else:
            # Ground point selection (enhanced with IMU tilt info)
            Ys = pts3D[:, 1]
            tilt_angle = imu_calibration.get_tilt_angle() if imu_calibration.is_ready() else 0.0
            
            # Adjust threshold based on tilt
            if tilt_angle > 15.0:
                # More permissive threshold when tilted
                threshold = np.percentile(Ys, 20)
            else:
                threshold = np.percentile(Ys, 25)
            
            ground_pts = pts3D[Ys < threshold]
            
            if ground_pts.shape[0] < MIN_GROUND_POINTS:
                raw_H16 = np.zeros((GRID_R, GRID_A), dtype=int)
            else:
                # Plane detection with IMU enhancement
                start_time = time.time()
                plane = detector.detect_plane(ground_pts)
                detection_time = (time.time() - start_time) * 1000
                
                total_detection_time += detection_time
                method_counts[detector.method_used] += 1
                
                if plane is None:
                    raw_H16 = np.zeros((GRID_R, GRID_A), dtype=int)
                else:
                    # Compute raw polar histogram
                    H8 = compute_votes(pts3D, plane, ang_edges)
                    raw_H16 = duplicate_bins(H8)
        
        # Apply temporal filtering with IMU info
        imu_confidence = imu_calibration.get_orientation_confidence() if imu_calibration.is_ready() else 0.0
        tilt_angle = imu_calibration.get_tilt_angle() if imu_calibration.is_ready() else 0.0
        
        filtered_H16 = temporal_filter.update(raw_H16, imu_confidence, tilt_angle)
        
        # Print filtered matrix
        for row in filtered_H16:
            print(",".join(str(int(v)) for v in row))
        print("---")
        
        frames_processed += 1
        
        # Progress reporting every 50 frames
        if frames_processed % 50 == 0:
            avg_time = total_detection_time / max(1, frames_processed - IMU_CALIBRATION_FRAMES)
            debug_info = temporal_filter.get_debug_info()
            
            # IMU status
            imu_status = "Ready" if imu_calibration.is_ready() else "Not Ready"
            confidence = imu_calibration.get_orientation_confidence()
            tilt = imu_calibration.get_tilt_angle()
            
            print(f"# Progress: {frames_processed}/{NUM_FRAMES}, "
                  f"avg: {avg_time:.2f}ms, "
                  f"active_cells: {debug_info['active_cells']}, "
                  f"IMU: {imu_status} (conf: {confidence:.2f}, tilt: {tilt:.1f}°)", 
                  file=sys.stderr)
    
    # Final statistics
    pipe.stop()
    
    processing_frames = max(1, frames_processed - IMU_CALIBRATION_FRAMES)
    avg_time = total_detection_time / processing_frames
    
    print(f"\n=== PERFORMANCE SUMMARY ===", file=sys.stderr)
    print(f"Total frames: {frames_processed}", file=sys.stderr)
    print(f"Processing frames: {processing_frames}", file=sys.stderr)
    print(f"Average detection time: {avg_time:.2f}ms", file=sys.stderr)
    print(f"Detection methods used: {method_counts}", file=sys.stderr)
    
    # IMU summary
    if imu_calibration.enable_imu:
        print(f"\n=== IMU SUMMARY ===", file=sys.stderr)
        print(f"IMU Status: {'Calibrated' if imu_calibration.is_calibrated else 'Not Calibrated'}", file=sys.stderr)
        if imu_calibration.is_calibrated:
            print(f"Final orientation confidence: {imu_calibration.get_orientation_confidence():.3f}", file=sys.stderr)
            print(f"Final tilt angle: {imu_calibration.get_tilt_angle():.1f}°", file=sys.stderr)
            
            # Method efficiency with IMU
            imu_methods = method_counts["imu_guided"] + method_counts["refinement_imu"] + method_counts["hough_imu"] + method_counts["imu_only"]
            total_methods = sum(method_counts.values())
            if total_methods > 0:
                imu_efficiency = (imu_methods / total_methods) * 100
                print(f"IMU-enhanced detections: {imu_efficiency:.1f}%", file=sys.stderr)
    
    # Temporal filter summary
    debug_info = temporal_filter.get_debug_info()
    print(f"\n=== TEMPORAL FILTER SUMMARY ===", file=sys.stderr)
    print(f"Final active cells: {debug_info['active_cells']}", file=sys.stderr)
    print(f"Max accumulator value: {debug_info['accumulator_max']:.0f}", file=sys.stderr)
    print(f"Mean accumulator value: {debug_info['accumulator_mean']:.2f}", file=sys.stderr)

if __name__ == "__main__":
    main()