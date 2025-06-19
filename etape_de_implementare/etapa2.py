#!/usr/bin/env python3

import math
import numpy as np
import cv2
import pyrealsense2 as rs
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

# ───────── Enhanced parameters for blind navigation ─────────
# Simplified radial zones - focus on immediate navigation needs
RADIAL_EDGES = np.array([0.0, 0.8, 1.5, 3.0])  # Near, arm-reach, step, far
GRID_R, GRID_A = len(RADIAL_EDGES) - 1, 12  # Fewer angular bins for clarity

# Tactile feedback parameters
DANGER_THRESHOLD = 0.6      # Distance below which obstacle is dangerous (meters)
WARNING_THRESHOLD = 1.2     # Distance for early warning
OBSTACLE_MIN_HEIGHT = 0.15  # Minimum height to consider as obstacle
OBSTACLE_MAX_HEIGHT = 2.0   # Maximum relevant height
HEAD_HEIGHT = 1.6           # Typical head height for hanging obstacles

# ANTI-SPIKE PARAMETERS - New smoothing controls
MATRIX_HISTORY_SIZE = 8     # Number of previous matrices to keep for smoothing
TEMPORAL_ALPHA = 0.3        # Temporal smoothing factor (lower = more smoothing)
SPIKE_THRESHOLD = 150       # Intensity jump threshold to detect spikes
MIN_PERSISTENCE_FRAMES = 3  # Frames an obstacle must persist to be valid
MAX_INTENSITY_CHANGE = 100  # Maximum allowed intensity change per frame
MEDIAN_FILTER_SIZE = 5      # Size of median filter for spike removal

# Smoothing and stability
EMA_ALPHA = 0.5           # Faster response for safety
STABILITY_FRAMES = 5       # Frames to confirm obstacle presence
MIN_OBSTACLE_POINTS = 100  # Minimum points to confirm obstacle

# Audio/Haptic zones (left-center-right and near-far)
HAPTIC_ZONES = {
    'immediate': 0,    # 0-0.8m
    'close': 1,        # 0.8-1.5m  
    'moderate': 2      # 1.5-3.0m
}

DIRECTION_ZONES = {
    'hard_left': 0,    # -60° to -30°
    'left': 1,         # -30° to -10°
    'center': 2,       # -10° to 10°
    'right': 3,        # 10° to 30°
    'hard_right': 4    # 30° to 60°
}

@dataclass
class ObstacleInfo:
    distance: float
    angle: float
    height: float
    confidence: float
    zone: str
    direction: str
    is_hanging: bool = False
    is_moving: bool = False

class TactileNavigationSystem:
    def __init__(self):
        self.setup_realsense()
        self.obstacle_history = deque(maxlen=STABILITY_FRAMES)
        self.plane = None
        self.ema_grid = np.zeros((GRID_R, GRID_A), float)
        self.last_obstacles = []
        
        # ANTI-SPIKE ADDITIONS - Smoothing buffers
        self.matrix_history = deque(maxlen=MATRIX_HISTORY_SIZE)
        self.last_smooth_matrix = np.zeros(48)
        self.obstacle_persistence = {}  # Track how long obstacles have been detected
        self.frame_count = 0
        
        # Enhanced filters for better obstacle detection
        self.setup_filters()
        
    def setup_realsense(self):
        self.pipe = rs.pipeline()
        self.cfg = rs.config()
        
        # Higher resolution for better obstacle detection
        self.cfg.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        self.cfg.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
        
        self.profile = self.pipe.start(self.cfg)
        self.d_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        self.intr = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        
        # Camera parameters
        self.fx, self.fy = self.intr.fx, self.intr.fy
        self.ppx, self.ppy = self.intr.ppx, self.intr.ppy
        
        # Field of view - focus on navigation-relevant area
        self.FOV = math.radians(120)  # 120 degrees total FOV
        self.ANG_EDGES = np.linspace(-self.FOV/2, self.FOV/2, GRID_A+1)
        
        self.align = rs.align(rs.stream.color)
        
    def setup_filters(self):
        """Enhanced filtering for obstacle detection with stronger noise reduction"""
        self.dec_filter = rs.decimation_filter(2)
        self.thr_filter = rs.threshold_filter(0.1, 4.0)
        self.d2d = rs.disparity_transform(True)
        
        # STRONGER spatial filtering to reduce noise spikes
        self.spat_filter = rs.spatial_filter()
        self.spat_filter.set_option(rs.option.filter_magnitude, 5)
        self.spat_filter.set_option(rs.option.filter_smooth_alpha, 0.8)  # Increased from 0.6
        self.spat_filter.set_option(rs.option.filter_smooth_delta, 40)   # Increased from 25
        
        # STRONGER temporal filter for stability
        self.temp_filter = rs.temporal_filter()
        self.temp_filter.set_option(rs.option.filter_smooth_alpha, 0.7)  # Increased from 0.5
        self.temp_filter.set_option(rs.option.filter_smooth_delta, 50)   # Increased from 30
        
        self.fill_holes = rs.hole_filling_filter(2)
        
    def depth_to_points(self, depth_map):
        """Convert depth map to 3D points with outlier removal"""
        ys, xs = np.nonzero(depth_map)
        if len(ys) == 0:
            return np.empty((0, 3))
            
        zs = depth_map[ys, xs]
        
        # OUTLIER REMOVAL - Remove obviously bad depth values
        valid_mask = (zs > 0.1) & (zs < 5.0)  # Keep points between 10cm and 5m
        ys, xs, zs = ys[valid_mask], xs[valid_mask], zs[valid_mask]
        
        if len(zs) == 0:
            return np.empty((0, 3))
            
        X = (xs - self.ppx) * zs / self.fx
        Y = (ys - self.ppy) * zs / self.fy
        points = np.vstack((X, Y, zs)).T
        
        # Additional outlier removal using statistical methods
        if len(points) > 100:
            # Remove points that are too far from the median distance
            distances = np.sqrt(np.sum(points**2, axis=1))
            median_dist = np.median(distances)
            mad = np.median(np.abs(distances - median_dist))
            threshold = median_dist + 3 * mad  # 3-sigma rule
            valid_mask = distances < threshold
            points = points[valid_mask]
            
        return points
    
    def enhanced_ground_detection(self, pts):
        """Improved ground plane detection with better stability"""
        if len(pts) < 100:
            return self.plane
            
        # Use lower portion of points for ground detection
        ground_candidates = pts[pts[:, 1] > np.percentile(pts[:, 1], 70)]
        
        if len(ground_candidates) < 50:
            return self.plane
            
        # RANSAC with multiple iterations and better validation
        best_plane, best_inliers, best_score = None, 0, 0
        
        for _ in range(150):  # More iterations for better accuracy
            sample_idx = np.random.choice(len(ground_candidates), 3, replace=False)
            sample_pts = ground_candidates[sample_idx]
            
            # Calculate plane normal
            v1 = sample_pts[1] - sample_pts[0]
            v2 = sample_pts[2] - sample_pts[0]
            normal = np.cross(v1, v2)
            
            if np.linalg.norm(normal) < 1e-6:
                continue
                
            normal = normal / np.linalg.norm(normal)
            
            # Plane equation: ax + by + cz + d = 0
            a, b, c = normal
            d = -normal.dot(sample_pts[0])
            
            # Count inliers with tighter tolerance
            distances = np.abs((ground_candidates @ normal) + d)
            inliers = np.sum(distances < 0.03)  # Tighter 3cm tolerance
            
            # Additional validation: plane should be roughly horizontal
            angle_from_horizontal = np.abs(np.arccos(np.abs(normal[1])))
            if angle_from_horizontal > np.pi/6:  # More than 30 degrees
                continue
                
            # Score based on inliers and plane orientation
            score = inliers * (1.0 - angle_from_horizontal / (np.pi/8))
            
            if score > best_score:
                best_score = score
                best_inliers = inliers
                best_plane = np.array([a, b, c, d])
        
        if best_plane is not None:
            # STRONGER smoothing for plane updates
            if self.plane is not None:
                self.plane = 0.85 * self.plane + 0.15 * best_plane  # More conservative update
            else:
                self.plane = best_plane
                
        return self.plane
    
    def detect_obstacles(self, pts, plane):
        """Enhanced obstacle detection with persistence tracking"""
        if plane is None or len(pts) == 0:
            return []
            
        a, b, c, d = plane
        plane_normal = np.array([a, b, c])
        
        # Calculate height above ground plane
        heights = ((pts @ plane_normal) + d) / np.linalg.norm(plane_normal)
        
        # Filter points that are actual obstacles with stricter criteria
        obstacle_mask = (heights > OBSTACLE_MIN_HEIGHT) & (heights < OBSTACLE_MAX_HEIGHT)
        obstacle_pts = pts[obstacle_mask]
        obstacle_heights = heights[obstacle_mask]
        
        if len(obstacle_pts) < MIN_OBSTACLE_POINTS:
            return []
            
        # Calculate polar coordinates
        distances = np.sqrt(obstacle_pts[:, 0]**2 + obstacle_pts[:, 2]**2)
        angles = np.arctan2(obstacle_pts[:, 0], obstacle_pts[:, 2])
        
        # Group obstacles by proximity and angle
        obstacles = []
        
        # Create angular bins for obstacle grouping
        ang_bins = np.linspace(-self.FOV/2, self.FOV/2, 13)
        dist_bins = RADIAL_EDGES
        
        for i in range(len(ang_bins) - 1):
            for j in range(len(dist_bins) - 1):
                # Find points in this angular and distance bin
                ang_mask = (angles >= ang_bins[i]) & (angles < ang_bins[i+1])
                dist_mask = (distances >= dist_bins[j]) & (distances < dist_bins[j+1])
                bin_mask = ang_mask & dist_mask
                
                # STRICTER point count requirement for immediate zone
                min_points = MIN_OBSTACLE_POINTS * 1.5 if j == 0 else MIN_OBSTACLE_POINTS
                if np.sum(bin_mask) < min_points:
                    continue
                    
                bin_pts = obstacle_pts[bin_mask]
                bin_heights = obstacle_heights[bin_mask]
                bin_distances = distances[bin_mask]
                
                # Calculate obstacle properties
                avg_distance = np.mean(bin_distances)
                avg_angle = (ang_bins[i] + ang_bins[i+1]) / 2
                max_height = np.max(bin_heights)
                avg_height = np.mean(bin_heights)
                
                # PERSISTENCE TRACKING - Create unique ID for this obstacle
                obstacle_id = f"{j}_{i}"  # distance_zone_angular_bin
                
                # Track persistence
                if obstacle_id not in self.obstacle_persistence:
                    self.obstacle_persistence[obstacle_id] = 1
                else:
                    self.obstacle_persistence[obstacle_id] += 1
                
                # Only include obstacles that have persisted for minimum frames
                if self.obstacle_persistence[obstacle_id] < MIN_PERSISTENCE_FRAMES:
                    continue
                
                # Determine zones
                zone = self.get_distance_zone(avg_distance)
                direction = self.get_direction_zone(avg_angle)
                
                # Check for hanging obstacles (head height)
                is_hanging = max_height > HEAD_HEIGHT * 0.8
                
                # Enhanced confidence calculation with persistence bonus
                base_confidence = min(1.0, np.sum(bin_mask) / (min_points * 2))
                persistence_bonus = min(0.3, self.obstacle_persistence[obstacle_id] * 0.05)
                confidence = min(1.0, base_confidence + persistence_bonus)
                
                obstacle = ObstacleInfo(
                    distance=avg_distance,
                    angle=avg_angle,
                    height=avg_height,
                    confidence=confidence,
                    zone=zone,
                    direction=direction,
                    is_hanging=is_hanging
                )
                
                obstacles.append(obstacle)
        
        # Clean up old persistence entries
        current_ids = set()
        for i in range(len(ang_bins) - 1):
            for j in range(len(dist_bins) - 1):
                current_ids.add(f"{j}_{i}")
        
        # Remove persistence entries for obstacles no longer detected
        to_remove = []
        for obstacle_id in self.obstacle_persistence:
            if obstacle_id not in current_ids:
                to_remove.append(obstacle_id)
        
        for obstacle_id in to_remove:
            del self.obstacle_persistence[obstacle_id]
        
        return obstacles
    
    def get_distance_zone(self, distance):
        """Map distance to tactile feedback zone"""
        if distance < RADIAL_EDGES[1]:
            return 'immediate'
        elif distance < RADIAL_EDGES[2]:
            return 'close'
        else:
            return 'moderate'
    
    def get_direction_zone(self, angle):
        """Map angle to direction zone"""
        angle_deg = math.degrees(angle)
        if angle_deg < -30:
            return 'hard_left'
        elif angle_deg < -10:
            return 'left'
        elif angle_deg < 10:
            return 'center'
        elif angle_deg < 30:
            return 'right'
        else:
            return 'hard_right'
    
    def median_filter_1d(self, data, kernel_size=5):
        """Apply median filter to remove spikes"""
        if len(data) < kernel_size:
            return data
            
        filtered = np.copy(data)
        pad_size = kernel_size // 2
        
        for i in range(pad_size, len(data) - pad_size):
            window = data[i - pad_size:i + pad_size + 1]
            filtered[i] = np.median(window)
            
        return filtered
    
    def apply_temporal_smoothing(self, current_matrix):
        """Apply multiple smoothing techniques to reduce spikes"""
        # Convert to 2D for easier processing
        current_2d = current_matrix.reshape(3, 16)
        
        # 1. MEDIAN FILTERING - Remove isolated spikes
        smoothed_2d = np.zeros_like(current_2d)
        for i in range(3):  # For each distance zone
            smoothed_2d[i] = self.median_filter_1d(current_2d[i], MEDIAN_FILTER_SIZE)
        
        smoothed_matrix = smoothed_2d.flatten()
        
        # 2. INTENSITY CHANGE LIMITING - Prevent sudden jumps
        if len(self.matrix_history) > 0:
            last_matrix = self.matrix_history[-1]
            intensity_diff = smoothed_matrix - last_matrix
            
            # Limit changes to maximum allowed per frame
            clamped_diff = np.clip(intensity_diff, -MAX_INTENSITY_CHANGE, MAX_INTENSITY_CHANGE)
            smoothed_matrix = last_matrix + clamped_diff
        
        # 3. TEMPORAL EXPONENTIAL SMOOTHING
        if len(self.matrix_history) > 0:
            smoothed_matrix = (TEMPORAL_ALPHA * smoothed_matrix + 
                             (1 - TEMPORAL_ALPHA) * self.last_smooth_matrix)
        
        # 4. SPIKE DETECTION AND SUPPRESSION
        if len(self.matrix_history) >= 2:
            # Compare with recent history
            recent_avg = np.mean([self.matrix_history[i] for i in range(-2, 0)], axis=0)
            spike_mask = np.abs(smoothed_matrix - recent_avg) > SPIKE_THRESHOLD
            
            # Replace spikes with smoothed values
            smoothed_matrix[spike_mask] = recent_avg[spike_mask]
        
        # 5. ENSURE NON-NEGATIVE VALUES
        smoothed_matrix = np.maximum(0, smoothed_matrix)
        
        # Store for next iteration
        self.matrix_history.append(smoothed_matrix.copy())
        self.last_smooth_matrix = smoothed_matrix.copy()
        
        return smoothed_matrix
    
    def generate_tactile_pattern(self, obstacles):
        """Generate tactile feedback pattern for haptic device"""
        # Initialize 5-zone haptic pattern (left, center-left, center, center-right, right)
        haptic_intensity = np.zeros(5)
        haptic_frequency = np.zeros(5)  # For vibration frequency
        
        direction_map = {
            'hard_left': 0, 'left': 1, 'center': 2, 'right': 3, 'hard_right': 4
        }
        
        for obstacle in obstacles:
            zone_idx = direction_map[obstacle.direction]
            
            # Intensity based on proximity and confidence
            if obstacle.zone == 'immediate':
                intensity = 1.0 * obstacle.confidence
                frequency = 20  # High frequency for immediate danger
            elif obstacle.zone == 'close':
                intensity = 0.7 * obstacle.confidence
                frequency = 10  # Medium frequency
            else:
                intensity = 0.4 * obstacle.confidence
                frequency = 5   # Low frequency
            
            # Boost intensity for hanging obstacles
            if obstacle.is_hanging:
                intensity *= 1.5
                frequency += 10
            
            # Update haptic arrays
            haptic_intensity[zone_idx] = max(haptic_intensity[zone_idx], intensity)
            haptic_frequency[zone_idx] = max(haptic_frequency[zone_idx], frequency)
        
        return haptic_intensity, haptic_frequency
    
    def generate_48_stimulus_matrix(self, pts, plane):
        """Generate 48-stimulus depth matrix with enhanced smoothing"""
        # Initialize 48-element matrix: 3 distances × 16 directions
        stimulus_matrix = np.zeros((3, 16), dtype=float)
        
        if plane is None or len(pts) == 0:
            return self.apply_temporal_smoothing(stimulus_matrix.flatten())
        
        # Calculate height above ground plane
        a, b, c, d = plane
        plane_normal = np.array([a, b, c])
        heights = ((pts @ plane_normal) + d) / np.linalg.norm(plane_normal)
        
        # Filter for relevant obstacles
        obstacle_mask = (heights > OBSTACLE_MIN_HEIGHT) & (heights < OBSTACLE_MAX_HEIGHT)
        obstacle_pts = pts[obstacle_mask]
        
        if len(obstacle_pts) == 0:
            return self.apply_temporal_smoothing(stimulus_matrix.flatten())
        
        # Calculate polar coordinates
        distances = np.sqrt(obstacle_pts[:, 0]**2 + obstacle_pts[:, 2]**2)
        angles = np.arctan2(obstacle_pts[:, 0], obstacle_pts[:, 2])
        
        # Distance zones from RADIAL_EDGES
        distance_zones = RADIAL_EDGES  # [0.0, 0.8, 1.5, 3.0]
        
        # Define clustering patterns for intuitive spatial feedback
        LEFT_CLUSTER = list(range(0, 7))      # Indices 0-6: Left side activation (7 elements)
        CENTER_CLUSTER = list(range(7, 9))    # Indices 7-8: Center activation (2 elements) 
        RIGHT_CLUSTER = list(range(9, 16))    # Indices 9-15: Right side activation (7 elements)
        
        # Angular thresholds for left/center/right classification - EXPANDED RANGES
        LEFT_ANGLE_THRESHOLD = -np.pi/8     # -22.5 degrees (was -30)
        RIGHT_ANGLE_THRESHOLD = np.pi/8     # +22.5 degrees (was +30)
        
        # Process each distance zone (immediate, close, moderate)
        for distance_zone in range(3):
            r_min, r_max = distance_zones[distance_zone], distance_zones[distance_zone + 1]
            
            # Find all points in this distance zone
            distance_mask = (distances >= r_min) & (distances < r_max)
            
            if not np.any(distance_mask):
                continue
                
            zone_distances = distances[distance_mask]
            zone_angles = angles[distance_mask]
            zone_heights = heights[obstacle_mask][distance_mask]
            
            # STRICTER requirements for immediate zone to reduce false positives
            min_points_zone = MIN_OBSTACLE_POINTS * 2 if distance_zone == 0 else MIN_OBSTACLE_POINTS
            
            # IMPROVED angular classification - wider ranges and better distribution
            left_mask = zone_angles < LEFT_ANGLE_THRESHOLD        # < -22.5°
            center_mask = (zone_angles >= LEFT_ANGLE_THRESHOLD) & (zone_angles <= RIGHT_ANGLE_THRESHOLD)  # -22.5° to +22.5°
            right_mask = zone_angles > RIGHT_ANGLE_THRESHOLD      # > +22.5°
            
            # Debug: Print angular distribution
            if distance_zone == 0 and len(zone_angles) > 0:  # Only for immediate zone
                print(f"Angular distribution - Left: {np.sum(left_mask)}, Center: {np.sum(center_mask)}, Right: {np.sum(right_mask)}")
                print(f"Angle range: {np.degrees(np.min(zone_angles)):.1f}° to {np.degrees(np.max(zone_angles)):.1f}°")
            
            # Calculate intensity for each spatial region
            regions = [
                ('left', left_mask, LEFT_CLUSTER),
                ('center', center_mask, CENTER_CLUSTER), 
                ('right', right_mask, RIGHT_CLUSTER)
            ]
            
            for region_name, region_mask, cluster_indices in regions:
                if not np.any(region_mask):
                    continue
                    
                # Check minimum point requirement - REDUCED for better coverage
                point_count = np.sum(region_mask)
                min_points_required = min_points_zone // 2 if region_name != 'center' else min_points_zone // 3
                if point_count < min_points_required:
                    continue
                    
                # Extract region data
                region_distances = zone_distances[region_mask]
                region_heights = zone_heights[region_mask]
                
                # Calculate composite intensity based on multiple factors
                
                # 1. Point density factor (more points = stronger signal)
                density_factor = min(1.0, point_count / min_points_required)
                
                # 2. Proximity factor (closer obstacles = stronger signal)
                avg_distance = np.mean(region_distances)
                proximity_factor = 1.0 - (avg_distance - r_min) / (r_max - r_min)
                proximity_factor = max(0.1, proximity_factor)  # Minimum 10% intensity
                
                # 3. Height factor (head-level and low obstacles are more critical)
                max_height = np.max(region_heights)
                avg_height = np.mean(region_heights)
                
                height_factor = 1.0
                if max_height > HEAD_HEIGHT * 0.8:  # Hanging obstacle (head level)
                    height_factor = 1.5
                elif max_height < 0.5:  # Low obstacle (tripping hazard)
                    height_factor = 1.3
                elif avg_height > 1.0:  # Tall obstacles
                    height_factor = 1.2
                
                # 4. Distance zone urgency factor with REDUCED immediate zone sensitivity
                urgency_factor = 1.0
                if distance_zone == 0:      # Immediate zone - reduced from 1.5 to 1.2
                    urgency_factor = 1.2
                elif distance_zone == 1:    # Close zone  
                    urgency_factor = 1.1
                # Moderate zone keeps factor = 1.0
                
                # Combine all factors to calculate final intensity
                final_intensity = density_factor * proximity_factor * height_factor * urgency_factor
                
                # REDUCED maximum intensity to prevent spikes
                max_intensity = 200 if distance_zone == 0 else 255  # Immediate zone capped at 200
                intensity_value = min(max_intensity, int(final_intensity * max_intensity))
                
                # Apply intensity to ALL elements in the cluster
                if intensity_value > 0:
                    for cluster_idx in cluster_indices:
                        # Use max to handle overlapping detections
                        stimulus_matrix[distance_zone, cluster_idx] = max(
                            stimulus_matrix[distance_zone, cluster_idx], 
                            intensity_value
                        )
                    
                    # Debug: Show which cluster was activated
                    if distance_zone == 0:  # Only for immediate zone
                        print(f"Activated {region_name} cluster (indices {cluster_indices[0]}-{cluster_indices[-1]}) with intensity {intensity_value}")
        
        # FALLBACK: If no obstacles detected in main logic, try alternative approach
        if np.all(stimulus_matrix == 0) and len(obstacle_pts) > MIN_OBSTACLE_POINTS // 4:
            print("No obstacles in main logic, trying alternative approach...")
            
            # Alternative approach: Map directly to 16-bin angular grid
            for distance_zone in range(3):
                r_min, r_max = distance_zones[distance_zone], distance_zones[distance_zone + 1]
                distance_mask = (distances >= r_min) & (distances < r_max)
                
                if not np.any(distance_mask):
                    continue
                
                zone_angles = angles[distance_mask]
                zone_heights = heights[obstacle_mask][distance_mask]
                
                # Create 16 angular bins spanning the full FOV
                angular_bins = np.linspace(-self.FOV/2, self.FOV/2, 17)  # 16 bins + 1 edge
                
                for bin_idx in range(16):
                    angle_min, angle_max = angular_bins[bin_idx], angular_bins[bin_idx + 1]
                    bin_mask = (zone_angles >= angle_min) & (zone_angles < angle_max)
                    
                    if np.sum(bin_mask) > MIN_OBSTACLE_POINTS // 8:  # Much lower threshold
                        # Simple intensity based on point count
                        intensity = min(150, int(np.sum(bin_mask) / (MIN_OBSTACLE_POINTS // 8) * 150))
                        stimulus_matrix[distance_zone, bin_idx] = intensity
                        print(f"Alternative: Set bin {bin_idx} to intensity {intensity}")
        
        # Apply temporal smoothing to reduce spikes
        return self.apply_temporal_smoothing(stimulus_matrix.flatten())

    def visualize_48_stimulus_matrix_enhanced(self, stimulus_matrix):
        """Enhanced visualization showing the clustered activation pattern with smoothing info"""
        # Reshape flat array back to 3x16 matrix
        matrix_2d = stimulus_matrix.reshape(3, 16)
        
        # Create larger visualization with better labeling
        cell_size = 50
        vis_img = np.zeros((3 * cell_size + 160, 16 * cell_size + 200, 3), dtype=np.uint8)
        
        # Color mapping for intensities
        for i in range(3):
            for j in range(16):
                intensity = matrix_2d[i, j]
                
                # Map intensity to color (0-255 -> blue to red)
                if intensity > 0:
                    color_ratio = intensity / 255.0
                    blue = int(255 * (1 - color_ratio))
                    red = int(255 * color_ratio)
                    color = (blue, 0, red)
                else:
                    color = (30, 30, 30)  # Dark gray for no obstacle
                
                # Fill cell with offset for labels
                y1, y2 = 60 + i * cell_size, 60 + (i + 1) * cell_size
                x1, x2 = 100 + j * cell_size, 100 + (j + 1) * cell_size
                vis_img[y1:y2, x1:x2] = color
                
                # Add intensity value text
                text_color = (255, 255, 255) if intensity < 128 else (0, 0, 0)
                cv2.putText(vis_img, str(int(intensity)), 
                           (x1 + 10, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, text_color, 1)
        
        # Add enhanced labels
        # Distance zone labels
        zone_labels = ["IMMEDIATE (0-0.8m)", "CLOSE (0.8-1.5m)", "MODERATE (1.5-3.0m)"]
        zone_colors = [(0, 0, 255), (0, 165, 255), (0, 255, 255)]  # Red, Orange, Yellow
        
        for i, (label, color) in enumerate(zip(zone_labels, zone_colors)):
            y_pos = 60 + i * cell_size + cell_size // 2
            cv2.putText(vis_img, label, (5, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Cluster region labels at top
        cv2.putText(vis_img, "LEFT", (int(100 + 3.5 * cell_size), 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_img, "CENTER", (int(100 + 7.5 * cell_size), 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_img, "RIGHT", (int(100 + 12 * cell_size), 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw cluster boundaries
        # Left cluster boundary (after index 6)
        x_boundary = 100 + 7 * cell_size
        cv2.line(vis_img, (x_boundary, 60), (x_boundary, 60 + 3 * cell_size), (255, 255, 255), 2)
        
        # Center cluster boundary (after index 8)  
        x_boundary = 100 + 9 * cell_size
        cv2.line(vis_img, (x_boundary, 60), (x_boundary, 60 + 3 * cell_size), (255, 255, 255), 2)
        
        # Add smoothing info
        smoothing_info = f"Smoothed | History: {len(self.matrix_history)}/{MATRIX_HISTORY_SIZE}"
        cv2.putText(vis_img, smoothing_info, (100, 60 + 3 * cell_size + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        
        # Add title
        cv2.putText(vis_img, "48-STIMULUS TACTILE MATRIX - ANTI-SPIKE SMOOTHED", 
                   (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_img
    
    def generate_audio_cues(self, obstacles):
        """Generate audio navigation cues"""
        if not obstacles:
            return "Path clear"
        
        # Prioritize by proximity and direction
        critical_obstacles = [obs for obs in obstacles if obs.zone == 'immediate']
        close_obstacles = [obs for obs in obstacles if obs.zone == 'close']
        
        cues = []
        
        if critical_obstacles:
            # Immediate danger
            for obs in critical_obstacles:
                direction_text = obs.direction.replace('_', ' ')
                if obs.is_hanging:
                    cues.append(f"STOP! Hanging obstacle {direction_text}")
                else:
                    cues.append(f"STOP! Obstacle {direction_text}")
        
        elif close_obstacles:
            # Close obstacles - provide navigation guidance
            left_obs = [obs for obs in close_obstacles if 'left' in obs.direction]
            right_obs = [obs for obs in close_obstacles if 'right' in obs.direction]
            center_obs = [obs for obs in close_obstacles if obs.direction == 'center']
            
            if center_obs:
                if left_obs and not right_obs:
                    cues.append("Obstacle ahead, move right")
                elif right_obs and not left_obs:
                    cues.append("Obstacle ahead, move left")
                else:
                    cues.append("Obstacle ahead, proceed carefully")
            
            elif left_obs and not right_obs:
                cues.append("Obstacle on left")
            elif right_obs and not left_obs:
                cues.append("Obstacle on right")
        
        return "; ".join(cues) if cues else "Path clear ahead"
    
    def visualize_48_stimulus_matrix(self, stimulus_matrix):
        """Visualize the 48-stimulus matrix as a heatmap"""
        # Reshape flat array back to 3x16 matrix
        matrix_2d = stimulus_matrix.reshape(3, 16)
        
        # Create larger visualization
        cell_size = 40
        vis_img = np.zeros((3 * cell_size, 16 * cell_size, 3), dtype=np.uint8)
        
        # Color mapping for intensities
        for i in range(3):
            for j in range(16):
                intensity = matrix_2d[i, j]
                
                # Map intensity to color (0-255 -> blue to red)
                if intensity > 0:
                    color_ratio = intensity / 255.0
                    blue = int(255 * (1 - color_ratio))
                    red = int(255 * color_ratio)
                    color = (blue, 0, red)
                else:
                    color = (50, 50, 50)  # Dark gray for no obstacle
                
                # Fill cell
                y1, y2 = i * cell_size, (i + 1) * cell_size
                x1, x2 = j * cell_size, (j + 1) * cell_size
                vis_img[y1:y2, x1:x2] = color
                
                # Add text showing intensity value
                text_color = (255, 255, 255) if intensity < 128 else (0, 0, 0)
                cv2.putText(vis_img, str(int(intensity)), 
                           (x1 + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, text_color, 1)
        
        # Add labels
        label_img = np.zeros((vis_img.shape[0] + 60, vis_img.shape[1] + 100, 3), dtype=np.uint8)
        label_img[30:30+vis_img.shape[0], 80:80+vis_img.shape[1]] = vis_img
        
        # Distance zone labels
        zone_labels = ["IMMEDIATE (0-0.8m)", "CLOSE (0.8-1.5m)", "MODERATE (1.5-3.0m)"]
        for i, label in enumerate(zone_labels):
            y_pos = 30 + i * cell_size + cell_size // 2
            cv2.putText(label_img, label, (5, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Angular labels (showing degrees)
        for j in range(0, 16, 4):  # Show every 4th angle
            angle_deg = int(-180 + j * 22.5)
            x_pos = int(80 + j * cell_size + cell_size // 2)
            cv2.putText(label_img, f"{angle_deg}°", (x_pos - 15, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        
        return label_img
    
    def visualize_navigation_aid(self, color_image, obstacles):
        """Create visual representation for testing/development"""
        img = color_image.copy()
        h, w = img.shape[:2]
        
        # Draw navigation zones
        zone_colors = {
            'immediate': (0, 0, 255),    # Red
            'close': (0, 165, 255),      # Orange
            'moderate': (0, 255, 255)    # Yellow
        }
        
        # Draw radial zones
        cx, cy = w // 2, h
        for i, (zone_name, color) in enumerate(zone_colors.items()):
            if i < len(RADIAL_EDGES) - 1:
                r_inner = int(RADIAL_EDGES[i] / RADIAL_EDGES[-1] * h * 0.4)
                r_outer = int(RADIAL_EDGES[i+1] / RADIAL_EDGES[-1] * h * 0.4)
                
                cv2.ellipse(img, (cx, cy), (r_outer, r_outer), 0, 
                           math.degrees(-self.FOV/2), math.degrees(self.FOV/2), 
                           color, 2)
        
        # Draw obstacles with persistence indicators
        for obstacle in obstacles:
            # Calculate position on image
            angle_norm = obstacle.angle / (self.FOV/2)
            dist_norm = obstacle.distance / RADIAL_EDGES[-1]
            
            x = int(cx + angle_norm * w * 0.3)
            y = int(cy - dist_norm * h * 0.4)
            
            # Color based on zone
            color = zone_colors.get(obstacle.zone, (255, 255, 255))
            
            # Draw obstacle marker with size based on confidence
            radius = int(8 * obstacle.confidence)
            cv2.circle(img, (x, y), radius, color, -1)
            
            # Add text info
            text = f"{obstacle.zone[0].upper()}"
            if obstacle.is_hanging:
                text += "H"
            
            cv2.putText(img, text, (x-10, y-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add smoothing status
        smoothing_text = f"Smoothing: {len(self.matrix_history)}/{MATRIX_HISTORY_SIZE} frames"
        cv2.putText(img, smoothing_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return img
    
    def run(self):
        """Main processing loop with enhanced anti-spike features"""
        print("ANTI-SPIKE Tactile Navigation System Active")
        print("Press 'q' to quit, 'r' to reset ground plane, 's' to reset smoothing")
        print("System generates 48-stimulus matrix: 16 directions × 3 distances")
        print(f"Smoothing: {MATRIX_HISTORY_SIZE} frame history, α={TEMPORAL_ALPHA}")
        print(f"Spike protection: threshold={SPIKE_THRESHOLD}, max_change={MAX_INTENSITY_CHANGE}")
        
        cv2.namedWindow('Navigation Aid', cv2.WINDOW_NORMAL)
        cv2.namedWindow('48-Stimulus Matrix', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Enhanced Matrix', cv2.WINDOW_NORMAL)
        
        try:
            while True:
                self.frame_count += 1
                frames = self.pipe.wait_for_frames()
                aligned = self.align.process(frames)
                
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                
                if not depth_frame or not color_frame:
                    continue
                
                # Apply enhanced filters
                filtered_depth = depth_frame
                filtered_depth = self.dec_filter.process(filtered_depth)
                filtered_depth = self.thr_filter.process(filtered_depth)
                filtered_depth = self.d2d.process(filtered_depth)
                filtered_depth = self.spat_filter.process(filtered_depth)
                filtered_depth = self.temp_filter.process(filtered_depth)
                filtered_depth = rs.disparity_transform(False).process(filtered_depth)
                filtered_depth = self.fill_holes.process(filtered_depth)
                
                # Convert to point cloud with enhanced outlier removal
                depth_data = np.asarray(filtered_depth.get_data(), dtype=float) * self.d_scale
                points = self.depth_to_points(depth_data)
                
                if len(points) == 0:
                    continue
                
                # Detect ground plane with enhanced stability
                self.plane = self.enhanced_ground_detection(points)
                
                # Detect obstacles with persistence tracking
                obstacles = self.detect_obstacles(points, self.plane)
                
                # Generate smoothed 48-stimulus matrix
                stimulus_matrix = self.generate_48_stimulus_matrix(points, self.plane)
                
                # Generate traditional feedback for comparison
                haptic_intensity, haptic_frequency = self.generate_tactile_pattern(obstacles)
                audio_cue = self.generate_audio_cues(obstacles)
                
                # Output smoothed 48-stimulus matrix
                if np.any(stimulus_matrix > 0):
                    # Print matrix in a readable format
                    matrix_2d = stimulus_matrix.reshape(3, 16)
                    print(f"\n[Frame {self.frame_count}] SMOOTHED 48-STIMULUS MATRIX:")
                    print("Immediate (0-0.8m):", matrix_2d[0].astype(int))
                    print("Close (0.8-1.5m)  :", matrix_2d[1].astype(int))
                    print("Moderate (1.5-3.0m):", matrix_2d[2].astype(int))
                    
                    # Show max intensity and smoothing status
                    max_intensity = np.max(stimulus_matrix)
                    print(f"Max intensity: {int(max_intensity)}, History frames: {len(self.matrix_history)}")
                
                # Output additional feedback
                if np.any(haptic_intensity > 0.5):
                    print(f"5-Zone HAPTIC: {haptic_intensity.round(2)}")
                
                if audio_cue != "Path clear":
                    print(f"AUDIO: {audio_cue}")
                
                # Visualization
                color_image = np.asarray(color_frame.get_data())
                
                # Show all visualizations
                nav_img = self.visualize_navigation_aid(color_image, obstacles)
                stimulus_vis = self.visualize_48_stimulus_matrix(stimulus_matrix)
                enhanced_vis = self.visualize_48_stimulus_matrix_enhanced(stimulus_matrix)
                
                cv2.imshow('Navigation Aid', nav_img)
                cv2.imshow('48-Stimulus Matrix', stimulus_vis)
                cv2.imshow('Enhanced Matrix', enhanced_vis)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.plane = None
                    print("Ground plane reset")
                elif key == ord('s'):
                    # Reset smoothing buffers
                    self.matrix_history.clear()
                    self.last_smooth_matrix = np.zeros(48)
                    self.obstacle_persistence.clear()
                    print("Smoothing buffers reset")
                    
        except KeyboardInterrupt:
            print("\nProgram interrupted by user")
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            self.pipe.stop()
            cv2.destroyAllWindows()
            print("Camera stopped and windows closed")

if __name__ == "__main__":
    try:
        system = TactileNavigationSystem()
        system.run()
    except Exception as e:
        print(f"Failed to start system: {e}")
    finally:
        print("System shutdown complete")