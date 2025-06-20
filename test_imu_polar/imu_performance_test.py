#!/usr/bin/env python3
"""
Performance Test Suite: IMU vs Non-IMU Plane Detection
Tests accuracy, speed, stability, and robustness of both implementations
"""

import math
import argparse
import sys
import time
import numpy as np
import pyrealsense2 as rs
import cv2
import json
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from datetime import datetime
import threading
import queue
from scipy.spatial.transform import Rotation
from scipy import stats

# Import the original detector (assuming it's in the same directory)
# You'll need to modify this import based on your file structure
# from original_detector import FastHybridPlaneDetector as OriginalDetector
# from original_detector import TemporalObstacleFilter as OriginalFilter

# Test configuration
TEST_DURATION = 120  # seconds
STABILITY_TEST_FRAMES = 300
ACCURACY_TEST_SAMPLES = 50
PERFORMANCE_SAMPLES = 1000

# Test scenarios
TEST_SCENARIOS = {
    'static': 'Camera remains stationary',
    'tilt_slow': 'Slow tilting movements',
    'tilt_fast': 'Fast tilting movements', 
    'rotation': 'Rotational movements',
    'translation': 'Translation movements',
    'vibration': 'Small vibrations and shakes',
    'mixed': 'Mixed movement patterns'
}

class PerformanceMetrics:
    """Tracks and analyzes performance metrics"""
    
    def __init__(self, name):
        self.name = name
        self.detection_times = []
        self.plane_estimates = []
        self.confidence_scores = []
        self.method_usage = defaultdict(int)
        self.frame_timestamps = []
        self.imu_data = []
        self.stability_scores = []
        self.convergence_times = []
        
        # Error tracking
        self.detection_failures = 0
        self.total_frames = 0
        
        # Plane stability metrics
        self.plane_history = deque(maxlen=30)
        self.angle_variations = []
        self.position_variations = []
        
    def add_detection(self, plane, detection_time, confidence, method, timestamp, imu_info=None):
        """Add a detection result"""
        self.total_frames += 1
        
        if plane is not None:
            self.plane_estimates.append(plane.copy())
            self.detection_times.append(detection_time)
            self.confidence_scores.append(confidence)
            self.method_usage[method] += 1
            self.frame_timestamps.append(timestamp)
            
            if imu_info:
                self.imu_data.append(imu_info.copy())
            
            # Track stability
            self._update_stability(plane)
            
        else:
            self.detection_failures += 1
    
    def _update_stability(self, plane):
        """Update stability metrics"""
        if len(self.plane_history) > 0:
            # Angular stability
            last_plane = self.plane_history[-1]
            angle_diff = self._compute_plane_angle_diff(plane, last_plane)
            self.angle_variations.append(angle_diff)
            
            # Position stability (distance difference)
            pos_diff = abs(plane[3] - last_plane[3])
            self.position_variations.append(pos_diff)
            
            # Overall stability score
            stability = max(0, 1.0 - (angle_diff / 0.2 + pos_diff / 0.1))
            self.stability_scores.append(stability)
        
        self.plane_history.append(plane.copy())
    
    def _compute_plane_angle_diff(self, plane1, plane2):
        """Compute angle difference between plane normals"""
        n1 = plane1[:3] / np.linalg.norm(plane1[:3])
        n2 = plane2[:3] / np.linalg.norm(plane2[:3])
        cos_angle = np.clip(abs(np.dot(n1, n2)), -1, 1)
        return np.arccos(cos_angle)
    
    def get_summary_stats(self):
        """Get comprehensive performance summary"""
        if not self.detection_times:
            return {"error": "No successful detections"}
        
        stats = {
            'name': self.name,
            'total_frames': self.total_frames,
            'successful_detections': len(self.detection_times),
            'success_rate': len(self.detection_times) / self.total_frames if self.total_frames > 0 else 0,
            'detection_failures': self.detection_failures,
            
            # Timing performance
            'avg_detection_time': np.mean(self.detection_times),
            'min_detection_time': np.min(self.detection_times),
            'max_detection_time': np.max(self.detection_times),
            'std_detection_time': np.std(self.detection_times),
            'median_detection_time': np.median(self.detection_times),
            
            # Method usage
            'method_distribution': dict(self.method_usage),
            'primary_method': max(self.method_usage.items(), key=lambda x: x[1])[0] if self.method_usage else 'none',
            
            # Stability metrics
            'avg_stability': np.mean(self.stability_scores) if self.stability_scores else 0,
            'angle_stability_std': np.std(self.angle_variations) if self.angle_variations else 0,
            'position_stability_std': np.std(self.position_variations) if self.position_variations else 0,
            
            # Confidence metrics
            'avg_confidence': np.mean(self.confidence_scores) if self.confidence_scores else 0,
            'confidence_std': np.std(self.confidence_scores) if self.confidence_scores else 0,
        }
        
        # Add IMU-specific metrics if available
        if self.imu_data:
            imu_confidences = [data.get('confidence', 0) for data in self.imu_data]
            tilt_angles = [data.get('tilt_angle', 0) for data in self.imu_data]
            
            stats.update({
                'avg_imu_confidence': np.mean(imu_confidences),
                'avg_tilt_angle': np.mean(tilt_angles),
                'max_tilt_angle': np.max(tilt_angles),
                'imu_ready_rate': sum(1 for data in self.imu_data if data.get('ready', False)) / len(self.imu_data)
            })
        
        return stats

class PerformanceTestSuite:
    """Main test suite for comparing IMU vs Non-IMU performance"""
    
    def __init__(self):
        self.results = {}
        self.test_start_time = None
        self.current_scenario = None
        
    def run_comprehensive_test(self, test_frames=500, scenarios=None):
        """Run comprehensive performance test"""
        print("=== IMU vs Non-IMU Performance Test Suite ===")
        print(f"Test duration: {test_frames} frames per scenario")
        print(f"Scenarios: {len(scenarios or TEST_SCENARIOS)}")
        print("=" * 50)
        
        scenarios = scenarios or TEST_SCENARIOS
        
        for scenario_name, scenario_desc in scenarios.items():
            print(f"\nTesting scenario: {scenario_name.upper()}")
            print(f"Description: {scenario_desc}")
            print("-" * 40)
            
            # Test with IMU
            print("Testing WITH IMU...")
            imu_metrics = self._run_scenario_test(scenario_name, test_frames, use_imu=True)
            
            # Test without IMU
            print("Testing WITHOUT IMU...")
            no_imu_metrics = self._run_scenario_test(scenario_name, test_frames, use_imu=False)
            
            # Store results
            self.results[scenario_name] = {
                'with_imu': imu_metrics.get_summary_stats(),
                'without_imu': no_imu_metrics.get_summary_stats(),
                'comparison': self._compare_metrics(imu_metrics, no_imu_metrics)
            }
            
            # Print immediate comparison
            self._print_scenario_comparison(scenario_name)
        
        # Generate comprehensive report
        self._generate_final_report()
        
        return self.results
    
    def _run_scenario_test(self, scenario_name, test_frames, use_imu=True):
        """Run test for specific scenario"""
        # Setup RealSense
        pipe, profile, imu_available = self._setup_realsense(enable_imu=use_imu)
        
        if use_imu and not imu_available:
            print("Warning: IMU not available, falling back to depth-only mode")
            use_imu = False
        
        # Import and initialize components
        # Note: Make sure to have your IMU-enhanced detector file available
        try:
            from imu_plane_detector import FastHybridPlaneDetector, TemporalObstacleFilter, IMUCalibration
        except ImportError:
            print("Error: Cannot import IMU detector. Make sure 'imu_plane_detector.py' is in the same directory.")
            print("Alternatively, adjust the import path to match your file structure.")
            sys.exit(1)
        
        imu_calibration = IMUCalibration(enable_imu=use_imu) if use_imu else None
        detector = FastHybridPlaneDetector(imu_calibration)
        temporal_filter = TemporalObstacleFilter()
        
        # Get camera parameters
        sensor = profile.get_device().first_depth_sensor()
        d_scale = sensor.get_depth_scale()
        intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        fx, fy, ppx, ppy = intr.fx, intr.fy, intr.ppx, intr.ppy
        
        # Initialize metrics
        mode_name = "IMU" if use_imu else "No-IMU"
        metrics = PerformanceMetrics(f"{scenario_name}_{mode_name}")
        
        # Calibration phase for IMU
        calibration_frames = 0
        if use_imu and imu_calibration:
            print("  Performing IMU calibration...")
            calibration_frames = self._calibrate_imu(pipe, imu_calibration)
            if imu_calibration.is_calibrated:
                print(f"IMU calibrated in {calibration_frames} frames")
            else:
                print("IMU calibration failed")
        
        # Main test loop
        frames_processed = 0
        print(f"  Processing {test_frames} frames...")
        
        # Precompute angular edges
        H_img, W_img = intr.height, intr.width
        FOV = 2 * math.atan((W_img / 2) / fx)
        ang_edges = np.linspace(-FOV / 2, FOV / 2, 9)  # 8 angular bins
        
        while frames_processed < test_frames:
            try:
                frames = pipe.wait_for_frames(timeout_ms=5000)
                frame_timestamp = time.time()
                
                # Process frame
                result = self._process_frame(
                    frames, detector, temporal_filter, imu_calibration,
                    d_scale, fx, fy, ppx, ppy, ang_edges, use_imu
                )
                
                if result:
                    plane, detection_time, confidence, method, imu_info = result
                    metrics.add_detection(
                        plane, detection_time, confidence, method, 
                        frame_timestamp, imu_info
                    )
                
                frames_processed += 1
                
                if frames_processed % 50 == 0:
                    progress = (frames_processed / test_frames) * 100
                    print(f"    Progress: {progress:.1f}%")
                    
            except Exception as e:
                print(f"    Error processing frame: {e}")
                continue
        
        pipe.stop()
        return metrics
    
    def _setup_realsense(self, enable_imu=True):
        """Setup RealSense pipeline"""
        pipe = rs.pipeline()
        cfg = rs.config()
        
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        
        imu_available = False
        if enable_imu:
            try:
                cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
                cfg.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)
                imu_available = True
            except:
                pass
        
        profile = pipe.start(cfg)
        return pipe, profile, imu_available
    
    def _calibrate_imu(self, pipe, imu_calibration, max_frames=100):
        """Calibrate IMU"""
        frames_processed = 0
        
        while frames_processed < max_frames and not imu_calibration.calibration_complete:
            try:
                frames = pipe.wait_for_frames(timeout_ms=5000)
                
                # Get IMU data
                accel_data, gyro_data = self._extract_imu_data(frames)
                if accel_data is not None and gyro_data is not None:
                    imu_calibration.add_calibration_sample(accel_data, gyro_data)
                
                frames_processed += 1
                
            except:
                continue
        
        return frames_processed
    
    def _extract_imu_data(self, frames):
        """Extract IMU data from frames"""
        accel_data = None
        gyro_data = None
        
        try:
            if frames.first_or_default(rs.stream.accel):
                accel_frame = frames.first_or_default(rs.stream.accel)
                motion_data = accel_frame.as_motion_frame().get_motion_data()
                accel_data = np.array([motion_data.x, motion_data.y, motion_data.z])
            
            if frames.first_or_default(rs.stream.gyro):
                gyro_frame = frames.first_or_default(rs.stream.gyro)
                motion_data = gyro_frame.as_motion_frame().get_motion_data()
                gyro_data = np.array([motion_data.x, motion_data.y, motion_data.z])
        except:
            pass
        
        return accel_data, gyro_data
    
    def _process_frame(self, frames, detector, temporal_filter, imu_calibration, 
                      d_scale, fx, fy, ppx, ppy, ang_edges, use_imu):
        """Process single frame and return results"""
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            return None
        
        # Update IMU if available
        imu_info = {}
        if use_imu and imu_calibration and imu_calibration.is_calibrated:
            accel_data, gyro_data = self._extract_imu_data(frames)
            if accel_data is not None and gyro_data is not None:
                current_timestamp = time.time() * 1000
                imu_calibration.update_orientation(accel_data, gyro_data, current_timestamp)
                
                imu_info = {
                    'ready': imu_calibration.is_ready(),
                    'confidence': imu_calibration.get_orientation_confidence(),
                    'tilt_angle': imu_calibration.get_tilt_angle()
                }
        
        # Convert depth to points
        depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
        pts3D = self._depth_to_points(depth_image, fx, fy, ppx, ppy, d_scale)
        
        if pts3D.shape[0] == 0:
            return None
        
        # Select ground points
        Ys = pts3D[:, 1]
        threshold = np.percentile(Ys, 25)
        ground_pts = pts3D[Ys < threshold]
        
        if ground_pts.shape[0] < 50:
            return None
        
        # Detect plane
        start_time = time.time()
        plane = detector.detect_plane(ground_pts)
        detection_time = (time.time() - start_time) * 1000
        
        if plane is None:
            return None
        
        # Compute confidence
        distances = np.abs(pts3D @ plane[:3] + plane[3])
        confidence = np.sum(distances < 0.05) / len(pts3D)
        
        method = detector.method_used
        
        return plane, detection_time, confidence, method, imu_info
    
    def _depth_to_points(self, depth_map, fx, fy, ppx, ppy, d_scale):
        """Convert depth map to 3D points"""
        ys, xs = np.nonzero(depth_map)
        if len(ys) == 0:
            return np.array([]).reshape(0, 3)
        
        zs = depth_map[ys, xs].astype(float) * d_scale
        X = (xs - ppx) * zs / fx
        Y = (ys - ppy) * zs / fy
        return np.vstack((X, Y, zs)).T
    
    def _compare_metrics(self, imu_metrics, no_imu_metrics):
        """Compare metrics between IMU and non-IMU"""
        imu_stats = imu_metrics.get_summary_stats()
        no_imu_stats = no_imu_metrics.get_summary_stats()
        
        if 'error' in imu_stats or 'error' in no_imu_stats:
            return {'error': 'Insufficient data for comparison'}
        
        comparison = {}
        
        # Speed comparison
        speed_improvement = ((no_imu_stats['avg_detection_time'] - imu_stats['avg_detection_time']) / 
                           no_imu_stats['avg_detection_time']) * 100
        comparison['speed_improvement_percent'] = speed_improvement
        
        # Accuracy comparison  
        accuracy_improvement = ((imu_stats['success_rate'] - no_imu_stats['success_rate']) / 
                              no_imu_stats['success_rate']) * 100 if no_imu_stats['success_rate'] > 0 else 0
        comparison['accuracy_improvement_percent'] = accuracy_improvement
        
        # Stability comparison
        stability_improvement = ((imu_stats['avg_stability'] - no_imu_stats['avg_stability']) / 
                               no_imu_stats['avg_stability']) * 100 if no_imu_stats['avg_stability'] > 0 else 0
        comparison['stability_improvement_percent'] = stability_improvement
        
        # Confidence comparison
        confidence_improvement = ((imu_stats['avg_confidence'] - no_imu_stats['avg_confidence']) / 
                                no_imu_stats['avg_confidence']) * 100 if no_imu_stats['avg_confidence'] > 0 else 0
        comparison['confidence_improvement_percent'] = confidence_improvement
        
        # Method diversity
        comparison['imu_method_diversity'] = len(imu_stats['method_distribution'])
        comparison['no_imu_method_diversity'] = len(no_imu_stats['method_distribution'])
        
        return comparison
    
    def _print_scenario_comparison(self, scenario_name):
        """Print comparison for a specific scenario"""
        result = self.results[scenario_name]
        comp = result['comparison']
        
        if 'error' in comp:
            print(f"{comp['error']}")
            return
        
        print(f"\n Results for {scenario_name.upper()}:")
        print(f"     Speed improvement: {comp['speed_improvement_percent']:+.1f}%")
        print(f"     Accuracy improvement: {comp['accuracy_improvement_percent']:+.1f}%") 
        print(f"     Stability improvement: {comp['stability_improvement_percent']:+.1f}%")
        print(f"     Confidence improvement: {comp['confidence_improvement_percent']:+.1f}%")
        
        # Highlight significant improvements
        if comp['speed_improvement_percent'] > 20:
            print(" Significant speed improvement!")
        if comp['stability_improvement_percent'] > 15:
            print("     🎯 Significant stability improvement!")
    
    def _generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n" + "=" * 60)
        print("FINAL PERFORMANCE REPORT")
        print("=" * 60)
        
        # Calculate overall statistics
        overall_speed_improvements = []
        overall_accuracy_improvements = []
        overall_stability_improvements = []
        
        for scenario_name, result in self.results.items():
            comp = result['comparison']
            if 'error' not in comp:
                overall_speed_improvements.append(comp['speed_improvement_percent'])
                overall_accuracy_improvements.append(comp['accuracy_improvement_percent'])
                overall_stability_improvements.append(comp['stability_improvement_percent'])
        
        if overall_speed_improvements:
            print(f"\n📈 OVERALL IMPROVEMENTS (IMU vs Non-IMU):")
            print(f"   Average speed improvement: {np.mean(overall_speed_improvements):+.1f}%")
            print(f"   Average accuracy improvement: {np.mean(overall_accuracy_improvements):+.1f}%")
            print(f"   Average stability improvement: {np.mean(overall_stability_improvements):+.1f}%")
            
            print(f"\nBEST PERFORMANCE SCENARIOS FOR IMU:")
            # Find best scenarios for each metric
            speed_best = max(self.results.items(), 
                           key=lambda x: x[1]['comparison'].get('speed_improvement_percent', -999))
            accuracy_best = max(self.results.items(), 
                              key=lambda x: x[1]['comparison'].get('accuracy_improvement_percent', -999))
            stability_best = max(self.results.items(), 
                               key=lambda x: x[1]['comparison'].get('stability_improvement_percent', -999))
            
            print(f"   Speed: {speed_best[0]} ({speed_best[1]['comparison']['speed_improvement_percent']:+.1f}%)")
            print(f"   Accuracy: {accuracy_best[0]} ({accuracy_best[1]['comparison']['accuracy_improvement_percent']:+.1f}%)")
            print(f"   Stability: {stability_best[0]} ({stability_best[1]['comparison']['stability_improvement_percent']:+.1f}%)")
        
        # Save detailed results
        self._save_results()
        
        print(f"\nDetailed results saved to: performance_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        print("\nPerformance testing completed!")
    
    def _save_results(self):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"performance_test_results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for scenario, data in self.results.items():
            serializable_results[scenario] = {
                'with_imu': self._make_json_serializable(data['with_imu']),
                'without_imu': self._make_json_serializable(data['without_imu']),
                'comparison': self._make_json_serializable(data['comparison'])
            }
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to Python types for JSON"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

def main():
    """Main test execution"""
    parser = argparse.ArgumentParser(description='IMU vs Non-IMU Performance Test')
    parser.add_argument('--frames', type=int, default=200, 
                       help='Number of frames per test scenario')
    parser.add_argument('--scenarios', nargs='+', 
                       choices=list(TEST_SCENARIOS.keys()),
                       default=list(TEST_SCENARIOS.keys()),
                       help='Test scenarios to run')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with fewer frames')
    
    args = parser.parse_args()
    
    if args.quick:
        test_frames = 100
        scenarios = {'static': TEST_SCENARIOS['static'], 
                    'tilt_slow': TEST_SCENARIOS['tilt_slow']}
    else:
        test_frames = args.frames
        scenarios = {k: TEST_SCENARIOS[k] for k in args.scenarios}
    
    # Run the test suite
    test_suite = PerformanceTestSuite()
    results = test_suite.run_comprehensive_test(test_frames, scenarios)
    
    return results

if __name__ == "__main__":
    results = main()
