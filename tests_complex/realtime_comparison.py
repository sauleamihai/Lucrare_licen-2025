#!/usr/bin/env python3
"""
Real-Time Performance Analyzer for Ground Plane Detection Algorithms
Specialized tests for real-time scenarios with frame rate analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import queue
from collections import deque
import psutil
import gc
from dataclasses import dataclass
from typing import List, Dict, Tuple
import seaborn as sns

@dataclass
class FrameResult:
    """Result for a single frame processing"""
    frame_id: int
    algorithm: str
    processing_time: float
    accuracy: float
    memory_used: float
    cpu_usage: float
    success: bool
    
@dataclass 
class RealTimeMetrics:
    """Real-time performance metrics"""
    algorithm: str
    avg_fps: float
    min_fps: float
    max_fps: float
    fps_std: float
    avg_latency: float
    max_latency: float
    frame_drops: int
    memory_efficiency: float
    cpu_efficiency: float
    stability_score: float

class RealTimeSimulator:
    """Simulates real-time processing conditions"""
    
    def __init__(self, target_fps=30, buffer_size=100):
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.buffer_size = buffer_size
        self.frame_buffer = queue.Queue(maxsize=buffer_size)
        self.results_buffer = deque(maxlen=1000)
        self.running = False
        
    def generate_frame_data(self, frame_id):
        """Generate synthetic frame data"""
        # Simulate varying conditions
        if frame_id % 100 < 20:  # 20% of frames have obstacles
            points = SyntheticDataGenerator.add_obstacles(
                SyntheticDataGenerator.generate_ground_plane(num_points=800)
            )
        elif frame_id % 100 < 40:  # 20% have high noise
            points = SyntheticDataGenerator.generate_ground_plane(
                num_points=800, noise_std=0.08
            )
        elif frame_id % 100 < 50:  # 10% have outliers
            points = SyntheticDataGenerator.add_outliers(
                SyntheticDataGenerator.generate_ground_plane(num_points=800), 0.2
            )
        else:  # 50% normal conditions
            points = SyntheticDataGenerator.generate_ground_plane(num_points=800)
        
        return points
    
    def process_frame_threaded(self, algorithm_func, algorithm_name, frame_data, frame_id):
        """Process a single frame and measure performance"""
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        start_cpu = psutil.cpu_percent()
        
        try:
            # Process the frame
            result, exec_time, conv_rate = algorithm_func(frame_data)
            
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            end_cpu = psutil.cpu_percent()
            
            # Calculate metrics
            processing_time = end_time - start_time
            memory_used = end_memory - start_memory
            cpu_usage = (start_cpu + end_cpu) / 2
            
            # Evaluate accuracy (simplified - assume ground plane at y=0)
            true_plane = np.array([0, 1, 0, 0])
            accuracy = self.evaluate_plane_accuracy(result, true_plane) if result is not None else 0.0
            
            return FrameResult(
                frame_id=frame_id,
                algorithm=algorithm_name,
                processing_time=processing_time,
                accuracy=accuracy,
                memory_used=memory_used,
                cpu_usage=cpu_usage,
                success=result is not None
            )
            
        except Exception as e:
            return FrameResult(
                frame_id=frame_id,
                algorithm=algorithm_name,
                processing_time=float('inf'),
                accuracy=0.0,
                memory_used=0.0,
                cpu_usage=0.0,
                success=False
            )
    
    def evaluate_plane_accuracy(self, estimated_plane, true_plane):
        """Evaluate plane accuracy"""
        if estimated_plane is None or true_plane is None:
            return 0.0
        
        est_norm = estimated_plane[:3] / np.linalg.norm(estimated_plane[:3])
        true_norm = true_plane[:3] / np.linalg.norm(true_plane[:3])
        
        dot_product = np.clip(np.abs(np.dot(est_norm, true_norm)), 0, 1)
        angle_diff = np.arccos(dot_product)
        
        return 1.0 - (angle_diff / (np.pi / 2))
    
    def simulate_real_time_processing(self, algorithm_func, algorithm_name, num_frames=1000):
        """Simulate real-time processing for an algorithm"""
        results = []
        frame_times = []
        dropped_frames = 0
        
        print(f"Simulating real-time processing for {algorithm_name}...")
        
        for frame_id in range(num_frames):
            frame_start = time.perf_counter()
            
            # Generate frame data
            frame_data = self.generate_frame_data(frame_id)
            
            # Process frame
            frame_result = self.process_frame_threaded(
                algorithm_func, algorithm_name, frame_data, frame_id
            )
            
            frame_end = time.perf_counter()
            frame_time = frame_end - frame_start
            frame_times.append(frame_time)
            
            # Check if frame processing exceeded target interval
            if frame_time > self.frame_interval:
                dropped_frames += 1
            
            results.append(frame_result)
            
            # Simulate frame rate (sleep if processing was faster than target)
            remaining_time = self.frame_interval - frame_time
            if remaining_time > 0:
                time.sleep(remaining_time)
            
            # Progress indicator
            if frame_id % 100 == 0:
                print(f"  Processed {frame_id}/{num_frames} frames...")
        
        return results, frame_times, dropped_frames

class PerformanceAnalyzer:
    """Analyzes real-time performance metrics"""
    
    def __init__(self):
        self.algorithms = {
            'IRLS': self.irls_wrapper,
            'RANSAC': self.ransac_wrapper,
            'V-Disparity': self.vdisparity_wrapper
        }
    
    def irls_wrapper(self, points):
        """Wrapper for IRLS algorithm"""
        return AlgorithmImplementations.irls_plane_fit(points, max_iterations=8)
    
    def ransac_wrapper(self, points):
        """Wrapper for RANSAC algorithm"""
        return AlgorithmImplementations.ransac_plane_fit(points, max_iterations=40)
    
    def vdisparity_wrapper(self, points):
        """Wrapper for V-Disparity algorithm"""
        return AlgorithmImplementations.vdisparity_ground_detection(points)
    
    def calculate_real_time_metrics(self, results, frame_times, dropped_frames, target_fps=30):
        """Calculate comprehensive real-time metrics"""
        if not results:
            return None
        
        algorithm_name = results[0].algorithm
        processing_times = [r.processing_time for r in results if r.success]
        accuracies = [r.accuracy for r in results if r.success]
        memory_usage = [r.memory_used for r in results]
        cpu_usage = [r.cpu_usage for r in results]
        
        if not processing_times:
            return None
        
        # FPS calculations
        actual_fps = [1.0 / max(ft, 1e-6) for ft in frame_times]
        avg_fps = np.mean(actual_fps)
        min_fps = np.min(actual_fps)
        max_fps = np.max(actual_fps)
        fps_std = np.std(actual_fps)
        
        # Latency calculations
        avg_latency = np.mean(processing_times) * 1000  # ms
        max_latency = np.max(processing_times) * 1000  # ms
        
        # Efficiency calculations
        memory_efficiency = 1.0 / (np.mean(np.abs(memory_usage)) + 0.1)
        cpu_efficiency = 1.0 / (np.mean(cpu_usage) + 0.1)
        
        # Stability score (based on FPS consistency and accuracy consistency)
        fps_stability = 1.0 / (fps_std + 0.1)
        accuracy_stability = 1.0 / (np.std(accuracies) + 0.01)
        stability_score = (fps_stability + accuracy_stability) / 2
        
        return RealTimeMetrics(
            algorithm=algorithm_name,
            avg_fps=avg_fps,
            min_fps=min_fps,
            max_fps=max_fps,
            fps_std=fps_std,
            avg_latency=avg_latency,
            max_latency=max_latency,
            frame_drops=dropped_frames,
            memory_efficiency=memory_efficiency,
            cpu_efficiency=cpu_efficiency,
            stability_score=stability_score
        )
    
    def run_comprehensive_real_time_tests(self, num_frames=2000):
        """Run comprehensive real-time tests"""
        simulator = RealTimeSimulator(target_fps=30)
        all_metrics = {}
        all_results = {}
        
        for alg_name, alg_func in self.algorithms.items():
            print(f"\n{'='*60}")
            print(f"Testing {alg_name} Algorithm")
            print(f"{'='*60}")
            
            # Run simulation
            results, frame_times, dropped_frames = simulator.simulate_real_time_processing(
                alg_func, alg_name, num_frames
            )
            
            # Calculate metrics
            metrics = self.calculate_real_time_metrics(results, frame_times, dropped_frames)
            
            all_metrics[alg_name] = metrics
            all_results[alg_name] = (results, frame_times, dropped_frames)
            
            # Print immediate results
            if metrics:
                print(f"Average FPS: {metrics.avg_fps:.2f}")
                print(f"Frame Drops: {metrics.frame_drops}")
                print(f"Average Latency: {metrics.avg_latency:.2f} ms")
                print(f"Stability Score: {metrics.stability_score:.3f}")
        
        return all_metrics, all_results
    
    def create_real_time_visualizations(self, metrics, detailed_results):
        """Create comprehensive real-time visualizations"""
        fig = plt.figure(figsize=(20, 15))
        
        # 1. FPS Comparison
        ax1 = plt.subplot(3, 3, 1)
        algorithms = list(metrics.keys())
        avg_fps = [metrics[alg].avg_fps for alg in algorithms]
        min_fps = [metrics[alg].min_fps for alg in algorithms]
        max_fps = [metrics[alg].max_fps for alg in algorithms]
        
        x = np.arange(len(algorithms))
        width = 0.25
        
        bars1 = ax1.bar(x - width, avg_fps, width, label='Average FPS', alpha=0.8)
        bars2 = ax1.bar(x, min_fps, width, label='Min FPS', alpha=0.8)
        bars3 = ax1.bar(x + width, max_fps, width, label='Max FPS', alpha=0.8)
        
        ax1.set_ylabel('Frames Per Second')
        ax1.set_title('FPS Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(algorithms)
        ax1.legend()
        ax1.axhline(y=30, color='red', linestyle='--', alpha=0.7, label='30 FPS Target')
        ax1.grid(True, alpha=0.3)
        
        # 2. Latency Comparison
        ax2 = plt.subplot(3, 3, 2)
        avg_latency = [metrics[alg].avg_latency for alg in algorithms]
        max_latency = [metrics[alg].max_latency for alg in algorithms]
        
        bars1 = ax2.bar(x - width/2, avg_latency, width, label='Average Latency', alpha=0.8)
        bars2 = ax2.bar(x + width/2, max_latency, width, label='Max Latency', alpha=0.8)
        
        ax2.set_ylabel('Latency (ms)')
        ax2.set_title('Processing Latency Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(algorithms)
        ax2.legend()
        ax2.axhline(y=33.33, color='red', linestyle='--', alpha=0.7, label='33ms (30 FPS)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Frame Drops
        ax3 = plt.subplot(3, 3, 3)
        frame_drops = [metrics[alg].frame_drops for alg in algorithms]
        bars = ax3.bar(algorithms, frame_drops, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax3.set_ylabel('Dropped Frames')
        ax3.set_title('Frame Drop Comparison')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, drops in zip(bars, frame_drops):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{drops}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Stability Score
        ax4 = plt.subplot(3, 3, 4)
        stability_scores = [metrics[alg].stability_score for alg in algorithms]
        bars = ax4.bar(algorithms, stability_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax4.set_ylabel('Stability Score')
        ax4.set_title('Processing Stability')
        ax4.grid(True, alpha=0.3)
        
        # 5. FPS Over Time for each algorithm
        ax5 = plt.subplot(3, 3, 5)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for i, alg in enumerate(algorithms):
            results, frame_times, _ = detailed_results[alg]
            fps_over_time = [1.0 / max(ft, 1e-6) for ft in frame_times[:200]]  # First 200 frames
            ax5.plot(fps_over_time, label=alg, color=colors[i], alpha=0.7)
        
        ax5.set_xlabel('Frame Number')
        ax5.set_ylabel('FPS')
        ax5.set_title('FPS Over Time (First 200 Frames)')
        ax5.legend()
        ax5.axhline(y=30, color='red', linestyle='--', alpha=0.5)
        ax5.grid(True, alpha=0.3)
        
        # 6. Processing Time Distribution
        ax6 = plt.subplot(3, 3, 6)
        for i, alg in enumerate(algorithms):
            results, _, _ = detailed_results[alg]
            processing_times = [r.processing_time * 1000 for r in results if r.success]  # ms
            ax6.hist(processing_times, bins=30, alpha=0.6, label=alg, color=colors[i])
        
        ax6.set_xlabel('Processing Time (ms)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Processing Time Distribution')
        ax6.legend()
        ax6.axvline(x=33.33, color='red', linestyle='--', alpha=0.7)
        ax6.grid(True, alpha=0.3)
        
        # 7. Memory Usage
        ax7 = plt.subplot(3, 3, 7)
        memory_effs = [metrics[alg].memory_efficiency for alg in algorithms]
        bars = ax7.bar(algorithms, memory_effs, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax7.set_ylabel('Memory Efficiency')
        ax7.set_title('Memory Efficiency Comparison')
        ax7.grid(True, alpha=0.3)
        
        # 8. CPU Usage
        ax8 = plt.subplot(3, 3, 8)
        cpu_effs = [metrics[alg].cpu_efficiency for alg in algorithms]
        bars = ax8.bar(algorithms, cpu_effs, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        ax8.set_ylabel('CPU Efficiency')
        ax8.set_title('CPU Efficiency Comparison')
        ax8.grid(True, alpha=0.3)
        
        # 9. Real-Time Suitability Radar Chart
        ax9 = plt.subplot(3, 3, 9, projection='polar')
        
        # Normalize metrics for radar chart
        categories = ['FPS', 'Latency', 'Stability', 'Memory', 'CPU']
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, alg in enumerate(algorithms):
            m = metrics[alg]
            values = [
                min(m.avg_fps / 30, 1.0),  # Normalize to 30 FPS
                min(33.33 / m.avg_latency, 1.0),  # Invert latency (lower is better)
                min(m.stability_score / 5.0, 1.0),  # Normalize stability
                min(m.memory_efficiency / 5.0, 1.0),  # Normalize memory efficiency
                min(m.cpu_efficiency / 5.0, 1.0)  # Normalize CPU efficiency
            ]
            values += values[:1]  # Complete the circle
            
            ax9.plot(angles, values, 'o-', linewidth=2, label=alg, color=colors[i])
            ax9.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax9.set_xticks(angles[:-1])
        ax9.set_xticklabels(categories)
        ax9.set_ylim(0, 1)
        ax9.set_title('Real-Time Suitability Radar', pad=20)
        ax9.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax9.grid(True)
        
        plt.tight_layout()
        return fig
    
    def generate_real_time_report(self, metrics):
        """Generate detailed real-time performance report"""
        report = []
        report.append("=" * 80)
        report.append("REAL-TIME PERFORMANCE ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Sort algorithms by overall real-time performance
        def real_time_score(m):
            fps_score = min(m.avg_fps / 30, 1.0) * 0.4
            latency_score = min(33.33 / m.avg_latency, 1.0) * 0.3
            stability_score = min(m.stability_score / 5.0, 1.0) * 0.2
            efficiency_score = (m.memory_efficiency + m.cpu_efficiency) / 10.0 * 0.1
            return fps_score + latency_score + stability_score + efficiency_score
        
        sorted_algorithms = sorted(metrics.items(), key=lambda x: real_time_score(x[1]), reverse=True)
        
        report.append("REAL-TIME PERFORMANCE RANKING:")
        report.append("-" * 40)
        for rank, (alg, m) in enumerate(sorted_algorithms, 1):
            score = real_time_score(m)
            report.append(f"{rank}. {alg}: {score:.4f}")
        report.append("")
        
        # Detailed metrics for each algorithm
        for alg, m in sorted_algorithms:
            report.append(f"{alg} ALGORITHM - DETAILED METRICS:")
            report.append("-" * 50)
            report.append(f"  Frame Rate Performance:")
            report.append(f"    Average FPS: {m.avg_fps:.2f}")
            report.append(f"    Minimum FPS: {m.min_fps:.2f}")
            report.append(f"    Maximum FPS: {m.max_fps:.2f}")
            report.append(f"    FPS Standard Deviation: {m.fps_std:.2f}")
            report.append(f"")
            report.append(f"  Latency Performance:")
            report.append(f"    Average Latency: {m.avg_latency:.2f} ms")
            report.append(f"    Maximum Latency: {m.max_latency:.2f} ms")
            report.append(f"")
            report.append(f"  Reliability:")
            report.append(f"    Frame Drops: {m.frame_drops}")
            report.append(f"    Stability Score: {m.stability_score:.3f}")
            report.append(f"")
            report.append(f"  Resource Efficiency:")
            report.append(f"    Memory Efficiency: {m.memory_efficiency:.3f}")
            report.append(f"    CPU Efficiency: {m.cpu_efficiency:.3f}")
            report.append(f"")
            
            # Real-time suitability assessment
            if m.avg_fps >= 30 and m.frame_drops < 10:
                suitability = "EXCELLENT for real-time applications"
            elif m.avg_fps >= 20 and m.frame_drops < 50:
                suitability = "GOOD for real-time applications"
            elif m.avg_fps >= 15:
                suitability = "ACCEPTABLE for some real-time applications"
            else:
                suitability = "NOT SUITABLE for real-time applications"
            
            report.append(f"  Real-Time Suitability: {suitability}")
            report.append("")
        
        # Specific recommendations
        report.append("=" * 80)
        report.append("RECOMMENDATIONS FOR REAL-TIME IMPLEMENTATION:")
        report.append("=" * 80)
        
        best_fps = max(metrics.values(), key=lambda m: m.avg_fps)
        lowest_latency = min(metrics.values(), key=lambda m: m.avg_latency)
        most_stable = max(metrics.values(), key=lambda m: m.stability_score)
        
        report.append(f"• Highest FPS: {best_fps.algorithm} ({best_fps.avg_fps:.1f} FPS)")
        report.append(f"• Lowest Latency: {lowest_latency.algorithm} ({lowest_latency.avg_latency:.1f} ms)")
        report.append(f"• Most Stable: {most_stable.algorithm} (score: {most_stable.stability_score:.3f})")
        report.append("")
        report.append("Implementation Guidelines:")
        report.append("1. For applications requiring consistent 30+ FPS:")
        for alg, m in metrics.items():
            if m.avg_fps >= 30 and m.frame_drops < 10:
                report.append(f"   → Use {alg}")
        report.append("")
        report.append("2. For latency-critical applications (<33ms):")
        for alg, m in metrics.items():
            if m.avg_latency < 33:
                report.append(f"   → Use {alg}")
        report.append("")
        report.append("3. For resource-constrained environments:")
        best_efficiency = max(metrics.values(), key=lambda m: m.memory_efficiency + m.cpu_efficiency)
        report.append(f"   → Use {best_efficiency.algorithm}")
        report.append("")
        report.append("4. For noisy/challenging environments:")
        report.append(f"   → Use {most_stable.algorithm} (highest stability)")
        
        return "\n".join(report)

# Import the classes from the main test suite
class SyntheticDataGenerator:
    """Generate synthetic point clouds for testing"""
    
    @staticmethod
    def generate_ground_plane(width=10, depth=10, num_points=1000, noise_std=0.02):
        x = np.random.uniform(-width/2, width/2, num_points)
        z = np.random.uniform(0, depth, num_points)
        y = np.random.normal(0, noise_std, num_points)
        return np.column_stack([x, y, z])
    
    @staticmethod
    def generate_tilted_plane(width=10, depth=10, num_points=1000, 
                            tilt_x=0.1, tilt_z=0.05, noise_std=0.02):
        x = np.random.uniform(-width/2, width/2, num_points)
        z = np.random.uniform(0, depth, num_points)
        y = tilt_x * x + tilt_z * z + np.random.normal(0, noise_std, num_points)
        return np.column_stack([x, y, z])
    
    @staticmethod
    def add_obstacles(points, num_obstacles=5, obstacle_height=0.5):
        obstacles = []
        for _ in range(num_obstacles):
            center_x = np.random.uniform(-4, 4)
            center_z = np.random.uniform(1, 8)
            size = np.random.uniform(0.2, 0.8)
            
            obs_points = np.random.randint(50, 200)
            x = np.random.normal(center_x, size, obs_points)
            z = np.random.normal(center_z, size, obs_points)
            y = np.random.uniform(0.1, obstacle_height, obs_points)
            
            obstacles.append(np.column_stack([x, y, z]))
        
        if obstacles:
            all_obstacles = np.vstack(obstacles)
            return np.vstack([points, all_obstacles])
        return points
    
    @staticmethod
    def add_outliers(points, outlier_ratio=0.1):
        num_outliers = int(len(points) * outlier_ratio)
        if num_outliers == 0:
            return points
        
        outliers = np.random.uniform(-10, 10, (num_outliers, 3))
        outliers[:, 1] = np.random.uniform(-2, 3, num_outliers)
        
        return np.vstack([points, outliers])

class AlgorithmImplementations:
    """Algorithm implementations for testing"""
    
    @staticmethod
    def irls_plane_fit(pts, max_iterations=10, tukey_c=0.1):
        start_time = time.perf_counter()
        
        N = pts.shape[0]
        if N < 3:
            return None, time.perf_counter() - start_time, 0
        
        w = np.ones(N, dtype=float)
        plane = None
        
        for iteration in range(max_iterations):
            w_sum = np.sum(w)
            if w_sum < 1e-6:
                break
                
            mu = (w.reshape(-1, 1) * pts).sum(axis=0) / w_sum
            diffs = pts - mu
            C = (w.reshape(-1, 1) * diffs).T @ diffs
            
            try:
                _, _, Vt = np.linalg.svd(C)
                n = Vt[-1, :]
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-6:
                    break
                n = n / norm_n
            except np.linalg.LinAlgError:
                break
            
            D = -np.dot(n, mu)
            r = (pts @ n) + D
            
            abs_r = np.abs(r)
            mask = abs_r < tukey_c
            w_new = np.zeros_like(w)
            w_new[mask] = (1 - (r[mask] / tukey_c) ** 2) ** 2
            w = w_new
            
            plane = np.array([n[0], n[1], n[2], D], dtype=float)
        
        return plane, time.perf_counter() - start_time, max_iterations
    
    @staticmethod
    def ransac_plane_fit(pts, max_iterations=60, tolerance=0.1):
        start_time = time.perf_counter()
        
        best_plane = None
        best_inliers = 0
        N = pts.shape[0]
        
        for iteration in range(max_iterations):
            if N < 3:
                break
                
            try:
                idx = np.random.choice(N, 3, replace=False)
                sample = pts[idx]
            except ValueError:
                break
            
            v1 = sample[1] - sample[0]
            v2 = sample[2] - sample[0]
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            
            if norm_n < 1e-6:
                continue
                
            A, B, C = n / norm_n
            D = -np.dot(n / norm_n, sample[0])
            
            dists = np.abs((pts @ (n / norm_n)) + D)
            inliers = np.sum(dists < tolerance)
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = np.array([A, B, C, D], dtype=float)
        
        return best_plane, time.perf_counter() - start_time, max_iterations
    
    @staticmethod
    def vdisparity_ground_detection(pts, fx=400, baseline=0.05):
        start_time = time.perf_counter()
        
        if len(pts) == 0:
            return None, time.perf_counter() - start_time, 0
        
        Z = pts[:, 2]
        Y = pts[:, 1]
        
        valid_mask = (Z > 0.1) & (Z < 10)
        if not np.any(valid_mask):
            return None, time.perf_counter() - start_time, 0
        
        valid_pts = pts[valid_mask]
        Z_valid = valid_pts[:, 2]
        Y_valid = valid_pts[:, 1]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            disparity = (fx * baseline) / Z_valid
            disparity = disparity[np.isfinite(disparity)]
        
        if len(disparity) == 0:
            return None, time.perf_counter() - start_time, 0
        
        try:
            valid_y = Y_valid[np.isfinite(disparity)]
            if len(valid_y) < 3:
                return None, time.perf_counter() - start_time, 0
            
            A = np.vstack([disparity, np.ones(len(disparity))]).T
            m, c = np.linalg.lstsq(A, valid_y, rcond=None)[0]
            
            plane = np.array([0, 1, m, c], dtype=float)
            
        except (np.linalg.LinAlgError, ValueError):
            return None, time.perf_counter() - start_time, 0
        
        return plane, time.perf_counter() - start_time, 1

def main():
    """Main function to run real-time performance tests"""
    print("Real-Time Algorithm Performance Analyzer")
    print("=" * 60)
    print("Testing algorithms under real-time constraints...")
    
    # Initialize analyzer
    analyzer = PerformanceAnalyzer()
    
    # Run comprehensive real-time tests
    print("\nRunning real-time simulation tests...")
    metrics, detailed_results = analyzer.run_comprehensive_real_time_tests(num_frames=2000)
    
    # Create visualizations
    print("\nGenerating real-time performance visualizations...")
    fig = analyzer.create_real_time_visualizations(metrics, detailed_results)
    fig.savefig('real_time_performance_analysis.png', dpi=300, bbox_inches='tight')
    
    # Generate report
    print("\nGenerating detailed performance report...")
    report = analyzer.generate_real_time_report(metrics)
    
    with open('real_time_performance_report.txt', 'w') as f:
        f.write(report)
    
    # Print summary
    print("\n" + "="*60)
    print("REAL-TIME PERFORMANCE SUMMARY")
    print("="*60)
    
    for alg_name, metric in metrics.items():
        if metric:
            print(f"\n{alg_name}:")
            print(f"  Average FPS: {metric.avg_fps:.2f}")
            print(f"  Average Latency: {metric.avg_latency:.2f} ms")
            print(f"  Frame Drops: {metric.frame_drops}")
            print(f"  Stability Score: {metric.stability_score:.3f}")
            
            # Real-time verdict
            if metric.avg_fps >= 30 and metric.frame_drops < 10:
                verdict = "✅ EXCELLENT for real-time"
            elif metric.avg_fps >= 20 and metric.frame_drops < 50:
                verdict = "✅ GOOD for real-time"
            elif metric.avg_fps >= 15:
                verdict = "⚠️  ACCEPTABLE for some real-time apps"
            else:
                verdict = "❌ NOT suitable for real-time"
            
            print(f"  Real-time Verdict: {verdict}")
    
    # Best algorithm recommendation
    best_overall = max(metrics.items(), 
                      key=lambda x: x[1].avg_fps if x[1] else 0)
    
    print(f"\n🏆 RECOMMENDED FOR REAL-TIME: {best_overall[0]}")
    print(f"   → {best_overall[1].avg_fps:.1f} FPS average")
    print(f"   → {best_overall[1].avg_latency:.1f} ms latency")
    print(f"   → {best_overall[1].frame_drops} frame drops")
    
    print(f"\nDetailed analysis saved to: real_time_performance_report.txt")
    print("Visualization saved as: real_time_performance_analysis.png")
    
    # Show plots
    plt.show()
    
    return metrics, detailed_results

if __name__ == "__main__":
    np.random.seed(42)  # For reproducible results
    metrics, results = main()