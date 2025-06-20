#!/usr/bin/env python3
"""
Complete Algorithm Test Suite for Documentation
Comprehensive testing framework for ground plane detection algorithms
Generates publication-ready graphs and detailed documentation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import pandas as pd
import time
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication-ready style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'font.family': 'serif'
})

class DocumentationTestSuite:
    """Comprehensive test suite for generating documentation"""
    
    def __init__(self):
        self.test_results = {}
        self.algorithms = {
            'IRLS': self.irls_algorithm,
            'RANSAC': self.ransac_algorithm,
            'V-Disparity': self.vdisparity_algorithm
        }
        self.colors = {
            'IRLS': '#E74C3C',
            'RANSAC': '#3498DB', 
            'V-Disparity': '#2ECC71'
        }
    
    def irls_algorithm(self, points, **kwargs):
        """IRLS implementation with performance tracking"""
        start_time = time.perf_counter()
        max_iterations = kwargs.get('max_iterations', 10)
        tukey_c = kwargs.get('tukey_c', 0.1)
        
        N = points.shape[0]
        if N < 3:
            return None, time.perf_counter() - start_time, {'iterations': 0, 'inliers': 0}
        
        w = np.ones(N, dtype=float)
        plane = None
        iteration_history = []
        
        for iteration in range(max_iterations):
            w_sum = np.sum(w)
            if w_sum < 1e-6:
                break
            
            # Weighted centroid
            mu = (w.reshape(-1, 1) * points).sum(axis=0) / w_sum
            
            # Weighted covariance
            diffs = points - mu
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
            r = (points @ n) + D
            
            # Store iteration info
            inliers = np.sum(np.abs(r) < tukey_c)
            iteration_history.append({
                'iteration': iteration,
                'inliers': inliers,
                'residual_mean': np.mean(np.abs(r)),
                'residual_std': np.std(np.abs(r))
            })
            
            # Update weights
            abs_r = np.abs(r)
            mask = abs_r < tukey_c
            w_new = np.zeros_like(w)
            w_new[mask] = (1 - (r[mask] / tukey_c) ** 2) ** 2
            w = w_new
            
            plane = np.array([n[0], n[1], n[2], D], dtype=float)
        
        execution_time = time.perf_counter() - start_time
        metadata = {
            'iterations': len(iteration_history),
            'inliers': iteration_history[-1]['inliers'] if iteration_history else 0,
            'convergence_history': iteration_history
        }
        
        return plane, execution_time, metadata
    
    def ransac_algorithm(self, points, **kwargs):
        """RANSAC implementation with performance tracking"""
        start_time = time.perf_counter()
        max_iterations = kwargs.get('max_iterations', 60)
        tolerance = kwargs.get('tolerance', 0.1)
        
        best_plane = None
        best_inliers = 0
        N = points.shape[0]
        iteration_history = []
        
        for iteration in range(max_iterations):
            if N < 3:
                break
            
            try:
                idx = np.random.choice(N, 3, replace=False)
                sample = points[idx]
            except ValueError:
                break
            
            # Compute normal
            v1 = sample[1] - sample[0]
            v2 = sample[2] - sample[0]
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            
            if norm_n < 1e-6:
                continue
            
            A, B, C = n / norm_n
            D = -np.dot(n / norm_n, sample[0])
            
            # Count inliers
            dists = np.abs((points @ (n / norm_n)) + D)
            inliers = np.sum(dists < tolerance)
            
            iteration_history.append({
                'iteration': iteration,
                'inliers': inliers,
                'best_so_far': max(best_inliers, inliers)
            })
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_plane = np.array([A, B, C, D], dtype=float)
        
        execution_time = time.perf_counter() - start_time
        metadata = {
            'iterations': len(iteration_history),
            'inliers': best_inliers,
            'convergence_history': iteration_history
        }
        
        return best_plane, execution_time, metadata
    
    def vdisparity_algorithm(self, points, **kwargs):
        """V-Disparity implementation with performance tracking"""
        start_time = time.perf_counter()
        fx = kwargs.get('fx', 400)
        baseline = kwargs.get('baseline', 0.05)
        
        if len(points) == 0:
            return None, time.perf_counter() - start_time, {'iterations': 0, 'inliers': 0}
        
        Z = points[:, 2]
        Y = points[:, 1]
        
        # Filter valid depths
        valid_mask = (Z > 0.1) & (Z < 10)
        if not np.any(valid_mask):
            return None, time.perf_counter() - start_time, {'iterations': 0, 'inliers': 0}
        
        valid_pts = points[valid_mask]
        Z_valid = valid_pts[:, 2]
        Y_valid = valid_pts[:, 1]
        
        # Compute disparity
        with np.errstate(divide='ignore', invalid='ignore'):
            disparity = (fx * baseline) / Z_valid
            disparity = disparity[np.isfinite(disparity)]
        
        if len(disparity) == 0:
            return None, time.perf_counter() - start_time, {'iterations': 0, 'inliers': 0}
        
        try:
            valid_y = Y_valid[np.isfinite(disparity)]
            if len(valid_y) < 3:
                return None, time.perf_counter() - start_time, {'iterations': 0, 'inliers': 0}
            
            # Fit line through v-disparity space
            A = np.vstack([disparity, np.ones(len(disparity))]).T
            m, c = np.linalg.lstsq(A, valid_y, rcond=None)[0]
            
            # Convert to plane equation
            plane = np.array([0, 1, m, c], dtype=float)
            inliers = len(valid_y)
            
        except (np.linalg.LinAlgError, ValueError):
            return None, time.perf_counter() - start_time, {'iterations': 0, 'inliers': 0}
        
        execution_time = time.perf_counter() - start_time
        metadata = {
            'iterations': 1,
            'inliers': inliers,
            'disparity_points': len(disparity)
        }
        
        return plane, execution_time, metadata
    
    def generate_test_scenarios(self):
        """Generate comprehensive test scenarios"""
        scenarios = {}
        
        # 1. Ideal conditions
        scenarios['ideal'] = {
            'name': 'Ideal Conditions',
            'description': 'Clean ground plane with minimal noise',
            'data': self.generate_ground_plane(num_points=1000, noise_std=0.01),
            'expected_accuracy': 0.95
        }
        
        # 2. Noisy environment
        scenarios['noisy'] = {
            'name': 'Noisy Environment',
            'description': 'Ground plane with significant sensor noise',
            'data': self.generate_ground_plane(num_points=1000, noise_std=0.05),
            'expected_accuracy': 0.80
        }
        
        # 3. Obstacles present
        obstacles_data = self.add_obstacles(
            self.generate_ground_plane(num_points=800), 
            num_obstacles=8, 
            obstacle_height=0.8
        )
        scenarios['obstacles'] = {
            'name': 'Obstacles Present',
            'description': 'Ground plane with multiple obstacles',
            'data': obstacles_data,
            'expected_accuracy': 0.75
        }
        
        # 4. High outlier ratio
        outlier_data = self.add_outliers(
            self.generate_ground_plane(num_points=800), 
            outlier_ratio=0.3
        )
        scenarios['outliers'] = {
            'name': 'High Outlier Ratio',
            'description': '30% random outlier points',
            'data': outlier_data,
            'expected_accuracy': 0.70
        }
        
        # 5. Tilted ground plane
        scenarios['tilted'] = {
            'name': 'Tilted Ground',
            'description': 'Inclined ground surface',
            'data': self.generate_tilted_plane(tilt_x=0.15, tilt_z=0.08),
            'expected_accuracy': 0.85
        }
        
        # 6. Sparse data
        scenarios['sparse'] = {
            'name': 'Sparse Data',
            'description': 'Limited number of data points',
            'data': self.generate_ground_plane(num_points=200, noise_std=0.02),
            'expected_accuracy': 0.75
        }
        
        # 7. Dense data
        scenarios['dense'] = {
            'name': 'Dense Data',
            'description': 'High density point cloud',
            'data': self.generate_ground_plane(num_points=5000, noise_std=0.02),
            'expected_accuracy': 0.90
        }
        
        return scenarios
    
    def generate_ground_plane(self, width=10, depth=10, num_points=1000, noise_std=0.02):
        """Generate synthetic ground plane data"""
        x = np.random.uniform(-width/2, width/2, num_points)
        z = np.random.uniform(0, depth, num_points)
        y = np.random.normal(0, noise_std, num_points)
        return np.column_stack([x, y, z])
    
    def generate_tilted_plane(self, width=10, depth=10, num_points=1000, 
                            tilt_x=0.1, tilt_z=0.05, noise_std=0.02):
        """Generate tilted plane data"""
        x = np.random.uniform(-width/2, width/2, num_points)
        z = np.random.uniform(0, depth, num_points)
        y = tilt_x * x + tilt_z * z + np.random.normal(0, noise_std, num_points)
        return np.column_stack([x, y, z])
    
    def add_obstacles(self, points, num_obstacles=5, obstacle_height=0.5):
        """Add obstacle points"""
        obstacles = []
        for _ in range(num_obstacles):
            center_x = np.random.uniform(-4, 4)
            center_z = np.random.uniform(1, 8)
            size = np.random.uniform(0.3, 0.8)
            
            obs_points = np.random.randint(80, 150)
            x = np.random.normal(center_x, size, obs_points)
            z = np.random.normal(center_z, size, obs_points)
            y = np.random.uniform(0.1, obstacle_height, obs_points)
            
            obstacles.append(np.column_stack([x, y, z]))
        
        if obstacles:
            all_obstacles = np.vstack(obstacles)
            return np.vstack([points, all_obstacles])
        return points
    
    def add_outliers(self, points, outlier_ratio=0.1):
        """Add random outlier points"""
        num_outliers = int(len(points) * outlier_ratio)
        if num_outliers == 0:
            return points
        
        outliers = np.random.uniform(-8, 8, (num_outliers, 3))
        outliers[:, 1] = np.random.uniform(-1, 2, num_outliers)
        
        return np.vstack([points, outliers])
    
    def evaluate_accuracy(self, estimated_plane, true_plane):
        """Evaluate plane estimation accuracy"""
        if estimated_plane is None or true_plane is None:
            return 0.0
        
        # Normalize normals
        est_norm = estimated_plane[:3] / np.linalg.norm(estimated_plane[:3])
        true_norm = true_plane[:3] / np.linalg.norm(true_plane[:3])
        
        # Angular accuracy
        dot_product = np.clip(np.abs(np.dot(est_norm, true_norm)), 0, 1)
        angle_diff = np.arccos(dot_product)
        accuracy = 1.0 - (angle_diff / (np.pi / 2))
        
        return max(0, accuracy)
    
    def run_comprehensive_tests(self, num_trials=25):
        """Run comprehensive algorithm tests"""
        scenarios = self.generate_test_scenarios()
        results = {}
        
        print("Running comprehensive algorithm tests...")
        print(f"Testing {len(scenarios)} scenarios with {num_trials} trials each")
        
        for scenario_name, scenario in scenarios.items():
            print(f"\nTesting scenario: {scenario['name']}")
            results[scenario_name] = {}
            
            for alg_name, alg_func in self.algorithms.items():
                print(f"  {alg_name}... ", end='', flush=True)
                
                trial_results = []
                execution_times = []
                accuracies = []
                metadata_list = []
                
                for trial in range(num_trials):
                    # Add some variation to each trial
                    test_data = scenario['data'].copy()
                    if trial > 0:  # Add slight variation to subsequent trials
                        noise = np.random.normal(0, 0.005, test_data.shape)
                        test_data += noise
                    
                    # Run algorithm
                    plane, exec_time, metadata = alg_func(test_data)
                    
                    # Evaluate
                    true_plane = np.array([0, 1, 0, 0])  # Ground plane assumption
                    accuracy = self.evaluate_accuracy(plane, true_plane)
                    
                    trial_results.append({
                        'plane': plane,
                        'accuracy': accuracy,
                        'execution_time': exec_time,
                        'metadata': metadata
                    })
                    
                    execution_times.append(exec_time)
                    accuracies.append(accuracy)
                    metadata_list.append(metadata)
                
                # Aggregate results
                results[scenario_name][alg_name] = {
                    'trials': trial_results,
                    'avg_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'avg_execution_time': np.mean(execution_times),
                    'std_execution_time': np.std(execution_times),
                    'success_rate': np.mean([r['plane'] is not None for r in trial_results]),
                    'metadata': metadata_list
                }
                
                print(f"✓ ({np.mean(accuracies):.3f} accuracy)")
        
        return results
    
    def create_performance_matrix_visualization(self, results):
        """Create comprehensive performance matrix"""
        scenarios = list(results.keys())
        algorithms = list(self.algorithms.keys())
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Accuracy Heatmap
        ax1 = fig.add_subplot(gs[0, :2])
        accuracy_matrix = np.zeros((len(algorithms), len(scenarios)))
        
        for i, alg in enumerate(algorithms):
            for j, scenario in enumerate(scenarios):
                accuracy_matrix[i, j] = results[scenario][alg]['avg_accuracy']
        
        im1 = ax1.imshow(accuracy_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax1.set_xticks(range(len(scenarios)))
        ax1.set_xticklabels([results[s]['ideal']['trials'][0]['metadata'] if s in results and 'ideal' in results[s] else s.replace('_', ' ').title() for s in scenarios], rotation=45, ha='right')
        ax1.set_yticks(range(len(algorithms)))
        ax1.set_yticklabels(algorithms)
        ax1.set_title('Algorithm Accuracy by Scenario', fontweight='bold', fontsize=14)
        
        # Add text annotations
        for i in range(len(algorithms)):
            for j in range(len(scenarios)):
                text = ax1.text(j, i, f'{accuracy_matrix[i, j]:.3f}',
                              ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im1, ax=ax1, label='Accuracy Score')
        
        # 2. Execution Time Heatmap
        ax2 = fig.add_subplot(gs[0, 2:])
        time_matrix = np.zeros((len(algorithms), len(scenarios)))
        
        for i, alg in enumerate(algorithms):
            for j, scenario in enumerate(scenarios):
                time_matrix[i, j] = results[scenario][alg]['avg_execution_time'] * 1000  # ms
        
        im2 = ax2.imshow(time_matrix, cmap='RdYlBu_r', aspect='auto')
        ax2.set_xticks(range(len(scenarios)))
        ax2.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45, ha='right')
        ax2.set_yticks(range(len(algorithms)))
        ax2.set_yticklabels(algorithms)
        ax2.set_title('Execution Time by Scenario (ms)', fontweight='bold', fontsize=14)
        
        # Add text annotations
        for i in range(len(algorithms)):
            for j in range(len(scenarios)):
                text = ax2.text(j, i, f'{time_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im2, ax=ax2, label='Execution Time (ms)')
        
        # 3. Success Rate Bar Chart
        ax3 = fig.add_subplot(gs[1, :2])
        success_rates = {alg: [] for alg in algorithms}
        
        for scenario in scenarios:
            for alg in algorithms:
                success_rates[alg].append(results[scenario][alg]['success_rate'])
        
        x = np.arange(len(scenarios))
        width = 0.25
        
        for i, alg in enumerate(algorithms):
            offset = (i - 1) * width
            bars = ax3.bar(x + offset, success_rates[alg], width, 
                          label=alg, color=self.colors[alg], alpha=0.8)
        
        ax3.set_xlabel('Test Scenarios')
        ax3.set_ylabel('Success Rate')
        ax3.set_title('Algorithm Success Rate by Scenario', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels([s.replace('_', ' ').title() for s in scenarios], rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.1)
        
        # 4. Overall Performance Score
        ax4 = fig.add_subplot(gs[1, 2:])
        
        # Calculate composite scores
        composite_scores = {}
        for alg in algorithms:
            scores = []
            for scenario in scenarios:
                r = results[scenario][alg]
                # Weighted score: accuracy (50%), speed (25%), success rate (25%)
                score = (r['avg_accuracy'] * 0.5 + 
                        (1.0 / (r['avg_execution_time'] + 0.001)) * 0.1 + 
                        r['success_rate'] * 0.4)
                scores.append(score)
            composite_scores[alg] = np.mean(scores)
        
        alg_names = list(composite_scores.keys())
        scores = list(composite_scores.values())
        bars = ax4.bar(alg_names, scores, color=[self.colors[alg] for alg in alg_names])
        
        ax4.set_ylabel('Composite Performance Score')
        ax4.set_title('Overall Algorithm Performance', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Real-time Suitability Analysis
        ax5 = fig.add_subplot(gs[2, :2])
        
        # Calculate real-time metrics
        rt_metrics = {}
        for alg in algorithms:
            avg_times = [results[scenario][alg]['avg_execution_time'] for scenario in scenarios]
            avg_accuracies = [results[scenario][alg]['avg_accuracy'] for scenario in scenarios]
            
            avg_time = np.mean(avg_times) * 1000  # ms
            avg_accuracy = np.mean(avg_accuracies)
            theoretical_fps = 1000 / avg_time if avg_time > 0 else 0
            
            rt_metrics[alg] = {
                'fps': theoretical_fps,
                'accuracy': avg_accuracy,
                'latency': avg_time
            }
        
        for alg in algorithms:
            m = rt_metrics[alg]
            ax5.scatter(m['latency'], m['accuracy'], 
                       s=200, alpha=0.8, color=self.colors[alg], label=alg)
            ax5.annotate(f"{alg}\n{m['fps']:.1f} FPS", 
                        (m['latency'], m['accuracy']),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=10, ha='left')
        
        ax5.set_xlabel('Average Latency (ms)')
        ax5.set_ylabel('Average Accuracy')
        ax5.set_title('Real-Time Suitability Analysis', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # Add target zones
        ax5.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good Accuracy (>0.8)')
        ax5.axvline(x=33.33, color='red', linestyle='--', alpha=0.5, label='30 FPS Limit (33ms)')
        
        # 6. Detailed Performance Breakdown
        ax6 = fig.add_subplot(gs[2, 2:])
        
        # Create radar chart data
        metrics_names = ['Accuracy', 'Speed', 'Robustness', 'Consistency', 'Success Rate']
        
        for alg in algorithms:
            # Calculate normalized metrics
            accuracies = [results[scenario][alg]['avg_accuracy'] for scenario in scenarios]
            times = [results[scenario][alg]['avg_execution_time'] for scenario in scenarios]
            success_rates = [results[scenario][alg]['success_rate'] for scenario in scenarios]
            consistency = [1.0 - results[scenario][alg]['std_accuracy'] for scenario in scenarios]
            
            # Robustness = performance in challenging scenarios
            challenging_scenarios = ['outliers', 'obstacles', 'noisy']
            robustness_scores = []
            for scenario in challenging_scenarios:
                if scenario in results:
                    robustness_scores.append(results[scenario][alg]['avg_accuracy'])
            
            values = [
                np.mean(accuracies),                    # Accuracy
                1.0 / (np.mean(times) * 1000 + 1),    # Speed (normalized)
                np.mean(robustness_scores) if robustness_scores else 0,  # Robustness
                np.mean(consistency),                   # Consistency
                np.mean(success_rates)                  # Success Rate
            ]
            
            # Plot as bar chart instead of radar for clarity
            x_pos = np.arange(len(metrics_names))
            ax6.bar(x_pos + (list(algorithms).index(alg) - 1) * 0.25, values, 
                   width=0.25, label=alg, color=self.colors[alg], alpha=0.8)
        
        ax6.set_xlabel('Performance Metrics')
        ax6.set_ylabel('Normalized Score')
        ax6.set_title('Detailed Performance Breakdown', fontweight='bold')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(metrics_names, rotation=45, ha='right')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim