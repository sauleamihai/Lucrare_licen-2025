# RealSense Advanced Smoothing Analysis Report

Generated on: 2025-06-19 14:31:31

## Executive Summary

This comprehensive analysis evaluated **99** different smoothing methods on real-time RealSense depth sensor data across **5** polar grid bins.

### Data Collection Summary

- **Total Frames Collected**: 9000
- **Collection Duration**: 300.9 seconds
- **Average Frame Rate**: 29.9 FPS
- **Average Processing Latency**: 33.4 ms
- **Tracked Bins**: (0, 4), (1, 4), (1, 8), (2, 0), (2, 8)

### Key Findings

- **Best Overall Method**: Gaussian_s0.5
- **Total Methods Tested**: 99
- **Real-time Performance**: ✗ Below target (Target: <33.33ms for 30 FPS)

## RealSense Configuration

### Hardware Setup
- **Depth Stream**: 640x480 @ 30 FPS
- **Depth Scale**: 0.0010000000474974513
- **Field of View**: 80.6 degrees
- **Polar Grid**: 3 radial × 16 angular bins

### Filter Pipeline
1. Decimation Filter (factor: 2)
2. Threshold Filter (0.15m - 4.5m)
3. Disparity Transform
4. Spatial Filter (magnitude: 5, alpha: 0.5, delta: 20)
5. Temporal Filter (alpha: 0.4, delta: 20)
6. Hole Filling (mode: 2)

### Ground Plane Detection
- **RANSAC Tolerance**: 0.1m
- **RANSAC Iterations**: 60
- **Plane Smoothing**: α = 0.8
- **Object Height Range**: 0.02m - 1.9m

## Method Analysis Results

### Top 10 Methods by Overall Performance

| Rank | Method | Mean RMSE | Std RMSE | Bins Tested |
|------|--------|-----------|----------|-------------|
| 1 | Gaussian_s0.5 | 0.0000 | 0.0000 | 5 |
| 2 | Kalman_q0.5_r0.1 | 133.0778 | 171.3073 | 5 |
| 3 | Kalman_q1.0_r0.1 | 169.7593 | 219.9210 | 5 |
| 4 | Kalman_q1.0_r0.5 | 176.8451 | 225.0373 | 5 |
| 5 | SavGol_w5_o4 | 258.3256 | 335.1021 | 5 |
| 6 | Kalman_q0.1_r0.1 | 283.6280 | 361.0629 | 5 |
| 7 | Kalman_q0.5_r0.5 | 283.6518 | 361.0806 | 5 |
| 8 | Kalman_q1.0_r1.0 | 283.7070 | 361.0918 | 5 |
| 9 | SavGol_w7_o4 | 404.4019 | 531.1671 | 5 |
| 10 | Kalman_q0.05_r0.1 | 404.8148 | 514.8814 | 5 |


## Bin-Specific Analysis

### Bin (0, 4)
- **Data Points**: 9000
- **Best Method**: Gaussian_s0.5 (RMSE: 0.0000)
- **Data Range**: 0.0 - 8880.0
- **Data Variance**: 587284.261

### Bin (1, 4)
- **Data Points**: 9000
- **Best Method**: Gaussian_s0.5 (RMSE: 0.0000)
- **Data Range**: 0.0 - 8723.0
- **Data Variance**: 1789371.992

### Bin (1, 8)
- **Data Points**: 9000
- **Best Method**: EMA_a0.01 (RMSE: 0.0000)
- **Data Range**: 0.0 - 0.0
- **Data Variance**: 0.000

### Bin (2, 0)
- **Data Points**: 9000
- **Best Method**: Gaussian_s0.5 (RMSE: 0.0000)
- **Data Range**: 0.0 - 12485.0
- **Data Variance**: 15122341.954

### Bin (2, 8)
- **Data Points**: 9000
- **Best Method**: EMA_a0.01 (RMSE: 0.0000)
- **Data Range**: 0.0 - 0.0
- **Data Variance**: 0.000

