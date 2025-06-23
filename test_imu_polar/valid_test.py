#!/usr/bin/env python3
"""
Setup script for IMU Performance Test
Run this first to ensure everything is properly configured
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'numpy',
        'pyrealsense2', 
        'opencv-python',
        'scipy',
        'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"{package}")
        except ImportError:
            missing_packages.append(package)
            print(f"{package} - MISSING")
    
    if missing_packages:
        print(f"\nInstalling missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    return len(missing_packages) == 0

def check_file_structure():
    """Check if required files exist"""
    required_files = {
        'imu_plane_detector.py': 'Your IMU-enhanced detector code',
        'imu_performance_test.py': 'The performance test suite'
    }
    
    missing_files = []
    
    for filename, description in required_files.items():
        if os.path.exists(filename):
            print(f"{filename}")
        else:
            missing_files.append((filename, description))
            print(f"{filename} - {description}")
    
    if missing_files:
        print("\nMissing files:")
        for filename, description in missing_files:
            print(f"   - {filename}: {description}")
        print("\nPlease ensure all required files are in the same directory.")
        return False
    
    return True

def check_realsense_connection():
    """Check if RealSense camera is connected"""
    try:
        import pyrealsense2 as rs
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print(" No RealSense devices found")
            print("   Make sure your camera is connected and drivers are installed")
            return False
        
        for device in devices:
            print(f"Found RealSense device: {device.get_info(rs.camera_info.name)}")
            
            # Check for IMU capability
            sensors = device.query_sensors()
            has_imu = any(sensor.get_info(rs.camera_info.name) == 'Motion Module' for sensor in sensors)
            
            if has_imu:
                print("IMU supported")
            else:
                print("No IMU detected (depth-only mode available)")
        
        return True
        
    except Exception as e:
        print(f"RealSense check failed: {e}")
        return False

def test_import_structure():
    """Test if imports work correctly"""
    print("\nTesting import structure...")
    
    try:
        # Test basic imports
        import numpy as np
        import pyrealsense2 as rs
        print("Basic imports working")
        
        # Test IMU detector import
        from imu_plane_detector import FastHybridPlaneDetector, TemporalObstacleFilter, IMUCalibration
        print("IMU detector imports working")
        
        # Test instantiation
        imu_cal = IMUCalibration(enable_imu=False)
        detector = FastHybridPlaneDetector(imu_cal)
        filter_obj = TemporalObstacleFilter()
        print("Object instantiation working")
        
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("   Check that 'imu_plane_detector.py' contains the required classes")
        return False
    except Exception as e:
        print(f"Instantiation error: {e}")
        return False

def create_test_config():
    """Create a test configuration file"""
    config = {
        "test_scenarios": ["static", "tilt_slow"],  # Start with basic scenarios
        "frames_per_test": 100,  # Shorter for initial testing
        "quick_mode": True
    }
    
    try:
        import json
        with open('test_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        print("Created test_config.json for quick testing")
        return True
    except Exception as e:
        print(f"Failed to create config: {e}")
        return False

def main():
    """Main setup function"""
    print("Setting up IMU Performance Test Environment")
    print("=" * 50)
    
    print("\nChecking dependencies...")
    deps_ok = check_dependencies()
    
    print("\nChecking file structure...")
    files_ok = check_file_structure()
    
    print("\nChecking RealSense connection...")
    camera_ok = check_realsense_connection()
    
    if files_ok:
        print("\nTesting imports...")
        imports_ok = test_import_structure()
    else:
        imports_ok = False
    
    print("\nCreating test configuration...")
    config_ok = create_test_config()
    
    print("\n" + "=" * 50)
    print("SETUP SUMMARY")
    print("=" * 50)
    
    status_items = [
        ("Dependencies", deps_ok),
        ("File structure", files_ok), 
        ("RealSense camera", camera_ok),
        ("Import structure", imports_ok),
        ("Test configuration", config_ok)
    ]
    
    all_ok = True
    for item, status in status_items:
        icon = "done" if status else "error"
        print(f"{icon} {item}")
        if not status:
            all_ok = False
    
    if all_ok:
        print("\nSetup completed successfully!")
        print("\n You can now run the performance test:")
        print("   python imu_performance_test.py --quick")
        print("   python imu_performance_test.py --frames 200")
    else:
        print("\n  Setup incomplete. Please fix the issues above before running tests.")
        
        print("\n Common solutions:")
        if not files_ok:
            print("   - Make sure your IMU detector code is saved as 'imu_plane_detector.py'")
        if not camera_ok:
            print("   - Connect your RealSense camera and install Intel RealSense SDK")
        if not imports_ok:
            print("   - Check that all required classes are properly defined in your detector file")

if __name__ == "__main__":
    main()
