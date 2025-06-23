#!/usr/bin/env python3

import subprocess
import time
import json
from flask import Flask, request, jsonify, Response
from typing import Dict, List, Tuple, Optional

app = Flask(__name__)

# ────────── Configuration ──────────

IMAGE_NAME = "my_saved_images:with_fast_hybrid"
CONTAINER_NAME = "polar_remote"

# Allowed scripts with their configurable versions
ALLOWED_SCRIPTS = {
    "FAST_HYBRID": "/hybrid_tcp.py",
    "POLAR_RANSAC": "/polar_with_tcp.py", 
    "V_DISPARITY": "/v_disparity_tcp.py",
    "IRLS": "/irls_with_tcp.py"
}

# Default script
DEFAULT_SCRIPT = "FAST_HYBRID"

# Base Docker run command
DOCKER_RUN_BASE = [
    "docker", "run",
    "--rm",
    "--name", CONTAINER_NAME,
    "--privileged",
    "--network", "host",
    "--device", "/dev/cpu_dma_latency",
    "--cap-add", "SYS_PTRACE",
    IMAGE_NAME
]

# ────────── Predefined Profiles ──────────

# CORRECTED PROFILES - Replace in your polar_control.py
# Based on actual parameter names from your scripts

PROFILES = {
    # ═══ FAST HYBRID PROFILES (hybrid_tcp.py) ═══
    "fast_hybrid_indoor": {
        "script": "FAST_HYBRID",
        "description": "Fast Hybrid optimized for indoor environments",
        "params": {
            "depth-min": 0.1,
            "depth-max": 3.0,
            "radial-edges": "0.0,0.3,0.8,2.0",
            "sample-size": 2000,
            "prosac-iterations": 20,
            "ground-eps": 0.01,
            "max-height": 1.8,
            "ema-alpha": 0.08,
            "temporal-decay": 0.9,
            "interval": 1.0  # FIXED: was "send-interval"
        }
    },
    
    "fast_hybrid_outdoor": {
        "script": "FAST_HYBRID", 
        "description": "Fast Hybrid optimized for outdoor environments",
        "params": {
            "depth-min": 0.5,
            "depth-max": 8.0,
            "radial-edges": "0.0,1.0,3.0,8.0",
            "sample-size": 2500,
            "prosac-iterations": 25,
            "ground-eps": 0.03,
            "max-height": 3.0,
            "ema-alpha": 0.05,
            "temporal-decay": 0.85,
            "interval": 1.0  # FIXED: was "send-interval"
        }
    },
    
    "fast_hybrid_speed": {
        "script": "FAST_HYBRID",
        "description": "Fast Hybrid optimized for maximum speed",
        "params": {
            "sample-size": 800,
            "prosac-iterations": 8,
            "method": "refinement",
            "no-temporal": True,
            "interval": 0.5  # FIXED: was "send-interval"
        }
    },
    
    "fast_hybrid_accuracy": {
        "script": "FAST_HYBRID",
        "description": "Fast Hybrid optimized for maximum accuracy",
        "params": {
            "sample-size": 3000,
            "prosac-iterations": 30,
            "hough-resolution": 0.02,
            "stability-threshold": 0.10,
            "ema-alpha": 0.03,
            "temporal-decay": 0.95,
            "interval": 2.0  # Added interval
        }
    },

    # ═══ POLAR RANSAC PROFILES (polar_with_tcp.py) ═══
    "ransac_indoor": {
        "script": "POLAR_RANSAC",
        "description": "RANSAC optimized for indoor environments", 
        "params": {
            "depth-min": 0.15,
            "depth-max": 4.0,
            "radial-edges": "0.0,0.5,1.0,3.0",
            "ransac-iterations": 80,
            "ransac-tolerance": 0.08,
            "plane-alpha": 0.85,
            "ema-alpha": 0.1,
            "min-ground-points": 60,
            "send-interval": 1.0  # Correct parameter name for this script
        }
    },
    
    "ransac_outdoor": {
        "script": "POLAR_RANSAC",
        "description": "RANSAC optimized for outdoor environments",
        "params": {
            "depth-min": 0.3,
            "depth-max": 6.0,
            "radial-edges": "0.0,1.0,2.5,6.0",
            "ransac-iterations": 100,
            "ransac-tolerance": 0.12,
            "plane-alpha": 0.9,
            "ground-eps": 0.04,
            "max-height": 2.5,
            "ema-alpha": 0.06,
            "send-interval": 1.0
        }
    },
    
    "ransac_robust": {
        "script": "POLAR_RANSAC",
        "description": "RANSAC with maximum robustness",
        "params": {
            "ransac-iterations": 120,
            "ransac-tolerance": 0.05,
            "plane-alpha": 0.95,
            "min-ground-points": 100,
            "ema-alpha": 0.04,
            "send-interval": 1.5
        }
    },

    # ═══ V-DISPARITY PROFILES (v_disparity_tcp.py) ═══
    "vdisp_indoor": {
        "script": "V_DISPARITY",
        "description": "V-Disparity optimized for indoor environments",
        "params": {
            "depth-min": 0.15,
            "depth-max": 4.0,
            "hough-threshold": 15,
            "canny-low": 20,
            "canny-high": 60,
            "d-max": 256,
            "baseline": 0.05,
            "ema-alpha": 0.6,
            "send-interval": 1.0
        }
    },
    
    "vdisp_outdoor": {
        "script": "V_DISPARITY", 
        "description": "V-Disparity optimized for outdoor environments",
        "params": {
            "depth-min": 0.3,
            "depth-max": 8.0,
            "hough-threshold": 25,
            "canny-low": 15,
            "canny-high": 50,
            "hough-min-theta": -20,
            "hough-max-theta": 20,
            "d-max": 512,
            "radial-edges": "0.0,1.5,4.0,8.0",
            "send-interval": 1.0
        }
    },
    
    "vdisp_sensitive": {
        "script": "V_DISPARITY",
        "description": "V-Disparity with high sensitivity for weak features",
        "params": {
            "hough-threshold": 8,
            "canny-low": 10,
            "canny-high": 40,
            "gaussian-kernel": 7,
            "hough-min-theta": -25,
            "hough-max-theta": 25,
            "send-interval": 1.0
        }
    },

    # ═══ IRLS PROFILES (irls_with_tcp.py) ═══
    "irls_indoor": {
        "script": "IRLS",
        "description": "IRLS optimized for indoor environments",
        "params": {
            "depth-min": 0.15,
            "depth-max": 4.0,
            "irls-iterations": 6,
            "huber-delta": 0.04,
            "estimator": "huber",
            "ema-alpha": 0.4,
            "min-ground-points": 5,
            "send-interval": 1.0
        }
    },
    
    "irls_outdoor": {
        "script": "IRLS",
        "description": "IRLS optimized for outdoor environments", 
        "params": {
            "depth-min": 0.3,
            "depth-max": 8.0,
            "irls-iterations": 8,
            "huber-delta": 0.06,
            "estimator": "tukey",
            "ground-eps": 0.04,
            "max-height": 2.5,
            "ema-alpha": 0.3,
            "send-interval": 1.0
        }
    },
    
    "irls_robust": {
        "script": "IRLS",
        "description": "IRLS with maximum robustness using Tukey estimator",
        "params": {
            "irls-iterations": 10,
            "huber-delta": 0.03,
            "estimator": "tukey",
            "ema-alpha": 0.2,
            "use-grid-intensity": True,
            "send-interval": 1.5
        }
    },

    # ═══ SPECIAL PROFILES ═══
    "debug_verbose": {
        "script": "FAST_HYBRID",
        "description": "Debug profile with verbose output",
        "params": {
            "verbose": True,
            "sample-size": 1500,
            "interval": 2.0  # FIXED: was "send-interval"
        }
    },
    
    "speed_test": {
        "script": "FAST_HYBRID",
        "description": "Fastest possible configuration for speed testing",
        "params": {
            "sample-size": 500,
            "prosac-iterations": 5,
            "method": "refinement",
            "no-temporal": True,
            "interval": 0.2,  # FIXED: was "send-interval"
            "depth-max": 3.0
        }
    },
    
    "warehouse": {
        "script": "FAST_HYBRID",
        "description": "Optimized for warehouse/industrial environments",
        "params": {
            "depth-min": 0.2,
            "depth-max": 15.0,
            "radial-edges": "0.0,2.0,5.0,15.0",
            "max-height": 4.0,
            "sample-size": 2500,
            "ground-eps": 0.02,
            "ema-alpha": 0.06,
            "interval": 1.0  # FIXED: was "send-interval"
        }
    },
    
    "corridor": {
        "script": "V_DISPARITY",
        "description": "Optimized for narrow corridors and hallways",
        "params": {
            "depth-min": 0.1,
            "depth-max": 10.0,
            "radial-edges": "0.0,1.0,3.0,10.0",
            "hough-min-theta": -10,
            "hough-max-theta": 10,
            "hough-threshold": 12,
            "send-interval": 1.0
        }
    }
} 

# ───────── Helpers ──────────

def is_container_running() -> bool:
    """Check if the container is currently running"""
    try:
        out = subprocess.check_output(
            ["docker", "ps", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Names}}"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return (CONTAINER_NAME in out.splitlines())
    except Exception:
        return False

def build_command_with_params(script_key: str, params: Dict) -> List[str]:
    """Build Docker command with script and parameters"""
    if script_key not in ALLOWED_SCRIPTS:
        raise ValueError(f"Invalid script: {script_key}")
    
    script_path = ALLOWED_SCRIPTS[script_key]
    cmd = DOCKER_RUN_BASE + ["python3", script_path]
    
    # Add parameters
    for key, value in params.items():
        if value is True:  # Boolean flags
            cmd.append(f"--{key}")
        elif value is not False and value is not None:  # Skip False and None
            cmd.extend([f"--{key}", str(value)])
    
    return cmd

def start_container(script_key: str, params: Optional[Dict] = None) -> Tuple[bool, str]:
    """Start container with specified script and parameters"""
    if is_container_running():
        return False, "Container already running."

    try:
        if params is None:
            params = {}
            
        docker_cmd = build_command_with_params(script_key, params)
        
        subprocess.Popen(
            docker_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(1.5)  # Give more time for startup
        
        if is_container_running():
            return True, f"Container started with {script_key}"
        else:
            return False, "Failed to start container; it exited prematurely."
            
    except Exception as e:
        return False, f"Error launching container: {e}"

def stop_container() -> Tuple[bool, str]:
    """Stop the running container"""
    if not is_container_running():
        return False, "Container not running."

    try:
        subprocess.check_call(
            ["docker", "stop", CONTAINER_NAME],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True, "Container stopped."
    except Exception as e:
        return False, f"Error stopping container: {e}"

def fetch_container_logs(lines: int = 1000) -> str:
    """Fetch container logs"""
    try:
        out = subprocess.check_output(
            ["docker", "logs", "--tail", str(lines), CONTAINER_NAME],
            stderr=subprocess.STDOUT
        ).decode(errors="ignore")
        return out
    except subprocess.CalledProcessError as cpe:
        msg = cpe.output.decode(errors="ignore")
        return f"(No logs available—container may not exist or has exited & been removed.)\n{msg}"
    except Exception as e:
        return f"(Error fetching logs: {e})"

def validate_params(script_key: str, params: Dict) -> Tuple[bool, str]:
    """Basic parameter validation"""
    # Common validations
    if "depth-min" in params and "depth-max" in params:
        if float(params["depth-min"]) >= float(params["depth-max"]):
            return False, "depth-min must be less than depth-max"
    
    if "ema-alpha" in params:
        alpha = float(params["ema-alpha"])
        if not (0.0 <= alpha <= 1.0):
            return False, "ema-alpha must be between 0.0 and 1.0"
    
    # Script-specific validations
    if script_key == "POLAR_RANSAC":
        if "ransac-iterations" in params and int(params["ransac-iterations"]) <= 0:
            return False, "ransac-iterations must be positive"
        if "ransac-tolerance" in params and float(params["ransac-tolerance"]) <= 0:
            return False, "ransac-tolerance must be positive"
    
    elif script_key == "IRLS":
        if "irls-iterations" in params and int(params["irls-iterations"]) <= 0:
            return False, "irls-iterations must be positive"
        if "huber-delta" in params and float(params["huber-delta"]) <= 0:
            return False, "huber-delta must be positive"
    
    elif script_key == "V_DISPARITY":
        if "d-max" in params and int(params["d-max"]) <= 0:
            return False, "d-max must be positive"
        if "hough-threshold" in params and int(params["hough-threshold"]) <= 0:
            return False, "hough-threshold must be positive"
    
    return True, "Parameters valid"

# ────────── Flask Routes ──────────

@app.route("/", methods=["GET"])
def index():
    """API documentation"""
    return jsonify({
        "message": "Enhanced Polar Algorithm Remote Control API",
        "endpoints": {
            "GET /container": "Check container status",
            "POST /container": "Start/stop container with script and parameters",
            "GET /container/logs": "Get container logs",
            "GET /scripts": "List available scripts",
            "GET /profiles": "List available profiles", 
            "GET /profiles/<name>": "Get specific profile details",
            "POST /container/profile": "Start container with predefined profile"
        },
        "available_scripts": list(ALLOWED_SCRIPTS.keys()),
        "available_profiles": list(PROFILES.keys())
    })

@app.route("/container", methods=["GET"])
def container_status():
    """Get container status"""
    status = is_container_running()
    return jsonify({"running": status})

@app.route("/container", methods=["POST"])
def container_control():
    """Start/stop container with custom parameters"""
    data = request.get_json(force=True)
    if not data or "action" not in data:
        return jsonify({"success": False, "message": "Missing 'action' field."}), 400

    action = data["action"].lower()

    if action == "start":
        script_key = data.get("script", DEFAULT_SCRIPT)
        params = data.get("params", {})

        if script_key not in ALLOWED_SCRIPTS:
            return jsonify({
                "success": False,
                "message": f"Invalid script '{script_key}'. Allowed: {list(ALLOWED_SCRIPTS.keys())}"
            }), 400

        # Validate parameters
        valid, msg = validate_params(script_key, params)
        if not valid:
            return jsonify({"success": False, "message": f"Parameter validation failed: {msg}"}), 400

        success, msg = start_container(script_key, params)
        code = 200 if success else 400
        
        response = {"success": success, "message": msg}
        if success:
            response["script"] = script_key
            response["params"] = params
            
        return jsonify(response), code

    elif action == "stop":
        success, msg = stop_container()
        code = 200 if success else 400
        return jsonify({"success": success, "message": msg}), code

    else:
        return jsonify({
            "success": False,
            "message": f"Unknown action '{action}'. Use 'start' or 'stop'."
        }), 400

@app.route("/container/profile", methods=["POST"])
def container_profile():
    """Start container with predefined profile"""
    data = request.get_json(force=True)
    if not data or "profile" not in data:
        return jsonify({"success": False, "message": "Missing 'profile' field."}), 400

    profile_name = data["profile"]
    if profile_name not in PROFILES:
        return jsonify({
            "success": False,
            "message": f"Invalid profile '{profile_name}'. Available: {list(PROFILES.keys())}"
        }), 400

    profile = PROFILES[profile_name]
    script_key = profile["script"]
    params = profile["params"].copy()
    
    # Allow parameter overrides
    if "param_overrides" in data:
        params.update(data["param_overrides"])

    # Validate parameters
    valid, msg = validate_params(script_key, params)
    if not valid:
        return jsonify({"success": False, "message": f"Parameter validation failed: {msg}"}), 400

    success, msg = start_container(script_key, params)
    code = 200 if success else 400
    
    response = {"success": success, "message": msg}
    if success:
        response["profile"] = profile_name
        response["description"] = profile["description"]
        response["script"] = script_key
        response["params"] = params
        
    return jsonify(response), code

@app.route("/container/logs", methods=["GET"])
def container_logs():
    """Get container logs"""
    lines = request.args.get("lines", 1000, type=int)
    logs = fetch_container_logs(lines)
    return Response(logs, content_type="text/plain")

@app.route("/scripts", methods=["GET"])
def list_scripts():
    """List available scripts with descriptions"""
    script_info = {}
    for key, path in ALLOWED_SCRIPTS.items():
        if "configurable" in path or "esp32" in path:
            script_info[key] = {
                "path": path,
                "configurable": True,
                "description": f"Configurable version of {key.lower().replace('_', ' ')}"
            }
        else:
            script_info[key] = {
                "path": path,
                "configurable": False,
                "description": f"Legacy version of {key.lower().replace('_', ' ')}"
            }
    
    return jsonify({
        "available_scripts": script_info,
        "default_script": DEFAULT_SCRIPT
    })

@app.route("/profiles", methods=["GET"])
def list_profiles():
    """List available profiles with descriptions"""
    profile_list = {}
    for name, profile in PROFILES.items():
        profile_list[name] = {
            "script": profile["script"],
            "description": profile["description"],
            "param_count": len(profile["params"])
        }
    
    return jsonify({"available_profiles": profile_list})

@app.route("/profiles/<profile_name>", methods=["GET"])
def get_profile(profile_name):
    """Get detailed profile information"""
    if profile_name not in PROFILES:
        return jsonify({
            "success": False,
            "message": f"Profile '{profile_name}' not found"
        }), 404
    
    profile = PROFILES[profile_name]
    return jsonify({
        "name": profile_name,
        "script": profile["script"],
        "description": profile["description"],
        "params": profile["params"]
    })

@app.route("/container/command", methods=["POST"])
def get_docker_command():
    """Get the Docker command that would be executed (for debugging)"""
    data = request.get_json(force=True)
    
    if "profile" in data:
        profile_name = data["profile"]
        if profile_name not in PROFILES:
            return jsonify({"success": False, "message": f"Invalid profile '{profile_name}'"}), 400
        
        profile = PROFILES[profile_name]
        script_key = profile["script"]
        params = profile["params"].copy()
        
        if "param_overrides" in data:
            params.update(data["param_overrides"])
    else:
        script_key = data.get("script", DEFAULT_SCRIPT)
        params = data.get("params", {})
        
        if script_key not in ALLOWED_SCRIPTS:
            return jsonify({"success": False, "message": f"Invalid script '{script_key}'"}), 400
    
    try:
        cmd = build_command_with_params(script_key, params)
        return jsonify({
            "success": True,
            "command": cmd,
            "script": script_key,
            "params": params
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 400

# ────────── Entry Point ──────────

if __name__ == "__main__":
    print("Enhanced Polar Algorithm Remote Control API")
    print(f"Available Scripts: {len(ALLOWED_SCRIPTS)}")
    print(f"Available Profiles: {len(PROFILES)}")
    print("Starting Flask server on 0.0.0.0:5000")
    
    app.run(host="0.0.0.0", port=5000, debug=False)
