#!/usr/bin/env bash
# ============================
# benchmark_polar.sh
# ============================
# This script benchmarks a single "v-Disparity" polar-grid approach (polar_vd_tuned.py).
#
# Metrics captured:
#   1. Per-frame latency & peak memory usage (GNU time) 
#   2. Total energy (perf stat with RAPL)
#   3. (Optional) Container-level CPU/MEM via `docker stats`
#
# Output files:
#   - time_log.txt
#   - perf_log.txt
#   - stdout_log.txt
#   - docker_stats_log.txt (optional; see NOTES at bottom)
#
# Usage: ./benchmark_polar.sh
# ============================

set -euo pipefail

# ────────────── Configurable Params ──────────────
PY_SCRIPT="polar_irls_test.py"   # The Python script under test
NUM_FRAMES=500                  # Number of frames the script should process before exiting
OUTPUT_DIR="/usr/src/app/logs"
TIME_LOG="${OUTPUT_DIR}/time_log.txt"
PERF_LOG="${OUTPUT_DIR}/perf_log.txt"
STDOUT_LOG="${OUTPUT_DIR}/stdout_log.txt"
DOCKER_STATS_LOG="${OUTPUT_DIR}/docker_stats_log.txt"

# Ensure output directory exists
mkdir -p "${OUTPUT_DIR}"

# ────────────── Helper Functions ──────────────

function warm_up() {
    echo "[INFO] Warming up: single run of ${PY_SCRIPT} (no logging)" | tee -a "${STDOUT_LOG}"
    python3 "${PY_SCRIPT}" --frames 50 >/dev/null 2>&1
    echo "[INFO] Warm-up complete." | tee -a "${STDOUT_LOG}"
    # Sleep to let CPU freq stabilize (if ondemand gov)
    sleep 2
}

function run_time_memory() {
    echo "[INFO] Running with /usr/bin/time to measure time + peak memory" | tee -a "${STDOUT_LOG}"
    # -f format: Elapsed (wall) time, user time, sys time, max RSS (kB), avg CPU%
    /usr/bin/time -v python3 "${PY_SCRIPT}" --frames "${NUM_FRAMES}" \
        > "${STDOUT_LOG}" 2> "${TIME_LOG}"
    echo "[INFO] Time+Memory logging complete. See ${TIME_LOG}" | tee -a "${STDOUT_LOG}"
}

function run_energy() {
    echo "[INFO] Running with perf stat to measure CPU package energy (RAPL)" | tee -a "${STDOUT_LOG}"
    # perf stat -e power/energy-pkg measures energy (in nanojoules) consumed by CPU package
    # -o <file>: write results to file; --quiet suppress text summary
    perf stat -e power/energy-pkg -o "${PERF_LOG}" -- python3 "${PY_SCRIPT}" --frames "${NUM_FRAMES}" \
        >> "${STDOUT_LOG}" 2>&1
    echo "[INFO] Energy logging complete. See ${PERF_LOG}" | tee -a "${STDOUT_LOG}"
}

# ────────────── Main Benchmark Workflow ──────────────

echo "=============================================="
echo "  Polar-Grid Benchmark: v-Disparity Approach"
echo "        $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Num frames to process: ${NUM_FRAMES}"
echo "=============================================="
echo "" | tee -a "${STDOUT_LOG}"

# 1) Warm-up run
warm_up

# 2) Measure Time + Memory
run_time_memory

# 3) Measure Energy
run_energy

# 4) (Optional) Docker stats
#    Note: Inside container, `docker stats` cannot see itself. To capture container-level stats,
#    run the following command from *host* while this script is running:
#      docker stats --no-stream <CONTAINER_ID> > "${DOCKER_STATS_LOG}"
#    You can also wrap this in a host-side script. See NOTES below.
echo "" | tee -a "${STDOUT_LOG}"
echo "[INFO] ** If you want container-level CPU/MEM stats, run: **" | tee -a "${STDOUT_LOG}"
echo "  docker stats --no-stream \$(hostname) > ${DOCKER_STATS_LOG}" | tee -a "${STDOUT_LOG}"
echo "[INFO] Benchmark completed." | tee -a "${STDOUT_LOG}"
