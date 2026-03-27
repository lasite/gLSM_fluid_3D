#!/bin/bash
# Build and run the peristaltic pumping simulation
# Usage: bash run_peristaltic.sh
# Env overrides: RUNSTEP, GEL_SIZE_X, WAVE_DELAY, GPU_ARCH

set -e
cd /mnt/sdb2/wangyunjie/gLSM_fluid_3D

# ── GPU architecture (override as needed) ─────────────────────────────────────
ARCH=${GPU_ARCH:-sm_86}

echo "=== Compiling gLSM_peristaltic (arch=${ARCH}) ==="
nvcc -O3 -std=c++17 -arch=${ARCH} \
     sim_peristaltic.cpp gel.cu fluid.cu coupling.cu \
     gel_kernels.cu fluid_kernels.cu coupling_kernels.cu \
     -o gLSM_peristaltic

echo "=== Binary built: gLSM_peristaltic ==="

# ── Create dedicated output directory ─────────────────────────────────────────
mkdir -p data_peristaltic
pushd data_peristaltic

echo "=== Starting simulation (RUNSTEP=${RUNSTEP:-150000}) ==="
nohup env RUNSTEP=${RUNSTEP:-150000} \
          GEL_SIZE_X=${GEL_SIZE_X:-200} \
          WAVE_DELAY=${WAVE_DELAY:-500} \
          ../gLSM_peristaltic \
     > ../run_peristaltic.log 2>&1 &

echo "Started gLSM_peristaltic PID=$!  log → run_peristaltic.log"
popd
