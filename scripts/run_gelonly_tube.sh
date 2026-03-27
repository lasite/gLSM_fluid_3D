#!/bin/bash
# Build and run the Gel-Only Tube simulation
# Usage: bash run_gelonly_tube.sh
# Env overrides: RUNSTEP, WAVE_DELAY, WALL_THICKNESS, TUBE_CIRCULAR (0/1), GPU_ARCH

set -e
cd /mnt/sdb2/wangyunjie/gLSM_fluid_3D

ARCH=${GPU_ARCH:-sm_70}

echo "=== Compiling gLSM_gelonly_tube (arch=${ARCH}) ==="
nvcc -O3 -std=c++17 -arch=${ARCH} \
     sim_gelonly_tube.cpp gel.cu gel_kernels.cu \
     -o gLSM_gelonly_tube

echo "=== Binary built: gLSM_gelonly_tube ==="

mkdir -p data_gelonly_tube
pushd data_gelonly_tube

# default to circular tube (1) and wall_thickness 4
export WALL_THICKNESS=${WALL_THICKNESS:-4}
export TUBE_CIRCULAR=${TUBE_CIRCULAR:-1}

echo "=== Starting simulation (RUNSTEP=${RUNSTEP:-20000}, MODE=${TUBE_CIRCULAR}, WALL=${WALL_THICKNESS}) ==="
nohup ../gLSM_gelonly_tube > ../run_gelonly_tube.log 2>&1 &

echo "Started gLSM_gelonly_tube PID=$!  log → run_gelonly_tube.log"
popd
