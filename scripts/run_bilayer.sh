#!/bin/bash
# Bilayer BZ gel cilia array
# Passive half: xi=1..2 (x < gel_center), Active half: xi=3..4 (x > gel_center)
# Breaks symmetry → lateral bending → dipole flow → hydrodynamic coupling
export GEL_AZ0=10
export IBM_BETA=3.2
export GEL_BILAYER_X=2    # LX=4, split at midpoint
export RAND_SEED=42
cd /mnt/sdb2/wangyunjie/gLSM_fluid_3D
nohup ./gLSM_cilia_bilayer > run_bilayer.log 2>&1 &
echo "Started gLSM_cilia_bilayer PID=$!"
