#!/bin/bash
# 4x4 Cilia array experiment — GEL_AZ0=10 (matches antiphase run params)
# BZ parameters: AZ0=10, default beta/etc
# 16 gels, random initial phases (seed=42), runstep=200000

export GEL_AZ0=10
export RAND_SEED=42

cd /mnt/sdb2/wangyunjie/gLSM_fluid_3D
nohup ./gLSM_cilia > run_cilia.log 2>&1 &
echo "Started gLSM_cilia PID=$!"
