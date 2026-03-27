#!/bin/bash
export GEL_AZ0=10
export IBM_BETA=3.2
export GEL_BILAYER_X=2    # LX=4, bilayer at x=1,2 passive / x=3,4 active
export RAND_SEED=42
cd /mnt/sdb2/wangyunjie/gLSM_fluid_3D
# 先跑2000步快速测试（改runstep不方便，用timeout代替）
timeout 60 ./gLSM_cilia_bilayer 2>&1 | head -15
