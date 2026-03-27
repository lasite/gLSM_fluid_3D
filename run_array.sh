#!/bin/bash
# 3x3 凝胶阵列实验：随机初始相位，观察集体相位演化
# runstep=200000 (~4个振荡周期), 预计运行约2小时
set -e
cd "$(dirname "$0")"
mkdir -p data_array
rm -f data_array/gel*bodycenter* data_array/gel*vm* data_array/gel*rn* \
       data_array/gel*um* data_array/gel*wm* data_array/gel*Fn* \
       data_array/gel*Veln* data_array/gel*rm* data_array/Conc* data_array/Velb*

# 临时把 data 指向 data_array
ln -sfn data_array data 2>/dev/null || true

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting 3x3 array run, RAND_SEED=42"
IBM_BETA=3.2 IBM_RAMP_STEPS=0 GEL_AZ0=10 RAND_SEED=42 ./gLSM_array 2>&1 | tee run_array.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done"
