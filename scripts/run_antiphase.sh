#!/bin/bash
# 反相实验：GEL2_DELAY=24575 → Δφ≈π (T≈49150 iters)
# runstep=150000 (3个振荡周期)
# 跑之前先清数据目录
set -e
cd "$(dirname "$0")"
mkdir -p data
rm -f data/gel*bodycenter* data/gel*vm* data/gel*rn* data/gel*um* \
       data/gel*wm* data/gel*Fn* data/gel*Veln* data/gel*rm* data/Conc* data/Velb*
echo "[$(date)] Starting anti-phase run GEL2_DELAY=24575"
GEL2_DELAY=24575 ./gLSM_fluid_3D 2>&1 | tee run_antiphase.log
echo "[$(date)] Done"
