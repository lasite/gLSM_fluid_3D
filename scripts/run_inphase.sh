#!/bin/bash
# 同相对照：GEL2_DELAY=0
set -e
cd "$(dirname "$0")"
mkdir -p data
rm -f data/gel*bodycenter* data/gel*vm* data/gel*rn* data/gel*um* \
       data/gel*wm* data/gel*Fn* data/gel*Veln* data/gel*rm* data/Conc* data/Velb*
echo "[$(date)] Starting in-phase run GEL2_DELAY=0"
GEL2_DELAY=0 ./gLSM_fluid_3D 2>&1 | tee run_inphase.log
echo "[$(date)] Done"
