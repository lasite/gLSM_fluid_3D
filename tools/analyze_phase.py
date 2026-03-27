#!/usr/bin/env python3
"""Analyze phase difference between gel1 and gel2 from bodycenter data."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load data (columns: iter, cx, cy, cz, vx, vy, vz, fx, fy, fz, vm, ...)
g1 = np.loadtxt('data/gel1bodycenter.dat')
g2 = np.loadtxt('data/gel2bodycenter.dat')

iters = g1[:, 0]
vm1 = g1[:, 10]
vm2 = g2[:, 10]

# Find peaks for phase estimation
peaks1, _ = find_peaks(vm1, height=np.max(vm1)*0.3, distance=200)
peaks2, _ = find_peaks(vm2, height=np.max(vm2)*0.3, distance=200)

print(f"Gel1 peaks at iters: {iters[peaks1]}")
print(f"Gel2 peaks at iters: {iters[peaks2]}")

# Estimate period from gel1
if len(peaks1) >= 2:
    periods1 = np.diff(iters[peaks1])
    T1 = np.mean(periods1)
    print(f"Gel1 mean period T1 = {T1:.1f} iters")
else:
    T1 = None
    print("Not enough peaks for period estimation in gel1")

if len(peaks2) >= 2:
    periods2 = np.diff(iters[peaks2])
    T2 = np.mean(periods2)
    print(f"Gel2 mean period T2 = {T2:.1f} iters")
else:
    T2 = None

# Phase difference from matched peaks (late-stage stable window)
if len(peaks1) >= 1 and len(peaks2) >= 1 and T1:
    # Use last few peaks for stable phase diff
    n = min(len(peaks1), len(peaks2), 5)
    diffs = []
    for i in range(1, n+1):
        dt = iters[peaks2[-i]] - iters[peaks1[-i]]
        dphi = (dt / T1) * 2 * np.pi
        # Wrap to [-pi, pi]
        dphi = (dphi + np.pi) % (2*np.pi) - np.pi
        diffs.append(dphi)
    mean_dphi = np.mean(diffs)
    print(f"Late-stage Δφ (gel2-gel1) = {mean_dphi:.4f} rad = {mean_dphi/np.pi:.4f}π")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

ax = axes[0]
ax.plot(iters, vm1, label='Gel1 vm', color='royalblue')
ax.plot(iters, vm2, label='Gel2 vm', color='tomato', alpha=0.8)
if len(peaks1): ax.plot(iters[peaks1], vm1[peaks1], 'b^', ms=6)
if len(peaks2): ax.plot(iters[peaks2], vm2[peaks2], 'rv', ms=6)
ax.set_xlabel('Iteration')
ax.set_ylabel('vm (a.u.)')
ax.set_title('Anti-phase Experiment: Gel vm(t) — GEL2_DELAY=24575 (Δφ₀≈π)')
ax.legend()
ax.grid(True, alpha=0.3)

# Compute rolling phase difference
ax2 = axes[1]
# Use Hilbert transform for instantaneous phase
from scipy.signal import hilbert
# Only use gel2 after it starts (skip quiescent period)
start_idx = np.searchsorted(iters, 24575)
vm1_sig = vm1[start_idx:] - np.mean(vm1[start_idx:])
vm2_sig = vm2[start_idx:] - np.mean(vm2[start_idx:])
phi1 = np.unwrap(np.angle(hilbert(vm1_sig)))
phi2 = np.unwrap(np.angle(hilbert(vm2_sig)))
dphi = phi2 - phi1
# Wrap to [-pi, pi]
dphi_wrapped = (dphi + np.pi) % (2*np.pi) - np.pi

ax2.plot(iters[start_idx:], dphi_wrapped / np.pi, color='purple')
ax2.axhline(0, color='gray', linestyle='--', alpha=0.5, label='In-phase')
ax2.axhline(1, color='orange', linestyle='--', alpha=0.5, label='Anti-phase (π)')
ax2.axhline(-1, color='orange', linestyle='--', alpha=0.5)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Δφ / π')
ax2.set_title('Instantaneous Phase Difference (gel2 - gel1)')
ax2.set_ylim(-1.5, 1.5)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('antiphase_analysis.png', dpi=150)
print("Saved: antiphase_analysis.png")
plt.close()
