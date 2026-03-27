// ============================================================
// Peristaltic Pumping Simulation (Method A: Two Flat Slabs)
//
// Geometry:
//   Fluid channel:  Lx x Ly x Lz = 200 x 40 x 30 (physical units)
//   LBM grid:       (Lx/h+1) x (Ly/h+1) x (Lz/h+1)
//
//   Top gel slab:   gelSize 200×40×6, center at z=26 → spans z=23..29
//   Bot gel slab:   gelSize 200×40×6, center at z= 4 → spans z=1.. 7
//   Internal fluid channel: z ∈ [7, 23]  (channel height ≈ 16 units)
//
// Physics:
//   BZ wave excited simultaneously at X=0 face of both slabs.
//   Wave propagates in +X direction → peristaltic squeezing → net +X flow.
//
// Env overrides:
//   RUNSTEP   : total iterations (default 150000, ~3 oscillation periods)
//   GEL_SIZE_X: gel X extent (default 200, should match fluid Lx)
//   WAVE_DELAY: iterations after start before firing excitation (default 500)
// ============================================================

#include "gel.h"
#include "fluid.h"
#include "coupling.h"

#include <string>
#include <vector>
#include <filesystem>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include <cstdlib>

using namespace std;
namespace fs = std::filesystem;

int main(int argc, char** argv)
{
    cudaSetDevice(0);

    fs::path baseDir = fs::current_path();
    fs::path dataDir = baseDir / "data";
    fs::create_directories(dataDir);
    fs::current_path(dataDir);

    // ── Simulation parameters ────────────────────────────────
    int runstep = 150000;
    if (const char* e = getenv("RUNSTEP")) runstep = atoi(e);

    int gel_lx = 200;
    if (const char* e = getenv("GEL_SIZE_X")) gel_lx = atoi(e);

    int wave_delay = 500;   // let fluid settle before firing pulse
    if (const char* e = getenv("WAVE_DELAY")) wave_delay = atoi(e);

    // ── Fluid domain ─────────────────────────────────────────
    // Physical size: 200 × 40 × 30; LBM grid follows h inside Fluid constructor
    int3 fluidSize = make_int3(gel_lx, 40, 30);

    // ── Gel geometry ─────────────────────────────────────────
    // Each slab: Lx × 40 × 6  (same X extent as fluid, Y full width, 6 thick in Z)
    int3  gelSize = make_int3(gel_lx, 40, 6);

    // Channel center is at z=15 (half of fluidSize.z=30).
    // channel half-height ≈ 8, gel half-thickness = 3
    // Top slab center: 15 + 8 + 3 = 26   → spans z ≈ 23..29
    // Bot slab center: 15 - 8 - 3 =  4   → spans z ≈  1.. 7
    double3 posTop = make_double3(gel_lx / 2.0, 20.0, 26.0);
    double3 posBot = make_double3(gel_lx / 2.0, 20.0,  4.0);

    printf("=== Peristaltic Pump (Two Flat Slabs) ===\n");
    printf("Fluid domain  : %d × %d × %d\n", fluidSize.x, fluidSize.y, fluidSize.z);
    printf("Gel size      : %d × %d × %d\n", gelSize.x, gelSize.y, gelSize.z);
    printf("Top gel center: (%.1f, %.1f, %.1f)\n", posTop.x, posTop.y, posTop.z);
    printf("Bot gel center: (%.1f, %.1f, %.1f)\n", posBot.x, posBot.y, posBot.z);
    printf("runstep=%d  wave_delay=%d\n", runstep, wave_delay);
    fflush(stdout);

    // ── Create gels ──────────────────────────────────────────
    // Both slabs start quiescent; excitation pulse fired at iter==wave_delay.
    Gel* gelTop = new Gel(gelSize, posTop, 1, 1, 0);
    Gel* gelBot = new Gel(gelSize, posBot, 1, 2, 0);
    gelTop->resetToQuiescent();
    gelBot->resetToQuiescent();

    vector<Gel*> gels = { gelTop, gelBot };

    Coupler* coupler = new Coupler(gels);
    Fluid*   fluid   = new Fluid(fluidSize, 0, coupler);

    // ── Main time-stepping loop ───────────────────────────────
    for (int iter = 0; iter <= runstep; ++iter)
    {
        // Fire simultaneous excitation at X=0 end to seed a +X travelling wave.
        if (iter == wave_delay) {
            gelTop->fireExcitationPulse();
            gelBot->fireExcitationPulse();
            printf("iter=%d: BZ excitation pulse fired on both slabs\n", iter);
            fflush(stdout);
        }

        // Advance gel chemistry & elasticity (skip before excitation)
        for (auto g : gels) {
            if (iter < wave_delay) {
                g->recordCenter(iter);
                continue;
            }
            g->stepElasticity(iter);
            g->stepChemistry(iter);
            g->recordCenter(iter);
        }

        // Couple gel ↔ fluid
        coupler->packFromGels();
        fluid->stepVelocity(iter);
        fluid->stepConcentration(iter);
        coupler->applyGelRepulsion();
        coupler->scatterToGels();

        // Progress report every 1000 steps
        if (iter % 1000 == 0) {
            cudaDeviceSynchronize();
            float vg_max = 0.f;
            for (auto g : gels) {
                cudaMemcpy(g->m_hVeln, g->m_dVeln,
                           g->m_numGelNodes * sizeof(double3),
                           cudaMemcpyDeviceToHost);
                for (int ni = 0; ni < g->m_numGelNodes; ++ni) {
                    float v = (float)sqrt(
                        g->m_hVeln[ni].x * g->m_hVeln[ni].x +
                        g->m_hVeln[ni].y * g->m_hVeln[ni].y +
                        g->m_hVeln[ni].z * g->m_hVeln[ni].z);
                    if (v > vg_max) vg_max = v;
                }
            }
            float lbm2gel = (float)fluid->Nsub / fluid->h_fp->h;
            printf("iter=%6d  Vg_max=%.5f  lbm2gel=%.2f\n", iter, vg_max, lbm2gel);
            fflush(stdout);
        }

        // Write output files
        for (auto g : gels) g->writeFiles(iter);
        fluid->writeFiles(iter);
    }

    // ── Cleanup ───────────────────────────────────────────────
    for (auto g : gels) delete g;
    delete fluid;
    delete coupler;
    fs::current_path(baseDir);
    return 0;
}
