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

// ============================================================
// Single bilayer gel debug simulation
// Minimal fluid domain to iterate quickly.
// Gel size:     gelSize via env GEL_SIZE_X/Y/Z (default 2,4,24)
// Fluid domain: 32 x 32 x 32
// Gel position: center of fluid
// Default runstep=10000; override with RUNSTEP env var
// ============================================================
static int get_runstep() {
    if (const char* e = getenv("RUNSTEP")) return atoi(e);
    return 10000;
}
#define runstep get_runstep()

int main(int argc, char** argv)
{
    cudaSetDevice(0);

    fs::path baseDir = fs::current_path();
    fs::path dataDir = baseDir / "data";
    fs::create_directories(dataDir);
    fs::current_path(dataDir);

    // Gel size: default 2x4x24 (narrow in x → easier to bend toward passive side)
    int gsx = 2, gsy = 4, gsz = 24;
    if (const char* e = getenv("GEL_SIZE_X")) gsx = atoi(e);
    if (const char* e = getenv("GEL_SIZE_Y")) gsy = atoi(e);
    if (const char* e = getenv("GEL_SIZE_Z")) gsz = atoi(e);

    int3    gelSize   = make_int3(gsx, gsy, gsz);
    // Fluid domain: default 100x100x30 (same as sim_cilia) for consistent coupling
    // Can override with env FLUID_SIZE_X/Y/Z
    int fdx = 100, fdy = 100, fdz = 30;
    if (const char* e = getenv("FLUID_SIZE_X")) fdx = atoi(e);
    if (const char* e = getenv("FLUID_SIZE_Y")) fdy = atoi(e);
    if (const char* e = getenv("FLUID_SIZE_Z")) fdz = atoi(e);
    int3    fluidSize = make_int3(fdx, fdy, fdz);

    // Gel center: default position same as sim_cilia (center of 100x100 fluid, z=15)
    double gel_x_center = fdx / 2.0;
    double gel_y_center = fdy / 2.0;
    double gel_z_center = 15.0;   // fixed at z=15 (sim_cilia convention)
    double anchor_z     = 3.0;    // fixed at z=3 (sim_cilia convention)

    printf("=== Single Bilayer Gel Debug ===\n");
    printf("Fluid domain: %d x %d x %d\n", fluidSize.x, fluidSize.y, fluidSize.z);
    printf("Gel size: %d x %d x %d\n", gelSize.x, gelSize.y, gelSize.z);
    printf("Gel center: (%.1f, %.1f, %.1f)\n", gel_x_center, gel_y_center, gel_z_center);
    printf("Anchor z = %.1f,  Free tip z = %.1f\n", anchor_z, gel_z_center + gsz/2.0);
    fflush(stdout);

    double3 pos = make_double3(gel_x_center, gel_y_center, gel_z_center);
    Gel* g = new Gel(gelSize, pos, 1, 1, 0);
    g->setAnchorZ(anchor_z);
    // Use resetToQuiescent + delayed pulse (like sim_cilia does)
    // Pulse at iter=500 → BZ wave starts from middle of gel, not from anchor layer
    g->resetToQuiescent();
    const int fire_iter = 500;
    printf("Gel reset to quiescent; excitation pulse at iter=%d\n", fire_iter);
    fflush(stdout);

    vector<Gel*> gels = {g};
    Coupler* coupler = new Coupler(gels);
    Fluid*   fluid   = new Fluid(fluidSize, 0, coupler);

    for (int iter = 0; iter <= runstep; ++iter)
    {
        if (iter == fire_iter)
            g->fireExcitationPulse();

        if (iter < fire_iter) {
            g->recordCenter(iter);
        } else {
            g->stepElasticity(iter);
            g->stepChemistry(iter);
            g->recordCenter(iter);
        }

        coupler->packFromGels();
        fluid->stepVelocity(iter);
        fluid->stepConcentration(iter);
        coupler->applyGelRepulsion();
        coupler->scatterToGels();

        if (iter % 500 == 0) {
            cudaDeviceSynchronize();
            float vg_max = 0.f;
            cudaMemcpy(g->m_hVeln, g->m_dVeln,
                       g->m_numGelNodes * sizeof(double3),
                       cudaMemcpyDeviceToHost);
            for (int ni = 0; ni < g->m_numGelNodes; ++ni) {
                float v = (float)sqrt(
                    g->m_hVeln[ni].x*g->m_hVeln[ni].x +
                    g->m_hVeln[ni].y*g->m_hVeln[ni].y +
                    g->m_hVeln[ni].z*g->m_hVeln[ni].z);
                if (v > vg_max) vg_max = v;
            }

            // Report tip deflection from bodycenter
            float lbm2gel = (float)fluid->Nsub / fluid->h_fp->h;
            printf("iter=%6d  Vg_max=%.5f  lbm2gel=%.2f\n", iter, vg_max, lbm2gel);
            fflush(stdout);
        }

        g->writeFiles(iter);
        fluid->writeFiles(iter);
    }

    delete g;
    delete fluid;
    delete coupler;
    fs::current_path(baseDir);
    return 0;
}
