#include "gel.h"

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
// Gel-only simulation: NO fluid, NO coupling
// Pure BZ chemistry + elasticity in a single bilayer gel
// Goal: find parameter regime where bilayer produces visible bending
// ============================================================

int main(int argc, char** argv)
{
    cudaSetDevice(0);

    fs::path baseDir = fs::current_path();
    fs::path dataDir = baseDir / "data";
    fs::create_directories(dataDir);
    fs::current_path(dataDir);

    // Gel size
    int gsx = 2, gsy = 4, gsz = 24;
    if (const char* e = getenv("GEL_SIZE_X")) gsx = atoi(e);
    if (const char* e = getenv("GEL_SIZE_Y")) gsy = atoi(e);
    if (const char* e = getenv("GEL_SIZE_Z")) gsz = atoi(e);
    int3 gelSize = make_int3(gsx, gsy, gsz);

    int runstep = 60000;
    if (const char* e = getenv("RUNSTEP")) runstep = atoi(e);

    // Place gel at arbitrary position (no fluid, doesn't matter for physics)
    double gel_x_center = 50.0;
    double gel_y_center = 50.0;
    double gel_z_center = double(gsz) / 2.0 + 3.0;  // anchor at z=3
    double anchor_z     = 3.0;

    printf("=== Gel-Only Bilayer Bending Test ===\n");
    printf("Gel size: %d x %d x %d\n", gelSize.x, gelSize.y, gelSize.z);
    printf("Center: (%.1f, %.1f, %.1f)\n", gel_x_center, gel_y_center, gel_z_center);
    printf("Anchor z = %.1f,  Free tip z = %.1f\n", anchor_z, gel_z_center + gsz/2.0);
    printf("Runstep: %d\n", runstep);
    fflush(stdout);

    double3 pos = make_double3(gel_x_center, gel_y_center, gel_z_center);
    Gel* g = new Gel(gelSize, pos, 1, 1, 0);
    g->setAnchorZ(anchor_z);
    g->resetToQuiescent();

    const int fire_iter = 100;  // fire early since no fluid warm-up needed
    printf("Excitation pulse at iter=%d\n", fire_iter);
    fflush(stdout);

    for (int iter = 0; iter <= runstep; ++iter)
    {
        if (iter == fire_iter)
            g->fireExcitationPulse();

        if (iter >= fire_iter) {
            g->stepChemistry(iter);
            g->stepElasticity(iter);
        }
        g->recordCenter(iter);

        // Print progress every 500 iters
        if (iter % 500 == 0) {
            cudaDeviceSynchronize();
            // Read back node positions to compute tip deflection
            cudaMemcpy(g->m_hrn, g->m_drn,
                       (g->m_gelNodeGrid.x+2)*(g->m_gelNodeGrid.y+2)*(g->m_gelNodeGrid.z+2) * sizeof(double3),
                       cudaMemcpyDeviceToHost);

            int LX = g->m_gelNodeGrid.x;
            int LY = g->m_gelNodeGrid.y;
            int LZ = g->m_gelNodeGrid.z;

            // base = zi=1 (anchor), tip = zi=LZ (top)
            double base_x = 0, tip_x = 0;
            int nb = 0, nt = 0;
            for (int yi = 1; yi <= LY; yi++) {
                for (int xi = 1; xi <= LX; xi++) {
                    int gi_b = xi + yi*(LX+2) + 1*(LX+2)*(LY+2);
                    int gi_t = xi + yi*(LX+2) + LZ*(LX+2)*(LY+2);
                    base_x += g->m_hrn[gi_b].x; nb++;
                    tip_x  += g->m_hrn[gi_t].x; nt++;
                }
            }
            base_x /= nb; tip_x /= nt;
            double dx = tip_x - base_x;

            // Also read vm to check chemistry
            cudaMemcpy(g->m_hvm, g->m_dvm,
                       (LX+2)*(LY+2)*(LZ+2) * sizeof(double),
                       cudaMemcpyDeviceToHost);
            double vm_max = 0;
            for (int zi = 1; zi <= LZ; zi++)
                for (int yi = 1; yi <= LY; yi++)
                    for (int xi = 1; xi <= LX; xi++) {
                        int gi = xi + yi*(LX+2) + zi*(LX+2)*(LY+2);
                        double v = g->m_hvm[gi];
                        if (!isnan(v) && v > vm_max) vm_max = v;
                    }

            printf("iter=%6d  tip_dx=%+.6f  vm_max=%.5f\n", iter, dx, vm_max);
            if (isnan(dx) || isnan(vm_max)) {
                printf("NaN detected! Stopping.\n");
                break;
            }
            fflush(stdout);
        }

        g->writeFiles(iter);
    }

    printf("Done.\n");
    delete g;
    fs::current_path(baseDir);
    return 0;
}
