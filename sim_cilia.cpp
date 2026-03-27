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
// 4x4 array of elongated cilia-like gels along Z
// Gel size:      4 x 4 x 24  (aspect ratio 6:1)
// Fluid domain:  100 x 100 x 30
// Gel centers:   x=10,30,50,70  y=10,30,50,70  z=15
//   → gel spans z=3 to z=27 (anchor bottom at z=3, top free)
// Spacing:       20 (4 gel + 16 gap)
// runstep=200000 (~4 oscillation periods), ~2h wall time
// ============================================================
#define runstep 200000

int main(int argc, char** argv)
{
    cudaSetDevice(0);

    fs::path baseDir = fs::current_path();
    fs::path dataDir = baseDir / "data";
    fs::create_directories(dataDir);
    fs::current_path(dataDir);

    int3    fluidSize = make_int3(100, 100, 30);
    int3    gelSize   = make_int3(4, 4, 24);   // elongated along z, aspect ratio 6:1
    double  gel_z_center = 15.0;               // center z → spans z=3 to z=27
    double  anchor_z     = 3.0;                // bottom layer z position

    const int NX = 4, NY = 4;
    const int NGEL = NX * NY;
    int xs[4] = {10, 30, 50, 70};
    int ys[4] = {10, 30, 50, 70};

    // Random initial phase delays (uniform over [0, T), T=50000)
    long seed = 42;
    if (const char* env = getenv("RAND_SEED")) seed = atol(env);
    srand((unsigned)seed);
    double T_period = 50000.0;

    int delays[NGEL];
    for (int i = 0; i < NGEL; ++i)
        delays[i] = (int)((double)rand() / RAND_MAX * T_period);

    // Print layout
    printf("=== 4x4 Cilia-like Gel Array (elongated along Z, anchored bottom) ===\n");
    printf("Fluid domain: %d x %d x %d\n", fluidSize.x, fluidSize.y, fluidSize.z);
    printf("Gel size: %d x %d x %d  (aspect ratio 6:1, z-elongated)\n", gelSize.x, gelSize.y, gelSize.z);
    printf("Anchor z=%.1f  Gel center z=%.1f  Top z=%.1f\n",
           anchor_z, gel_z_center, gel_z_center + gelSize.z/2.0);
    printf("Gel ID  pos(x,y,z)    delay   phase(pi)\n");
    int gelIdx = 0;
    for (int iy = 0; iy < NY; iy++)
        for (int ix = 0; ix < NX; ix++) {
            printf("  gel%02d (%3d,%3d,%4.1f)  %6d  %.4f\n",
                gelIdx+1, xs[ix], ys[iy], gel_z_center,
                delays[gelIdx], delays[gelIdx]/T_period*2.0);
            gelIdx++;
        }
    fflush(stdout);

    // Create gels
    vector<Gel*> gels;
    gelIdx = 0;
    for (int iy = 0; iy < NY; iy++)
        for (int ix = 0; ix < NX; ix++) {
            double3 pos = make_double3((double)xs[ix], (double)ys[iy], gel_z_center);
            Gel* g = new Gel(gelSize, pos, 1, gelIdx+1, 0);
            g->setAnchorZ(anchor_z);   // anchor bottom layer
            if (delays[gelIdx] > 0)
                g->resetToQuiescent();
            gels.push_back(g);
            gelIdx++;
        }

    Coupler* coupler = new Coupler(gels);
    Fluid*   fluid   = new Fluid(fluidSize, 0, coupler);

    for (int iter = 0; iter <= runstep; ++iter)
    {
        // Fire excitation pulses at respective delays
        for (int gi = 0; gi < NGEL; gi++)
            if (delays[gi] > 0 && iter == delays[gi])
                gels[gi]->fireExcitationPulse();

        for (int gi = 0; gi < NGEL; gi++) {
            Gel* g = gels[gi];
            if (delays[gi] > 0 && iter < delays[gi]) {
                g->recordCenter(iter);
                continue;
            }
            g->stepElasticity(iter);
            g->stepChemistry(iter);
            g->recordCenter(iter);
        }

        coupler->packFromGels();
        fluid->stepVelocity(iter);
        fluid->stepConcentration(iter);
        coupler->applyGelRepulsion();
        coupler->scatterToGels();

        if (iter % 1000 == 0) {
            cudaDeviceSynchronize();
            float vg_max = 0.f;
            for (auto g : gels) {
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
            }
            float lbm2gel = (float)fluid->Nsub / fluid->h_fp->h;
            printf("iter=%6d  Vg_max=%.5f  lbm2gel=%.2f\n", iter, vg_max, lbm2gel);
            fflush(stdout);
        }

        for (auto g : gels) g->writeFiles(iter);
        fluid->writeFiles(iter);
    }

    for (auto g : gels) delete g;
    delete fluid;
    delete coupler;
    fs::current_path(baseDir);
    return 0;
}
