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
#include <cstring>

using namespace std;
namespace fs = std::filesystem;

#define runstep 200000

int main(int argc, char** argv)
{
    cudaSetDevice(0);

    fs::path baseDir = fs::current_path();
    fs::path dataDir = baseDir / "data";
    fs::create_directories(dataDir);
    fs::current_path(dataDir);

    // -------------------------------------------------------------------------
    // 3x3 square array of gels
    // Gel size: 20x20x4, fluid domain: 140x140x10
    // Gel centers: x=30,70,110  y=30,70,110  z=7
    // -------------------------------------------------------------------------
    int3    fluidSize = make_int3(140, 140, 10);
    int3    gelSize   = make_int3(20, 20, 4);

    const int NGEL = 9;
    int xs[3] = {30, 70, 110};
    int ys[3] = {30, 70, 110};

    // Random initial phase delays (uniform over [0, T), T=50000)
    // Seed from env var RAND_SEED for reproducibility, default 42
    long seed = 42;
    if (const char* env = getenv("RAND_SEED")) seed = atol(env);
    srand((unsigned)seed);
    double T_period = 50000.0;

    int delays[NGEL];
    for (int i = 0; i < NGEL; ++i) {
        double r = (double)rand() / RAND_MAX;
        delays[i] = (int)(r * T_period);
    }

    // Print layout
    printf("=== 3x3 Gel Array ===\n");
    printf("Fluid domain: %d x %d x %d\n", fluidSize.x, fluidSize.y, fluidSize.z);
    printf("Gel ID  pos(x,y,z)      delay   phase(pi)\n");
    int gelIdx = 0;
    for (int iy = 0; iy < 3; iy++) {
        for (int ix = 0; ix < 3; ix++) {
            printf("  gel%d  (%3d,%3d,%d)    %6d  %.4f\n",
                gelIdx+1, xs[ix], ys[iy], 7,
                delays[gelIdx], delays[gelIdx]/T_period*2.0);
            gelIdx++;
        }
    }
    fflush(stdout);

    // Create gels
    vector<Gel*> gels;
    gelIdx = 0;
    for (int iy = 0; iy < 3; iy++) {
        for (int ix = 0; ix < 3; ix++) {
            double3 pos = make_double3((double)xs[ix], (double)ys[iy], 7.0);
            Gel* g = new Gel(gelSize, pos, 1, gelIdx+1, 0);
            if (delays[gelIdx] > 0)
                g->resetToQuiescent();
            gels.push_back(g);
            gelIdx++;
        }
    }

    Coupler* coupler = new Coupler(gels);
    Fluid* fluid = new Fluid(fluidSize, 0, coupler);

    for (int iter = 0; iter <= runstep; ++iter)
    {
        // Fire excitation pulses at their respective delays
        for (int gi = 0; gi < NGEL; gi++) {
            if (delays[gi] > 0 && iter == delays[gi])
                gels[gi]->fireExcitationPulse();
        }

        for (int gi = 0; gi < NGEL; gi++) {
            Gel* g = gels[gi];
            // Skip chemistry/elasticity until delay fires
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
                cudaMemcpy(g->m_hVeln, g->m_dVeln, g->m_numGelNodes * sizeof(double3), cudaMemcpyDeviceToHost);
                for (int ni = 0; ni < g->m_numGelNodes; ++ni) {
                    float v = (float)sqrt(g->m_hVeln[ni].x*g->m_hVeln[ni].x
                                        +g->m_hVeln[ni].y*g->m_hVeln[ni].y
                                        +g->m_hVeln[ni].z*g->m_hVeln[ni].z);
                    if (v > vg_max) vg_max = v;
                }
            }
            float nsub = (float)fluid->Nsub;
            float h    = fluid->h_fp->h;
            float lbm2gel = nsub / h;
            printf("iter=%6d  Vg_max=%.5f  lbm2gel=%.2f\n", iter, vg_max, lbm2gel);
            fflush(stdout);
        }

        for (auto g : gels)
            g->writeFiles(iter);
        fluid->writeFiles(iter);
    }

    for (auto g : gels) delete g;
    delete fluid;
    delete coupler;
    fs::current_path(baseDir);

    return 0;
}
