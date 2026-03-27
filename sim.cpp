#include "gel.h"
#include "fluid.h"
#include "coupling.h"

#include <string>
#include <vector>
#include <filesystem>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;
namespace fs = std::filesystem;

#define runstep 150000

int main(int argc, char** argv)
{
    cudaSetDevice(0);

    fs::path baseDir = fs::current_path();
    fs::path dataDir = baseDir / "data";
    fs::create_directories(dataDir);
    fs::current_path(dataDir);

    int3    fluidSize = make_int3(150, 50, 10);
    int3    gelSize1 = make_int3(20, 20, 4);
    int3    gelSize2 = make_int3(20, 20, 4);
    double3 gelPos1 = make_double3(50.0, 25.0, 5.0);
    double3 gelPos2 = make_double3(100.0, 25.0, 5.0);
    int     gelType1 = 1;
    int     gelType2 = 1;
    int     gelId1 = 1;
    int     gelId2 = 2;
    int     startTime = 0;

    // GEL2_DELAY: gel2 stays quiescent for this many iters, then fires its
    // chemical excitation pulse.  This gives a controllable initial phase
    // offset: Δφ = 2π × GEL2_DELAY / T  (T ≈ 49150 iters at default params).
    // Examples: 12300 → Δφ≈π/2,  24600 → Δφ≈π,  0 → no delay (default).
    int gel2_delay = 0;
    if (const char* env = getenv("GEL2_DELAY"))
        gel2_delay = atoi(env);

    Gel* gel1 = new Gel(gelSize1, gelPos1, gelType1, gelId1, startTime);
    Gel* gel2 = new Gel(gelSize2, gelPos2, gelType2, gelId2, startTime);

    // If delay requested, reset gel2 to quiescent state; it will be
    // re-excited at iter == gel2_delay.
    if (gel2_delay > 0)
        gel2->resetToQuiescent();

    vector<Gel*> gels;
    gels.push_back(gel1);
    gels.push_back(gel2);

    Coupler* coupler = new Coupler(gels);
    Fluid* fluid = new Fluid(fluidSize, startTime, coupler);

    for (int iter = startTime * gels[0]->m_df; iter <= runstep; ++iter)
    {
        // Trigger gel2 excitation pulse at the requested delay step.
        if (gel2_delay > 0 && iter == gel2_delay)
            gel2->fireExcitationPulse();

        for (auto g : gels) {
            // Skip gel2 chemistry/elasticity until its excitation delay fires.
            // This prevents the unstable fixed point from self-exciting early.
            if (g == gel2 && gel2_delay > 0 && iter < gel2_delay) {
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
            // 从 GPU 抓一个凝胶节点速度 & 流体速度，打印当前速度比
            cudaDeviceSynchronize();
            // 用 host 上的 Velb 最大值近似
            float vg_max = 0.f, ub_max = 0.f;
            for (auto g : gels) {
                cudaMemcpy(g->m_hVeln, g->m_dVeln, g->m_numGelNodes * sizeof(double3), cudaMemcpyDeviceToHost);
                for (int ni = 0; ni < g->m_numGelNodes; ++ni) {
                    float v = (float)sqrt(g->m_hVeln[ni].x*g->m_hVeln[ni].x
                                        +g->m_hVeln[ni].y*g->m_hVeln[ni].y
                                        +g->m_hVeln[ni].z*g->m_hVeln[ni].z);
                    if (v > vg_max) vg_max = v;
                }
            }
            // fluid Velb is on GPU; read via fluid's host buffer
            float nsub = (float)fluid->Nsub;
            float h    = fluid->h_fp->h;
            float dt   = (float)gels[0]->m_dt;
            float lbm2gel = nsub / h;
            // ub_max from last written snapshot is stale; just report vg here
            printf("iter=%6d  Vg_max=%.5f  lbm2gel=%.2f\n", iter, vg_max, lbm2gel);
            fflush(stdout);
        }

        for (auto g : gels) {
            g->writeFiles(iter);
        }
        fluid->writeFiles(iter);
    }
    for (auto g : gels)
        delete g;
    delete fluid;
    delete coupler;
    fs::current_path(baseDir);

    return 0;
}
