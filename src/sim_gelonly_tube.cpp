// ============================================================
// Gel-Only Tube Simulation (Method B: Void Tube Mask)
//
// Demonstrates a single gel block with a hollowed-out interior
// using the built-in tube mask algorithm.
//
// Two modes can be tested:
//   TUBE_CIRCULAR=0  → Square/Rectangular tube
//   TUBE_CIRCULAR=1  → Cylindrical tube
// ============================================================

#include "gel.h"
#include <string>
#include <vector>
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace std;
namespace fs = std::filesystem;

int main(int argc, char** argv)
{
    cudaSetDevice(0);

    fs::path baseDir = fs::current_path();
    fs::path dataDir = baseDir / "data_gelonly_tube";
    fs::create_directories(dataDir);
    fs::current_path(dataDir);

    // ── Simulation parameters ────────────────────────────────
    int runstep = 20000;
    if (const char* e = getenv("RUNSTEP")) runstep = atoi(e);

    int wave_delay = 500;
    if (const char* e = getenv("WAVE_DELAY")) wave_delay = atoi(e);

    int wall_thickness = 4;
    if (const char* e = getenv("WALL_THICKNESS")) wall_thickness = atoi(e);

    bool circular = false;
    if (const char* e = getenv("TUBE_CIRCULAR")) circular = atoi(e) != 0;

    // ── Gel geometry ─────────────────────────────────────────
    // A long block: 200 x 30 x 30. We will hollow it out.
    int gel_lx = 200;
    int3 gelSize  = make_int3(gel_lx, 30, 30);
    double3 pos   = make_double3(gel_lx / 2.0, 15.0, 15.0);

    printf("=== Gel-Only Tube Simulation ===\n");
    printf("Gel outer size: %d × %d × %d\n", gelSize.x, gelSize.y, gelSize.z);
    printf("Wall thickness: %d elements\n", wall_thickness);
    printf("Tube cross-sec: %s\n", circular ? "Cylinder" : "Square");
    printf("runstep=%d  wave_delay=%d\n", runstep, wave_delay);
    fflush(stdout);

    // ── Create gel & build tube mask ─────────────────────────
    Gel* tube = new Gel(gelSize, pos, 1, 1, 0);
    tube->resetToQuiescent();
    tube->buildTubeMask(wall_thickness, circular);

    // ── Main time-stepping loop ───────────────────────────────
    for (int iter = 0; iter <= runstep; ++iter)
    {
        if (iter == wave_delay) {
            tube->fireExcitationPulse();
            printf("iter=%d: BZ excitation pulse fired\n", iter);
            fflush(stdout);
        }

        if (iter < wave_delay) {
            tube->recordCenter(iter);
        } else {
            tube->stepElasticity(iter);
            tube->stepChemistry(iter);
            tube->recordCenter(iter);
        }

        if (iter % 1000 == 0) {
            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA Error: %s\n", cudaGetErrorString(err));
            }
            float max_v = 0.0f;
            cudaMemcpy(tube->m_hVeln, tube->m_dVeln, tube->m_numGelNodes * sizeof(double3), cudaMemcpyDeviceToHost);
            for (int ni = 0; ni < tube->m_numGelNodes; ++ni) {
                float v = sqrt(tube->m_hVeln[ni].x*tube->m_hVeln[ni].x +
                               tube->m_hVeln[ni].y*tube->m_hVeln[ni].y +
                               tube->m_hVeln[ni].z*tube->m_hVeln[ni].z);
                if (v > max_v) max_v = v;
            }
            
            cudaMemcpy(tube->m_hum, tube->m_dum, tube->m_numGelElements * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(tube->m_hvm, tube->m_dvm, tube->m_numGelElements * sizeof(double), cudaMemcpyDeviceToHost);
            int test_gi = tube->get_index(1, 1, tube->m_gelNodeGrid.z/2, 1);
            int void_gi = tube->get_index(1, tube->m_gelNodeGrid.y/2, tube->m_gelNodeGrid.z/2, 1);

            printf("iter=%6d  Gel V_max = %.5f  um[solid]=%.5f vm[solid]=%.5f | um[void]=%.5f vm[void]=%.5f\n", iter, max_v, tube->m_hum[test_gi], tube->m_hvm[test_gi], tube->m_hum[void_gi], tube->m_hvm[void_gi]);
            

            fflush(stdout);
        }

        // Write output files
        tube->writeFiles(iter);
    }

    // ── Cleanup ───────────────────────────────────────────────
    delete tube;
    fs::current_path(baseDir);
    return 0;
}
