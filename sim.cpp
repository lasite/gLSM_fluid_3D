#include "gel.h"
#include "fluid.h"
#include "coupling.h"

#include <string>
#include <vector>
#include <filesystem>

using namespace std;
namespace fs = std::filesystem;

#define runstep 100000

int main(int argc, char** argv)
{
    cudaSetDevice(0);

    fs::path baseDir = fs::current_path();
    fs::path dataDir = baseDir / "data";
    fs::create_directories(dataDir);
    fs::current_path(dataDir);

    int3    fluidSize = make_int3(150, 50, 10);
    int3    gelSize1 = make_int3(15, 15, 4);
    int3    gelSize2 = make_int3(25, 25, 4);
    double3 gelPos1 = make_double3(50.0, 25.0, 5.0);
    double3 gelPos2 = make_double3(100.0, 25.0, 5.0);
    int  gelType1 = 1;
    int  gelType2 = 2;
    int     gelId1 = 1;
    int     gelId2 = 2;
    int     startTime = 0;

    Gel* gel1 = new Gel(gelSize1, gelPos1, gelType1, gelId1, startTime);
    Gel* gel2 = new Gel(gelSize2, gelPos2, gelType2, gelId2, startTime);

    vector<Gel*> gels;
    gels.push_back(gel1);
    gels.push_back(gel2);

    Coupler* coupler = new Coupler(gels);
    Fluid* fluid = new Fluid(fluidSize, startTime, coupler);

    for (int iter = 0; iter <= runstep; ++iter)
    {
        for (auto g : gels) {
            g->stepElasticity(iter);
            g->stepChemistry(iter);
        }
        coupler->packFromGels();
        fluid->stepVelocity(iter);
        fluid->stepConcentration();
        coupler->scatterToGels();

        for (auto g : gels) {
            g->writeFiles(iter);
        }
        fluid->writeFiles(iter);
    }

    delete fluid;
    for (auto g : gels)
        delete g;
    delete coupler;
    fs::current_path(baseDir);

    return 0;
}
