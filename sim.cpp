#include "gel.h"
#include "fluid.h"
#include "coupling.h"

#include <string>

using namespace std;

#define runstep 100000

int main(int argc, char** argv)
{
    cudaSetDevice(0);
    int3    fluidSize = make_int3(301, 101, 21);
    int3    gelSize1 = make_int3(16, 16, 5);
    int3    gelSize2 = make_int3(26, 26, 5);
    double3 gelPos1 = make_double3(50.0, 25.0, 5.0);
    double3 gelPos2 = make_double3(100.0, 25.0, 5.0);
    string  gelType1 = "NIPAAM";
    string  gelType2 = "PAAM";
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
    return 0;
}
