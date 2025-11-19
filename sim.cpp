#include "gel.h"
#include "fluid.h"
#include "coupling.h"

#include <string>

using namespace std;

#define runstep 10000

int main(int argc, char** argv)
{
    cudaSetDevice(1);
    int3    fluidSize = make_int3(101, 101, 101);
    int3    gelSize1 = make_int3(6, 6, 6);
    int3    gelSize2 = make_int3(6, 6, 6);
    double3 gelPos1 = make_double3(15.0, 15.0, 15.0);
    double3 gelPos2 = make_double3(35.0, 35.0, 35.0);
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

    Fluid* fluid = new Fluid(fluidSize, startTime);
    Coupler* coupler = new Coupler(gels, fluid);

    for (long long int solverIterations = 0; solverIterations <= runstep; ++solverIterations)
    {
        for (auto g : gels) {
            g->update(solverIterations);
            g->recordUpdateCompleteEvent();
        }
        coupler->packFromGels();
        coupler->transferConcentration();
        for (int kk = 0; kk < fluid->Nsub; kk++) {
        //for (int kk = 0; kk < 1; kk++) {
            const bool spreadConcentration = (kk == 0);
            coupler->update(solverIterations, spreadConcentration);
            fluid->update(solverIterations);
        }
        fluid->convectionAndDiffusion();
        /*coupler->applyGelRepulsion();*/
        coupler->scatterToGels();
    }
    delete fluid;
    for (auto g : gels)
        delete g;
    delete coupler;
    return 0;
}
