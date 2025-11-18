#pragma once
#include <cuda_runtime.h>
#include <vector>
//#include "fluid.h"
class Gel;
class Fluid;

struct CouplerParams {
	int  M;
	int3   L;
	float  h;
	float beta;
	float delta;
	float gamma;
};

class Coupler {
public:
    Coupler(std::vector<Gel*>& gels, Fluid* fluid);
    ~Coupler();
	void packFromGels();
	void scatterToGels();
	void applyGelRepulsion();
	void update(long long int solverIterations);
	void _initialize();
	void _finalize();

protected: 
	void allocateHostStorage();
	void allocateDeviceStorage();
	void copyDataToDevice();
	void setInitValue();
	void freeHostMemory();
	void freeDeviceMemory();

protected:
	// CPU data
	CouplerParams* h_cp;
	std::vector<int> h_offsets;
	std::vector<int> h_gelBoundaryCount;
	std::vector<int> h_owner;
	// GPU data
	int* d_bIndex_all_;
	float3* d_lag_all_;
	float3* d_Ul_all_;
	float3* d_Vl_all_;
	float3* d_Fl_all_;
	float* d_Cl_all_;
	float* d_Dl_all_;
	float* d_partN;
	float* d_A;
	int* d_owner;
	CouplerParams* d_cp;

public:
	double dt = 1e-3;
	int numGels;
	int sumGelBoundaryCount;
        cudaStream_t coupler_stream;
        cudaEvent_t pack_complete_event;
        cudaEvent_t ibm_complete_event;
	int threads;
	int blocksM;
	std::vector<Gel*>& gels;
	Fluid* fluid;
};