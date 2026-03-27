#pragma once
#include <cuda_runtime.h>
#include <vector>
class Gel;

struct CouplerParams {
	int  M;
	float delta;
	float gamma;
	float gel_to_lbm_vel;
	float lbm_to_gel_vel;
};

class Coupler {
public:
    Coupler(std::vector<Gel*>& gels);
    ~Coupler();
	void packFromGels();
	void scatterToGels();
	void applyGelRepulsion();
	void _initialize();
	void _finalize();

protected: 
	void allocateHostStorage();
	void allocateDeviceStorage();
	void copyDataToDevice();
	void setInitValue();
	void freeHostMemory();
	void freeDeviceMemory();

public:
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
	float3* d_Fdrag_all_;
	float3* d_Frep_all_;
	float* d_Cl_all_;
	float* d_Dl_all_;
	float* d_Sl_all_;
	int* d_owner;
	CouplerParams* d_cp;

public:
	double dt = 1e-3;
	int numGels;
	int sumGelBoundaryCount;
    cudaStream_t coupler_stream;
	int threads;
	int blocksM;
	std::vector<Gel*>& gels;
};