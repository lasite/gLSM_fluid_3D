#ifndef GELSYSTEM_H
#define GELSYSTEM_H

#include <thread>
#include <cuda_runtime.h>

#include "gelParams.h"
#include "gel_kernel.cuh"

using namespace std;
//Gel system class
class GelSystem {
public:
	GelSystem(int3 gelSize, int3 fluidSize, int time, int i, int j, int k);
	~GelSystem();
	void _initialize(int time);
	void update(long long int solverIterations);
	void _finalize();
	void writeFiles(double time);


protected:  // methods
	void allocateHostStorage();
	void allocateDeviceStorage();
	void freeHostMemory();
	void freeDeviceMemory();
	void setInitValue();
	void setGoonValue(int time);
	double fu_h(double u, double v, double w, double phi);
	void steadyStateValue(double& u, double& v, double& w, double phi);
	int get_index(int xi, int yi, int zi, int size);
	void setType(int* a, int size);
	void copyDataToDevice();
	void copyDataToHost();
	void recordData(int time);
	void recordCenterElement(double time);
	void setChemicalWave(int type);
	int idx3(int x, int y, int z, int Nx, int Ny);
	void buildBoundaryIndex();


//protected:  // data
public:
	// CPU data
	//chemical variables
	double* m_hum;
	double* m_hvm;
	double* m_hwm;
	double* m_hvm_center;
	double* m_hvm_center_z;
	double* m_hwm_center;
	double3* m_hfilament;

	//dynamics variables
	double3* m_hrn;
	double3* m_hrm;
	double3* m_hrm_center;
	double3* m_hFn_center;
	double3* m_hVeln_center;
	double3* m_hVeln;
	double3* m_hFn;

	int* m_hmap_node;
	int* m_hmap_element;

	double m_htime = 0;

	// GPU data
	//chemical variables
	double* m_dum;
	double* m_dum_norm;
	double* m_dun_norm;
	double* m_dvm;
	double* m_dvm_norm;
	double* m_dvn_norm;
	double* m_dwm;
	double* m_dvm_center;
	double* m_dwm_center;
	double* m_dwmp;
	double* m_dT0m;
	double* m_dT1m;
	double* m_dT2m;
	double3* m_dfilament;

	//dynamics variables
	double3* m_drn;
	double3* m_drm;
	double3* m_drm_center;
	double3* m_dFn_center;
	double3* m_dVeln_center;
	double3* m_dVeln;
	double3* m_dVels;
	double3* m_dFn;
	double3* m_drm_loc;
	double3* m_dnmSm;
	double* m_dVolm;
	double* m_dPrem;
	double* m_dc0;

	int* m_dmap_node;
	int* m_dmap_element;

	double* m_dtime;

public:
	// params
	bool m_bInitialized;
	int3 m_gelSize;
	double m_dt;
	int m_df;
	int m_numGelElements;
	int m_numGelNodes;
	GelParams m_params;
	cudaStream_t m_gel_stream;
	thread m_file_writer_thread;
	thread filament_writer_thread;
	dim3 m_blockDim;
	dim3 m_gridDim_1;
	dim3 m_gridDim0;
	dim3 m_gridDim1;
	dim3 m_gridDim2;
	bool flag = false;
	bool result = true;
	int m_count;
	unsigned int* d_hitCnt;

	protected:  // data
		// CPU data
		float3* h_u;
		int* h_bIndex;
		// GPU data
		int* d_bIndex;
		float* d_f;
		float* d_fpost;
		float* d_fnext;
		float* d_rho;
		float3* d_u;
		float3* d_F;
		float3* d_F_ibm;
		float3* d_F_tot;
		float3* d_lag;
		float3* d_Ul;
		float3* d_Vl;
		float3* d_Fl;
		float* d_A;
		float* d_partN;
		float* d_partM;


public:
	// params
	int3 fluidSize;
	int3 gelSize;
	int df;
	int N;
	int M;
	int Nd;
	float dt_fluid;
	int Nsub;
	float niu;
	float dx_fluid;
	FluidParams fp;
	cudaStream_t fluid_stream;
	thread file_writer_thread;
	int threads;
	int blocksN;
	int blocksM;
};

#endif  // __GELSYSTEM_H__