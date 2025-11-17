#include <thread>
#include <cuda_runtime.h>
#include <string>

struct GelParams {
	int LX, LY, LZ;
	double f, q, ep, P1, P2;
	double dt, dtx;
	double dx, dy, dz;
	double CH0, CH1, CHS, C0, AZ0, FA0;
	double uss, vss, wss, I;
	int3 rn_offset[27];
	int3 um_offset_noflux[27];
	int3 um_offset_periodic[27];
};

class Gel {
public:
    Gel(int3 gelSize, double3 gelPosition, std::string gelType, int gel_id, int time);
    ~Gel();

    void _initialize(int time);
    void update(long long int solverIterations);
    void _finalize();
        void writeFiles(double time);

    cudaStream_t stream() const;

public:
    void allocateHostStorage();
    void allocateDeviceStorage();
    void freeHostMemory();
    void freeDeviceMemory();
    void setInitValue();
    double fu_h(double u, double v, double w, double phi);
    void steadyStateValue(double& u, double& v, double& w, double phi);
    int get_index(int xi, int yi, int zi, int size);
    void setType(int* a, int size);
    void copyDataToDevice();
    void copyDataToHost();
	void recordData(int time);
    void setChemicalWave(int type);
    int idx3(int x, int y, int z, int Nx, int Ny);
	void buildBoundaryIndex();

	// CPU data
	//paramas
	GelParams* m_hgp;
	//chemical variables
	double* m_hum;
	double* m_hvm;
	double* m_hwm;
	double* m_hvm_center;
	double* m_hwm_center;

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
	int* m_hbIndex;
	double m_htime = 0;

	// GPU data
	//paramas
	GelParams* m_dgp;
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

	int* m_dmap_node;
	int* m_dmap_element;
	int* m_dbIndex;
	double* m_dtime;

public:
	// params
	int m_gelId;
	std::string m_gelType;
	double3 m_gelPosition;
	int3 m_gelSize;
	double m_dt;
	int m_df;
	int m_numGelElements;
	int m_numGelNodes;
	int m_boundaryCount;
	std::thread m_file_writer_thread;
	cudaStream_t m_gel_stream;
	dim3 m_blockDim;
	dim3 m_gridDim_1;
	dim3 m_gridDim0;
	dim3 m_gridDim1;
	dim3 m_gridDim2;
};
