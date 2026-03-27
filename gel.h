#include <cuda_runtime.h>
#include <string>
#include <thread>

struct GelParams {
	int LX, LY, LZ;
	double f, q, ep, P1, P2;
	double dt, dtx;
	double dx, dy, dz;
	double CH0, CH1, CHS, C0, AZ0, FA0;
	double c0_bis, b;
	double uss, vss, wss, I;
	int3 rn_offset[27];
	int3 um_offset_noflux[27];
	int3 um_offset_periodic[27];
	int gelType;
	int maxFilamentlen;
	float lbm_to_gel_vel;  // velocity unit conversion: LBM lattice vel → gel physical vel

	// ── Tube mask (hollow cross-section along X) ───────────────────────────
	// tube_mode: 0=solid (default), 1=square tube, 2=cylinder tube
	// Mask is computed on-the-fly in kernels from element YZ indices.
	// cy, cz : center of tube in element index space (1-based)
	// inner_hy, inner_hz : inner half-size in Y and Z directions (element counts)
	//   for cylinder: inner_hy == inner_hz == inner_radius (elements)
	int   tube_mode;    // 0=solid, 1=square, 2=cylinder
	float tube_cy;     // = (LY+1)/2.0
	float tube_cz;     // = (LZ+1)/2.0
	float tube_inner_hy;  // inner half-height in Y
	float tube_inner_hz;  // inner half-height in Z
};

class Gel {
public:
    Gel(int3 gelSize, double3 gelPosition, int gelType, int gel_id, int time);
    ~Gel();

    void _initialize(int time);
	void setGoonValue(int time);
	void stepChemistry(int iter);
	void stepElasticity(int iter);
	void recordCenter(int iter);
    void _finalize();
	void writeFiles(int iter);

    // Phase-control helpers (used by sim_*.cpp entry points)
    void resetToQuiescent();       // Set all chem fields to quiescent fixed point
    void fireExcitationPulse();    // Inject BZ excitation at X=0 face
    void setAnchorZ(double anchor_z); // Pin bottom-layer (zi==1) nodes in Z

    // Tube/cylinder mask: hollow out the interior along X, keeping a wall of
    // thickness wall_thickness elements in YZ.  circular=false → square tube.
    // Must be called AFTER construction (setInitValue already ran).
    void buildTubeMask(int wall_thickness, bool circular = false);

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
	int* m_hbIndex;
	double m_htime = 0;

	// GPU data
	//paramas
	GelParams* m_dgp;
	//chemical variables
	double* m_dum;
	double* m_dum_norm;
	double* m_dun_norm;
	double* m_dun_robin;
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
	double3* m_dFn_robin;
	double3* m_dFdrag_robin;  // IBM drag force on boundary nodes (calNodesVelocityD/setZero)
	double3* m_drm_loc;
	double3* m_dnmSm;
	double* m_dVolm;
	double* m_dPrem;

	int* m_dmap_node;
	int* m_dmap_element;
	int* m_dbIndex;

public:
	// params
	int m_gelId;
	int m_gelType;
	double3 m_gelPosition;
	int3 m_gelSize;
	double m_dx;
	double m_dy;
	double m_dz;
	int3 m_gelNodeGrid;
	double m_dt;
	int m_df;
	int m_numGelElements;
	int m_numGelNodes;
	int m_boundaryCount;
    cudaStream_t m_gel_stream;
	std::thread m_file_writer_thread;
    dim3 m_blockDim;
	dim3 m_gridDim_1;
	dim3 m_gridDim0;
	dim3 m_gridDim1;
	dim3 m_gridDim2;
	bool flag = false;
	unsigned int* d_hitCnt;
    // Anchor support (set via setAnchorZ; used inside stepElasticity)
    double m_anchor_z    = 0.0;
    bool   m_use_anchor  = false;
};
