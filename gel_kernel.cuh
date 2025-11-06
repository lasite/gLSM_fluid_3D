#ifndef GEL_KERNEL_CUH
#define GEL_KERNEL_CUH

__global__ void calElementPropertiesD(double3* rn, double3* rm, double3* rm_loc, double3* nmSm, double* volume, double* wm, double* wmp);

__global__ void calPressureD(double* pm, double* vm, double* wm);

__global__ void calNodesVelocityD(double3* rn, double3* ven, double3* ves, double3* Fn, double3* nmSm, double* pm, double* wm);

__global__ void calInternalNodesPositionD(double3* rn, double3* ven);

__global__ void calServiceNodesPositionD(double3* rn, int* map_node);

__global__ void calTermsD(double* T0, double* T1, double* T2, double* wm, double* wmp, double3* ven, double3* nmSm, double* volume, double3* rm_loc, double* un_norm, double* um_norm, double3* rm);

__global__ void calChemD(double* vm, double* um, double* wm, double* T0, double* T1, double* T2, double3* rm, int time);

__global__ void calChemBoundaryD(double* um, double* um_norm, double* vm, double* vm_norm, double* wm, int* map_element, int time);

__global__ void calUnnormD(double* un_norm, double* um_norm, double* vn_norm, double* vm_norm);

__global__ void recordCenterElementD(double* vm_center, double* wm_center, double3* rm_center, double3* Fn_center, double3* Veln_center, double* vm, double* wm, double3* rm, double3* Fn, double3* Veln, int time);

__global__ void calFilamentD(double* vn_norm, double* un_norm, double3* filament, int time, unsigned int* hitCnt);

__global__ void k_set_force(float3* F);

__global__ void k_init(float* f, float* rho, float3* u);

__global__ void k_macros(float* f, float* rho, float3* u, float3* F);

__global__ void k_zero(float3* a);

__global__ void k_ibm_interpolate(float3* u, float3* Ul, float3* lag);

__global__ void k_scale_negbeta(float3* Ul, float3* Vl, float3* Fl, float beta_eff);

__global__ void k_add_reaction_to_gel(int* bIndex, double3* Fn, float3* Fl);

__global__ void k_ibm_spread(float3* Fl, float3* lag, float3* F_ibm, float* dA);

__global__ void k_collide(float* f, float* fpost, float* rho, float3* u, float3* F);

__global__ void k_stream_bounce(float* fpost, float* fnext);

__global__ void k_vec_add(float3* A, float3* B, float3* C);

__global__ void k_gather_boundary(int* bIndex, double3* rn, double3* vn, float3* lag, float3* Vl);

#endif
