#ifndef GEL_KERNEL_CUH
#define GEL_KERNEL_CUH
#include "coupling.h"
__global__ void k_ibm_interpolate(float3* u, float* c, float3* Ul, float* Cl, float3* lag, CouplerParams* cp);

__global__ void k_scale_negbeta(float3* Ul, float3* Vl, float3* Fl, float beta_eff, CouplerParams* cp);

__global__ void k_add_reaction_to_gel(int* bIndex, double3* Fn, double* un_norm, float3* Fl, float* Cl, CouplerParams* cp);

__global__ void k_ibm_spread(float3* F_ibm, float* c, float3* Fl, float* Dl, float3* lag, float* dA, CouplerParams* cp);

__global__ void k_gather_boundary(int* bIndex, double3* rn, double3* vn, double* un_norm, float3* lag, float3* Vl, float* Cl, CouplerParams* cp);

__global__ void k_gel_repulsion(float3* lag, int* owner, float3* Fl, CouplerParams* cp);
#endif
