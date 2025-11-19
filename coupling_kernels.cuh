#ifndef GEL_KERNEL_CUH
#define GEL_KERNEL_CUH
#include "coupling.h"
__global__ void k_ibm_interpolate_velocity(float3* u, float3* Ul, float3* lag, CouplerParams* cp);

__global__ void k_ibm_sample_concentration(float* c, float* Cl, float3* lag, CouplerParams* cp);

__global__ void k_scale_negbeta(float3* Ul, float3* Vl, float3* Fl, float beta_eff, CouplerParams* cp);

__global__ void k_add_reaction_to_gel(int* bIndex, double3* Fn, double* un_norm,
    float3* Fl, float* Cl, int M);

__global__ void k_ibm_spread_forces(float3* F_ibm, float3* Fl, float3* lag, float* dA, CouplerParams* cp);

__global__ void k_ibm_spread_concentration(float* c, float* Dl, float3* lag, float* dA, CouplerParams* cp);

__global__ void k_gather_boundary(int* bIndex, double3* rn, double3* vn, double* un_norm,
    float3* lag, float3* Vl, float* Cl, int M);

__global__ void k_gel_repulsion(float3* lag, int* owner, float3* Fl, CouplerParams* cp);
#endif
