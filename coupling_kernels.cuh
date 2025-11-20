#ifndef GEL_KERNEL_CUH
#define GEL_KERNEL_CUH
#include "coupling.h"
__global__ void k_add_reaction_to_gel(int* bIndex, double3* Fn, double* un_norm,
    float3* Fl, float* Cl, int M);

__global__ void k_gather_boundary(int* bIndex, double3* rn, double3* vn, double* un_norm,
    float3* lag, float3* Vl, float* Cl, int M);

__global__ void k_gel_repulsion(float3* lag, int* owner, float3* Fl, CouplerParams* cp);
#endif
