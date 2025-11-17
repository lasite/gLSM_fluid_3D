#ifndef GEL_KERNEL_CUH
#define GEL_KERNEL_CUH
#include "fluid.h"
__global__ void k_set_force(float3* F, FluidParams* fp);

__global__  void k_init(float* f, float* rho, float3* u, float* c1, float* c2, FluidParams* fp);

__global__ void k_macros(float* f, float* rho, float3* u, float3* F, FluidParams* fp);

__global__ void k_zero(float3* a, FluidParams* fp);

__global__ void k_collide(float* f, float* fpost, float* rho, float3* u, float3* F, FluidParams* fp);

__global__ void k_stream_bounce(float* fpost, float* fnext, FluidParams* fp);

__global__ void k_vec_add(float3* A, float3* B, float3* C, FluidParams* fp);

__global__ void k_convection_diffusion(float3* d_u, float* d_c1, float* d_c2, FluidParams* fp);

#endif
