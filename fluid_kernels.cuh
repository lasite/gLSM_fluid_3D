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

__global__ void k_ibm_interpolate_velocity(float3* u, float3* Ul, float3* lag, FluidParams* fp);

__global__ void k_ibm_sample_concentration(float* c, float* Cl, float3* lag, FluidParams* fp);

__global__ void k_scale_negbeta(float3* Ul, float3* Vl, float3* Fl, float beta_eff, FluidParams* fp);

__global__ void k_ibm_spread_forces(float3* F_ibm, float3* Fl, float3* lag, float* dA, FluidParams* fp);

__global__ void k_ibm_spread_concentration(float* c, float* Dl, float3* lag, float* dA, FluidParams* fp);
#endif