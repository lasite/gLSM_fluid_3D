#include "stdio.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include "fluid_kernels.cuh"

__device__ static float3 operator+(float3 a, float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ static int3 operator+(int3 a, int3 b)
{
	return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ static float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ static float3 operator*(float a, int3 b)
{
	return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ static float3 operator*(float a, float3 b)
{
	return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ static float3 operator*(float3 a, float b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ static float operator*(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ static float operator*(int3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ static float3 operator/(float3 a, float b)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ static void operator+=(float3& a, float3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

__device__ static float delta4(float r) {
	r = fabsf(r);
	if (r <= 1.f) {
		return 0.125f * (3.f - 2.f * r + sqrtf(1.f + 4.f * r - 4.f * r * r));
	}
	else if (r <= 2.f) {
		return 0.125f * (5.f - 2.f * r - sqrtf(-7.f + 12.f * r - 4.f * r * r));
	}
	else {
		return 0.f;
	}
}

__device__ static int id3(int x, int y, int z, int Nx, int Ny) {
	return x + y * Nx + z * Nx * Ny;
}

__global__  void k_set_force(float3* F, FluidParams* fp) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < fp->N) {
		F[i] = fp->F_const;
	}
}

__global__  void k_init(float* f, float* rho, float3* u, float* c1, float* c2, FluidParams* fp) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= fp->N)
		return;
	rho[i] = 1.f;
	u[i] = make_float3(0.f, 0.f, 0.f);
	c1[i] = 0.f;
	c2[i] = 0.f;
#pragma unroll
	for (int q = 0; q < 19; ++q)
		f[i + q * (size_t)fp->N] = fp->w[q];
}

__global__  void k_macros(float* f, float* rho, float3* u, float3* F, FluidParams* fp) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= fp->N) return;
	float rh = 0.f;
	float3 s = make_float3(0.f, 0.f, 0.f);
#pragma unroll
	for (int q = 0; q < 19; ++q) {
		float fi = f[i + q * (size_t)fp->N];
		rh += fi;
		s += fi * fp->c[q];
	}
	rho[i] = rh;
	u[i] = (s + 0.5f * F[i]) / rh;
}

__global__  void k_zero(float3* a, FluidParams* fp) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < fp->N)
		a[i] = make_float3(0.f, 0.f, 0.f);
}

__global__  void k_collide(float* f, float* fpost, float* rho, float3* u, float3* F, FluidParams* fp) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= fp->N) return;
	float rh = rho[i];
	float inv_tau = 1.f / fp->tau;
	float uu = u[i] * u[i];
#pragma unroll
	for (int q = 0; q < 19; ++q) {
		int3 c = fp->c[q];
		float cu = c * u[i];
		float feq = fp->w[q] * rh * (1.f + (cu / fp->cs2) + 0.5f * (cu * cu) / (fp->cs2 * fp->cs2) - 0.5f * (uu / fp->cs2));
		float ciF = c * F[i];
		float uF = u[i] * F[i];
		float Fterm = fp->w[q] * (1.f - 0.5f * inv_tau) * ((ciF / fp->cs2) + (cu * ciF) / (fp->cs2 * fp->cs2) - (uF / fp->cs2));
		float fi = f[i + q * (size_t)fp->N];
		fpost[i + q * (size_t)fp->N] = fi - inv_tau * (fi - feq) + Fterm;
	}
}

__global__  void k_stream_bounce(float* fpost, float* fnext, FluidParams* fp) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int N = fp->N; if (i >= N) return;
	const int Nx = fp->L.x, Ny = fp->L.y, Nz = fp->L.z;
	int tmp = i / Nx;
	int3 r = make_int3(i % Nx, tmp % Ny, tmp / Ny);
#pragma unroll
	for (int q = 0; q < 19; ++q) {
		int3 rn = r + fp->c[q];
		bool inside = (0 <= rn.x && rn.x < Nx &&
			0 <= rn.y && rn.y < Ny &&
			0 <= rn.z && rn.z < Nz);
		if (inside) {
			int idn = id3(rn.x, rn.y, rn.z, Nx, Ny);
			fnext[idn + q * (size_t)N] = fpost[i + q * (size_t)N];
		}
		else {
			int qo = fp->opp[q];
			fnext[i + qo * (size_t)N] = fpost[i + q * (size_t)N];
		}
	}
}

__global__  void k_vec_add(float3* A, float3* B, float3* C, FluidParams* fp) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < fp->N)
		C[i] = A[i] + B[i];
}

__device__ __forceinline__ float dot3(const float3 a, const float3 b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__global__ void k_convection_diffusion(float3* d_u, float* d_c1, float* d_c2, FluidParams* fp)
{
	const int Nx = fp->L.x, Ny = fp->L.y, Nz = fp->L.z;
	const int N = fp->N;

	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= N) return;

	const int z = i / (Nx * Ny);
	const int r = i - z * (Nx * Ny);
	const int y = r / Nx;
	const int x = r - y * Nx;

	const int xp = (x + 1 < Nx) ? x + 1 : (fp->perX ? 0 : x);
	const int xm = (x - 1 >= 0) ? x - 1 : (fp->perX ? Nx - 1 : x);
	const int yp = (y + 1 < Ny) ? y + 1 : (fp->perY ? 0 : y);
	const int ym = (y - 1 >= 0) ? y - 1 : (fp->perY ? Ny - 1 : y);
	const int zp = (z + 1 < Nz) ? z + 1 : (fp->perZ ? 0 : z);
	const int zm = (z - 1 >= 0) ? z - 1 : (fp->perZ ? Nz - 1 : z);

	const int idx = id3(x, y, z, Nx, Ny);
	const int idx_xp = id3(xp, y, z, Nx, Ny);
	const int idx_xm = id3(xm, y, z, Nx, Ny);
	const int idx_yp = id3(x, yp, z, Nx, Ny);
	const int idx_ym = id3(x, ym, z, Nx, Ny);
	const int idx_zp = id3(x, y, zp, Nx, Ny);
	const int idx_zm = id3(x, y, zm, Nx, Ny);

	const float3 h = make_float3(fp->dx, fp->dy, fp->dz);
	const float3 inv_h = make_float3(1.0f / h.x, 1.0f / h.y, 1.0f / h.z);
	const float3 inv_h2 = make_float3(inv_h.x * inv_h.x, inv_h.y * inv_h.y, inv_h.z * inv_h.z);

	const float c = d_c1[idx];
	const float cxp = d_c1[idx_xp], cxm = d_c1[idx_xm];
	const float cyp = d_c1[idx_yp], cym = d_c1[idx_ym];
	const float czp = d_c1[idx_zp], czm = d_c1[idx_zm];

	const float uL = 0.5f * (d_u[idx_xm].x + d_u[idx].x);
	const float uR = 0.5f * (d_u[idx].x + d_u[idx_xp].x);
	const float vL = 0.5f * (d_u[idx_ym].y + d_u[idx].y);
	const float vR = 0.5f * (d_u[idx].y + d_u[idx_yp].y);
	const float wL = 0.5f * (d_u[idx_zm].z + d_u[idx].z);
	const float wR = 0.5f * (d_u[idx].z + d_u[idx_zp].z);

	const float3 vL3 = make_float3(uL, vL, wL);
	const float3 vR3 = make_float3(uR, vR, wR);

	const float3 cL3 = make_float3((uL >= 0.0f) ? cxm : c,
		(vL >= 0.0f) ? cym : c,
		(wL >= 0.0f) ? czm : c);
	const float3 cR3 = make_float3((uR >= 0.0f) ? c : cxp,
		(vR >= 0.0f) ? c : cyp,
		(wR >= 0.0f) ? c : czp);

	const float3 FL = make_float3(vL3.x * cL3.x, vL3.y * cL3.y, vL3.z * cL3.z);
	const float3 FR = make_float3(vR3.x * cR3.x, vR3.y * cR3.y, vR3.z * cR3.z);

	const float3 dF = make_float3(FR.x - FL.x, FR.y - FL.y, FR.z - FL.z);
	const float conv = -dot3(dF, inv_h);

	const float3 sec = make_float3(cxm - 2.0f * c + cxp,
		cym - 2.0f * c + cyp,
		czm - 2.0f * c + czp);
	const float diff = fp->D * dot3(sec, inv_h2);

	d_c2[idx] = c + fp->dt * (conv + diff);
}