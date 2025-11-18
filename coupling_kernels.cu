#include "stdio.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include "coupling_kernels.cuh"
using namespace std;

__device__ static float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ static float3 operator*(float a, float3 b)
{
	return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ static float3 operator*(float3 a, float b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
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

__global__ void k_ibm_interpolate(float3* u, float* c, float3* Ul, float* Cl, float3* lag, CouplerParams* cp) {
	int l = blockDim.x * blockIdx.x + threadIdx.x;
	if (l >= cp->M) return;

	const float h = cp->h;
	const int Nx = cp->L.x, Ny = cp->L.y, Nz = cp->L.z;

	const float gx = lag[l].x / h;
	const float gy = lag[l].y / h;
	const float gz = lag[l].z / h;

	const int ix = (int)floor(gx);
	const int iy = (int)floor(gy);
	const int iz = (int)floor(gz);

	float3 ul = make_float3(0.0, 0.0, 0.0);
	float cl = 0.0;
	float wsum = 0.0;

	for (int ii = max(0, ix - 1); ii <= min(Nx - 1, ix + 2); ++ii) {
		const float phix = delta4(gx - (float)ii);
		for (int jj = max(0, iy - 1); jj <= min(Ny - 1, iy + 2); ++jj) {
			const float phiy = delta4(gy - (float)jj);
			for (int kk = max(0, iz - 1); kk <= min(Nz - 1, iz + 2); ++kk) {
				const float phiz = delta4(gz - (float)kk);
				const float w = phix * phiy * phiz;
				const int id = id3(ii, jj, kk, Nx, Ny);
				ul += u[id] * w;
				cl += c[id] * w;
				wsum += w;
			}
		}
	}
	Ul[l] = ul / wsum;
	Cl[l] = cl / wsum;
	//if (l == 0) {
	//	printf("%f, %f\n", Ul[l], Cl[l]);
	//}
}

__global__ void k_scale_negbeta(float3* Ul, float3* Vl, float3* Fl, float beta_eff, CouplerParams* cp) {
	int l = blockDim.x * blockIdx.x + threadIdx.x;
	if (l >= cp->M) return;
	//Fl[l] = cp->beta * (Vl[l] - Ul[l]);
	Fl[l] = beta_eff * (Vl[l] - Ul[l]);
	//if (l == 1) {
	//	printf("%12f, %12f\n", Vl[l].x, Vl[l].y);
	//	printf("%12f, %12f\n", Fl[l].x, Fl[l].y);
	//}
}

__global__ void k_add_reaction_to_gel(int* bIndex, double3* Fn, double* un_norm, float3* Fl, float* Cl, int M)
{
	int l = blockDim.x * blockIdx.x + threadIdx.x;
	if (l >= M) return;

	int id = bIndex[l];
	float3 f = Fl[l];
	float c = Cl[l];

	Fn[id].x = -(double)f.x;
	Fn[id].y = -(double)f.y;
	Fn[id].z = -(double)f.z;
	un_norm[id] = (double)c;
	//if (l == 0) {
	//	printf("%f, %f， %f\n", Fl[l].x, Fl[l].y, Fl[l].z);
	//}
}

__global__ void k_ibm_spread(float3* F_ibm, float* c, float3* Fl, float* Dl, float3* lag, float* dA, CouplerParams* cp) {
	int l = blockDim.x * blockIdx.x + threadIdx.x;
	if (l >= cp->M) return;

	const float h = cp->h;
	const int Nx = cp->L.x, Ny = cp->L.y, Nz = cp->L.z;

	const float gx = lag[l].x / h;
	const float gy = lag[l].y / h;
	const float gz = lag[l].z / h;

	const int ix = (int)floor(gx);
	const int iy = (int)floor(gy);
	const int iz = (int)floor(gz);

	const float3 fL = Fl[l];
	const float dL = Dl[l];

	float wsum = 0.0;
	for (int ii = max(0, ix - 1); ii <= min(Nx - 1, ix + 2); ++ii) {
		const float phix = delta4(gx - (float)ii);
		for (int jj = max(0, iy - 1); jj <= min(Ny - 1, iy + 2); ++jj) {
			const float phiy = delta4(gy - (float)jj);
			for (int kk = max(0, iz - 1); kk <= min(Nz - 1, iz + 2); ++kk) {
				const float phiz = delta4(gz - (float)kk);
				wsum += phix * phiy * phiz;
			}
		}
	}
	if (wsum == 0.0) return;

	// δ_h = (1/h^3) φ(·/h)  → 扩散时需要乘 1/h^3
	// scale 还乘 dA[l]，并做近壁面 wsum 归一化，保证 ∑_grid F_ibm * h^3 ≈ ∑_l Fl * dA_l
	const float inv_h3 = 1.0 / (h * h * h);
	const float scale = 1 * inv_h3 / wsum;

	for (int ii = max(0, ix - 1); ii <= min(Nx - 1, ix + 2); ++ii) {
		const float phix = delta4(gx - (float)ii);
		for (int jj = max(0, iy - 1); jj <= min(Ny - 1, iy + 2); ++jj) {
			const float phiy = delta4(gy - (float)jj);
			for (int kk = max(0, iz - 1); kk <= min(Nz - 1, iz + 2); ++kk) {
				const float phiz = delta4(gz - (float)kk);
				const float w = phix * phiy * phiz * scale;
				const int id = id3(ii, jj, kk, Nx, Ny);
				atomicAdd(&F_ibm[id].x, fL.x * w);
				atomicAdd(&F_ibm[id].y, fL.y * w);
				atomicAdd(&F_ibm[id].z, fL.z * w);
				atomicAdd(&c[id], dL * w);
				//if (l == 0) {
				//	printf("%f, %f, %f, %f\n", F_ibm[id].x, F_ibm[id].y, F_ibm[id].z, c[id]);
				//}
			}
		}
	}

}

__device__ __forceinline__ float3 to_float3(const double3 a) {
	return make_float3(__double2float_rn(a.x),
		__double2float_rn(a.y),
		__double2float_rn(a.z));
}

__global__ void k_gather_boundary(int* bIndex, double3* rn, double3* vn, double* un_norm,
    float3* lag, float3* Vl, float* Cl, int M)
{
        int l = blockDim.x * blockIdx.x + threadIdx.x;
        if (l >= M) return;

	int id = bIndex[l];

	const double3 r = rn[id];
	const double3 v = vn[id];
	const double u = un_norm[id];

	lag[l] = to_float3(r);
	Vl[l] = to_float3(v);
	Cl[l] = __double2float_rn(u);
	//if (l == 0) {
	//	printf("%f, %f, %f\n", r.x, r.y, r.z);
	//	printf("%f, %f, %f\n", v.x, v.y, v.z);
	//	printf("%f\n", u);
	//}
}

__device__ __forceinline__ void atomicAdd_float3(float3* p, const float3 v) {
	atomicAdd(&p->x, v.x);
	atomicAdd(&p->y, v.y);
	atomicAdd(&p->z, v.z);
}

__global__ void k_gel_repulsion(float3* lag, int* owner, float3* Fl, CouplerParams* cp)
{
    const int a = blockDim.x * blockIdx.x + threadIdx.x;
    if (a >= cp->M) return;

    const int oa = owner[a];
    const float3 xa = make_float3((double)lag[a].x,
                                    (double)lag[a].y,
                                    (double)lag[a].z);

    // 从 cp 取 δ/γ，并设定截断
    const double epsilon = cp->delta;    // δ = 1e-4
    const double sigma   = cp->gamma;    // γ = 1.5
	const double rcut = 2.5 * sigma;             // 典型截断
	const double rc2 = rcut * rcut;
	const double sig2 = sigma * sigma;

    for (int b = a + 1; b < cp->M; ++b) {
        const int ob = owner[b];
        if (ob == oa) continue; // 同 owner 不相互作用
        const float3 xb = make_float3((double)lag[b].x,
                                        (double)lag[b].y,
                                        (double)lag[b].z);

        const double dx = xa.x - xb.x;
        const double dy = xa.y - xb.y;
        const double dz = xa.z - xb.z;
        const double r2 = dx*dx + dy*dy + dz*dz;
        if (r2 <= 0.0 || r2 >= rc2) continue; // 避免 r=0 & 截断外

        // Lennard–Jones 力：F = 24ε[2(σ/r)^12 - (σ/r)^6] * (r_vec / r^2)
        const double inv_r2 = 1.0 / r2;
        const double sr2    = sig2 * inv_r2;           // (σ/r)^2
        const double sr6    = sr2 * sr2 * sr2;         // (σ/r)^6
        const double sr12   = sr6 * sr6;               // (σ/r)^12
        const double fac    = 24.0 * epsilon * (2.0*sr12 - sr6) * inv_r2;

        float3 F = make_float3(fac*dx, fac*dy, fac*dz);
		atomicAdd_float3(&Fl[a], F);
		atomicAdd_float3(&Fl[b], -1*F);
    }
}