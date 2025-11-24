#include "stdio.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include "coupling_kernels.cuh"
using namespace std;

__device__ static float3 operator*(float a, float3 b)
{
	return make_float3(a * b.x, a * b.y, a * b.z);
}

__global__ void k_add_reaction_to_gel(int* bIndex, double3* Fn, double* un_norm, float3* Fl, float* Sl, int M)
{
	int l = blockDim.x * blockIdx.x + threadIdx.x;
	if (l >= M) return;

	int id = bIndex[l];
	float3 f = Fl[l];
	float s = Sl[l];

	Fn[id].x = -(double)f.x;
	Fn[id].y = -(double)f.y;
	Fn[id].z = -(double)f.z;
	un_norm[id] = -(double)s;
	//Fn[id].x -= (double)f.x;
	//Fn[id].y -= (double)f.y;
	//Fn[id].z -= (double)f.z;
	//un_norm[id] -= (double)s;
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