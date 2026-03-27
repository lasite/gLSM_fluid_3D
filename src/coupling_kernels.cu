#include "stdio.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include "coupling_kernels.cuh"
using namespace std;

__global__ void k_add_reaction_to_gel(int* bIndex, double3* Fn, double* un_norm, float3* Fl, float* Sl, int M)
{
	int l = blockDim.x * blockIdx.x + threadIdx.x;
	if (l >= M) return;

	int id = bIndex[l];
	float3 f = Fl[l];
	float s = Sl[l];

	Fn[id].x -= (double)f.x;
	Fn[id].y -= (double)f.y;
	Fn[id].z -= (double)f.z;
	un_norm[id] -= (double)s;
}

__global__ void k_add_drag_to_gel(int* bIndex, double3* Fdrag, float3* Fl, int M)
{
	int l = blockDim.x * blockIdx.x + threadIdx.x;
	if (l >= M) return;

	int id = bIndex[l];
	float3 f = Fl[l];

	Fdrag[id].x -= (double)f.x;
	Fdrag[id].y -= (double)f.y;
	Fdrag[id].z -= (double)f.z;
}

__device__ __forceinline__ float3 to_float3(const double3 a) {
	return make_float3(__double2float_rn(a.x),
		__double2float_rn(a.y),
		__double2float_rn(a.z));
}

__global__ void k_gather_boundary(int* bIndex, double3* rn, double3* vn, double* un_norm,
    float3* lag, float3* Vl, float* Cl, int M, float gel_to_lbm_vel)
{
        int l = blockDim.x * blockIdx.x + threadIdx.x;
        if (l >= M) return;

	int id = bIndex[l];

	const double3 r = rn[id];
	const double3 v = vn[id];
	const double u = un_norm[id];

	lag[l] = to_float3(r);
	Vl[l] = make_float3(__double2float_rn(v.x * gel_to_lbm_vel),
		__double2float_rn(v.y * gel_to_lbm_vel),
		__double2float_rn(v.z * gel_to_lbm_vel));
	Cl[l] = __double2float_rn(u);
}

__device__ inline void bodyBodyInteraction(
    float3 posA, float3 posB,
    int ownerA, int ownerB,
    float3& force_acc,
    float delta, float gamma, float rc2, float sig2)
{
    if (ownerA == ownerB) return;

    float3 d;
    d.x = posA.x - posB.x;
    d.y = posA.y - posB.y;
    d.z = posA.z - posB.z;

    float distSqr = d.x * d.x + d.y * d.y + d.z * d.z;

    if (distSqr <= 0.0f || distSqr >= rc2) return;

    float invDistSqr = 1.0f / distSqr;
    float sr2 = sig2 * invDistSqr;
    float sr6 = sr2 * sr2 * sr2;
    float sr12 = sr6 * sr6;

    float fac = 24.0f * delta * (2.0f * sr12 - sr6) * invDistSqr;

    force_acc.x += fac * d.x;
    force_acc.y += fac * d.y;
    force_acc.z += fac * d.z;
}

__global__ void k_gel_repulsion(float3* lag, int* owner, float3* Fl, CouplerParams* cp)
{
    int a = blockDim.x * blockIdx.x + threadIdx.x;

    float3 pos_a;
    int owner_a;
    float3 force = make_float3(0.0f, 0.0f, 0.0f);

    const float delta = (float)cp->delta;
    const float gamma = (float)cp->gamma;
    const float rcut = 2.5f * gamma;
    const float rc2 = rcut * rcut;
    const float sig2 = gamma * gamma;
    const int M = cp->M;

    if (a < M) {
        pos_a = lag[a];
        owner_a = owner[a];
    }

    __shared__ float3 sharedPos[256];
    __shared__ int    sharedOwner[256];

    for (int tile = 0; tile < (M + blockDim.x - 1) / blockDim.x; ++tile) {
        int b_idx = tile * blockDim.x + threadIdx.x;

        if (b_idx < M) {
            sharedPos[threadIdx.x] = lag[b_idx];
            sharedOwner[threadIdx.x] = owner[b_idx];
        }
        else {
            sharedPos[threadIdx.x] = make_float3(0, 0, 0);
            sharedOwner[threadIdx.x] = -1;
        }

        __syncthreads();

        if (a < M) {
#pragma unroll
            for (int k = 0; k < blockDim.x; ++k) {
                if (sharedOwner[k] != -1) {
                    bodyBodyInteraction(pos_a, sharedPos[k],
                        owner_a, sharedOwner[k],
                        force, delta, gamma, rc2, sig2);
                }
            }
        }

        __syncthreads();
    }

    if (a < M) {
        atomicAdd(&(Fl[a].x), force.x);
        atomicAdd(&(Fl[a].y), force.y);
        atomicAdd(&(Fl[a].z), force.z);
    }
}