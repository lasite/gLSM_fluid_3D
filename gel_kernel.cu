#include "stdio.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#include "gelParams.h"

__constant__ GelParams params;
__constant__ FluidParams p;

__device__ static double3 operator+(double3 a, double3 b)
{
	return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ static float3 operator+(float3 a, float3 b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ static int3 operator+(int3 a, int3 b)
{
	return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ static double3 operator-(double3 a, double3 b)
{
	return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ static float3 operator-(float3 a, float3 b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ static double3 operator-(double3 a)
{
	return make_double3(-a.x, -a.y, -a.z);
}

__device__ static double3 operator*(double a, double3 b)
{
	return make_double3(a * b.x, a * b.y, a * b.z);
}

__device__ static double3 operator*(double a, int3 b)
{
	return make_double3(a * b.x, a * b.y, a * b.z);
}

__device__ static float3 operator*(float a, int3 b)
{
	return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ static float3 operator*(float a, float3 b)
{
	return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ static double3 operator*(double3 a, double b)
{
	return make_double3(a.x * b, a.y * b, a.z * b);
}

__device__ static float3 operator*(float3 a, float b)
{
	return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ static double operator*(double3 a, double3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ static float operator*(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ static double operator*(int3 a, double3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ static float operator*(int3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ static double3 operator/(double3 a, double3 b)
{
	return make_double3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ static double3 operator/(double3 a, double b)
{
	return make_double3(a.x / b, a.y / b, a.z / b);
}

__device__ static float3 operator/(float3 a, float b)
{
	return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ static double3 operator^(double3 a, double3 b)
{
	return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ static void operator+=(double3& a, double3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

__device__ static void operator+=(float3& a, float3 b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
}

__device__ static void operator-=(double3& a, double3 b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
}

__device__ static void operator*=(double3& a, double b)
{
	a.x *= b;
	a.y *= b;
	a.z *= b;
}

__device__ static void operator/=(double3& a, double b)
{
	a.x /= b;
	a.y /= b;
	a.z /= b;
}

__device__ static double3 cross(double3 a, double3 b)
{
	return make_double3(a.y * b.z - b.y * a.z, b.x * a.z - a.x * b.z, a.x * b.y - b.x * a.y);
}

__device__ static double distance(double3 a, double3 b)
{
	return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
}

__device__ static double3 pow(double3 a, int b)
{
	return make_double3(pow(a.x, b), pow(a.y, b), pow(a.z, b));
}

__device__ static int get_index(int xi, int yi, int zi, int size)
{
	return xi + yi * (params.LX + size) + zi * (params.LY + size) * (params.LX + size);
}

__device__ static int get_index_rm_loc(int xi, int yi, int zi, int num)
{
	return num + xi * 8 + yi * (params.LX + 1) * 8 + zi * (params.LY + 1) * (params.LX + 1) * 8;
}

__device__ static int get_index_nmSm(int xi, int yi, int zi, int num)
{
	return num + xi * 6 + yi * (params.LX + 1) * 6 + zi * (params.LY + 1) * (params.LX + 1) * 6;
}

__device__ static double fv(double u, double v, double w, double I = 0)
{
	return (1.0 - w) * (1.0 - w) * u - (1.0 - w) * v + I * (0.5 * params.P1 + params.P2);
	//return (1.0 - w) * (1.0 - w) * u - (1.0 - w) * v;
}

__device__ static double fu(double u, double v, double w, double I = 0)
{
	double ww = (1.0 - w) * (1.0 - w);
	return ww * u - u * u - (1.0 - w) * (params.f * v + I * params.P1) * (u - params.q * ww) / (u + params.q * ww) + I * params.P2;
	//return ww * u - u * u - (1.0 - w) * (params.f * v + I) * (u - params.q * ww) / (u + params.q * ww);
}

__device__ static double3 cal_face_properties(double3 node1, double3 node2, double3 node3, double3 node4, double& vol)
{
	double3 cp = make_double3(0, 0, 0);
	double3 facecenter = 0.25 * (node1 + node2 + node3 + node4);
	cp += cross(node1 - facecenter, node2 - facecenter);
	cp += cross(node2 - facecenter, node3 - facecenter);
	cp += cross(node3 - facecenter, node4 - facecenter);
	cp += cross(node4 - facecenter, node1 - facecenter);
	cp *= 0.5;
	vol += cp * facecenter / 3;
	return cp;
}

__device__ __host__ inline double clamp_d(double v, double lo, double hi)
{
	return (v < lo) ? lo : (v > hi ? hi : v);
}

__device__ static bool solveBilinear(double A, double B, double C, double D, double E, double F, double G, double H, double vIso, double& xb, double& yb, double eps = 1e-12)
{
	// ---------- 系数 ----------
	const double a00 = A - vIso;
	const double a10 = B - A;
	const double a01 = C - A;
	const double a11 = A - B + D - C;

	const double b00 = E - vIso;
	const double b10 = F - E;
	const double b01 = G - E;
	const double b11 = E - F + H - G;

	const double c2 = b10 * a11 - b11 * a10;
	const double c1 = b00 * a11 + b10 * a01 - b01 * a10 - b11 * a00;
	const double c0 = b00 * a01 - b01 * a00;

	double roots[2];
	int n = 0;

	// ---------- 解二次 / 一次方程 ----------
	if (fabs(c2) < eps) {               // 降阶
		if (fabs(c1) < eps) return false;
		roots[n++] = -c0 / c1;
	}
	else {
		double disc = c1 * c1 - 4.0 * c2 * c0;
		if (disc < 0.0) return false;    // 无实根
		double s = sqrt(disc);
		roots[n++] = (-c1 + s) / (2.0 * c2);
		roots[n++] = (-c1 - s) / (2.0 * c2);
	}

	// ---------- 回代求 y，并筛选合法根 ----------
	for (int i = 0; i < n; ++i) {
		double x = roots[i];
		if (x < -eps || x > 1.0 + eps) continue;

		double denom = a01 + a11 * x;
		if (fabs(denom) < eps) continue;

		double y = -(a00 + a10 * x) / denom;
		if (y < -eps || y > 1.0 + eps) continue;

		xb = clamp_d(x, 0.0, 1.0);
		yb = clamp_d(y, 0.0, 1.0);
		return true;
	}
	return false;                        // 没有交点落在单元内
}

__global__ static void calElementPropertiesD(double3* rn, double3* rm, double3* rm_loc, double3* nmSm, double* volume, double* wm, double* wmp)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x;
	int yi = threadIdx.y + blockIdx.y * blockDim.y;
	int zi = threadIdx.z + blockIdx.z * blockDim.z;
	if (xi > params.LX || yi > params.LY || zi > params.LZ) {
		return;
	}

	int gi = get_index(xi, yi, zi, 1);
	int gi_rm_loc = get_index_rm_loc(xi, yi, zi, 0);
	int gi_nmSm = get_index_nmSm(xi, yi, zi, 0);
	double3 node[8];
	double3 bodycenter = make_double3(0, 0, 0);
	//
	int count = 0;
#pragma unroll
	for (int i = 0; i < 2; i++) {
#pragma unroll
		for (int j = 0; j < 2; j++) {
#pragma unroll
			for (int k = 0; k < 2; k++) {
				node[count] = rn[get_index(xi + i, yi + j, zi + k, 2)];
				bodycenter += node[count];
				count++;
			}
		}
	}
	bodycenter /= 8;
	rm[gi] = bodycenter;
#pragma unroll
	for (int i = 0; i < 8; i++) {
		node[i] -= bodycenter;
		rm_loc[gi_rm_loc + i] = node[i];
	}
	//
	if (true)
	{
		double point = 2.0;
		double k_s = 4 / pow(point, 2);
		double k_v = 8 / pow(point, 3);
		double s, n, t;
		double ksiini = -1 + 1 / point;
		double step = 2 / point;
		double ksiend = 1 - 1 / point;
		// 
		double3 coordinate_st[4];
		double3 detJ;
		double3 J_s[2];
		double dNsnt_st[4][2];
		//*****************************************************face1
		coordinate_st[0] = node[0];
		coordinate_st[1] = node[1];
		coordinate_st[2] = node[3];
		coordinate_st[3] = node[2];
		detJ = make_double3(0, 0, 0);
#pragma unroll
		for (n = ksiini; n <= ksiend; n += step) {
#pragma unroll
			for (t = ksiini; t <= ksiend; t += step) {
				s = -1;
				dNsnt_st[0][0] = -(1 - s) * (1 - t) / 8; dNsnt_st[0][1] = -(1 - s) * (1 - n) / 8;
				dNsnt_st[1][0] = -(1 - s) * (1 + t) / 8; dNsnt_st[1][1] = (1 - s) * (1 - n) / 8;
				dNsnt_st[2][0] = (1 - s) * (1 + t) / 8; dNsnt_st[2][1] = (1 - s) * (1 + n) / 8;
				dNsnt_st[3][0] = (1 - s) * (1 - t) / 8; dNsnt_st[3][1] = -(1 - s) * (1 + n) / 8;
				J_s[0] = make_double3(0, 0, 0);
				J_s[1] = make_double3(0, 0, 0);
#pragma unroll
				for (int i = 0; i < 4; i++) {
					J_s[0] += coordinate_st[i] * dNsnt_st[i][0];
					J_s[1] += coordinate_st[i] * dNsnt_st[i][1];
				}
				detJ += cross(J_s[0], J_s[1]);
			}
		}
		detJ *= k_s;
		nmSm[gi_nmSm + 0] = -detJ;
		//printf("%d, %d, %d, %f, %f, %f\n", xi, yi, zi, -detJ.x, -detJ.y, -detJ.z);
		//*****************************************************face2
		coordinate_st[0] = node[4];
		coordinate_st[1] = node[6];
		coordinate_st[2] = node[7];
		coordinate_st[3] = node[5];
		detJ = make_double3(0, 0, 0);
#pragma unroll
		for (n = ksiini; n <= ksiend; n += step) {
#pragma unroll
			for (t = ksiini; t <= ksiend; t += step) {
				s = 1;
				dNsnt_st[0][0] = -(1 + s) * (1 - t) / 8; dNsnt_st[0][1] = -(1 + s) * (1 - n) / 8;
				dNsnt_st[1][0] = (1 + s) * (1 - t) / 8; dNsnt_st[1][1] = -(1 + s) * (1 + n) / 8;
				dNsnt_st[2][0] = (1 + s) * (1 + t) / 8; dNsnt_st[2][1] = (1 + s) * (1 + n) / 8;
				dNsnt_st[3][0] = -(1 + s) * (1 + t) / 8; dNsnt_st[3][1] = (1 + s) * (1 - n) / 8;
				J_s[0] = make_double3(0, 0, 0);
				J_s[1] = make_double3(0, 0, 0);
#pragma unroll
				for (int i = 0; i < 4; i++) {
					J_s[0] += coordinate_st[i] * dNsnt_st[i][0];
					J_s[1] += coordinate_st[i] * dNsnt_st[i][1];
				}
				detJ += cross(J_s[0], J_s[1]);
			}
		}
		detJ *= k_s;
		nmSm[gi_nmSm + 1] = detJ;
		//printf("%d, %d, %d, %f, %f, %f\n", xi, yi, zi, detJ.x, detJ.y, detJ.z);
		//*****************************************************face3
		coordinate_st[0] = node[0];
		coordinate_st[1] = node[4];
		coordinate_st[2] = node[5];
		coordinate_st[3] = node[1];
		detJ = make_double3(0, 0, 0);
#pragma unroll
		for (s = ksiini; s <= ksiend; s += step) {
#pragma unroll
			for (t = ksiini; t <= ksiend; t += step) {
				n = -1;
				dNsnt_st[0][0] = -(1 - n) * (1 - t) / 8; dNsnt_st[0][1] = -(1 - s) * (1 - n) / 8;
				dNsnt_st[1][0] = (1 - n) * (1 - t) / 8; dNsnt_st[1][1] = -(1 + s) * (1 - n) / 8;
				dNsnt_st[2][0] = (1 - n) * (1 + t) / 8; dNsnt_st[2][1] = (1 + s) * (1 - n) / 8;
				dNsnt_st[3][0] = -(1 - n) * (1 + t) / 8; dNsnt_st[3][1] = (1 - s) * (1 - n) / 8;
				J_s[0] = make_double3(0, 0, 0);
				J_s[1] = make_double3(0, 0, 0);
#pragma unroll
				for (int i = 0; i < 4; i++) {
					J_s[0] += coordinate_st[i] * dNsnt_st[i][0];
					J_s[1] += coordinate_st[i] * dNsnt_st[i][1];
				}
				detJ += cross(J_s[0], J_s[1]);
			}
		}
		detJ *= k_s;
		nmSm[gi_nmSm + 2] = detJ;
		//printf("%d, %d, %d, %f, %f, %f\n", xi, yi, zi, detJ.x, detJ.y, detJ.z);
		//*****************************************************face4
		coordinate_st[0] = node[2];
		coordinate_st[1] = node[3];
		coordinate_st[2] = node[7];
		coordinate_st[3] = node[6];
		detJ = make_double3(0, 0, 0);
#pragma unroll
		for (s = ksiini; s <= ksiend; s += step) {
#pragma unroll
			for (t = ksiini; t <= ksiend; t += step) {
				n = 1;
				dNsnt_st[0][0] = -(1 + n) * (1 - t) / 8; dNsnt_st[0][1] = -(1 - s) * (1 + n) / 8;
				dNsnt_st[1][0] = -(1 + n) * (1 + t) / 8; dNsnt_st[1][1] = (1 - s) * (1 + n) / 8;
				dNsnt_st[2][0] = (1 + n) * (1 + t) / 8; dNsnt_st[2][1] = (1 + s) * (1 + n) / 8;
				dNsnt_st[3][0] = (1 + n) * (1 - t) / 8; dNsnt_st[3][1] = -(1 + s) * (1 + n) / 8;
				J_s[0] = make_double3(0, 0, 0);
				J_s[1] = make_double3(0, 0, 0);
#pragma unroll
				for (int i = 0; i < 4; i++) {
					J_s[0] += coordinate_st[i] * dNsnt_st[i][0];
					J_s[1] += coordinate_st[i] * dNsnt_st[i][1];
				}
				detJ += cross(J_s[0], J_s[1]);
			}
		}
		detJ *= k_s;
		nmSm[gi_nmSm + 3] = -detJ;
		//printf("%d, %d, %d, %f, %f, %f\n", xi, yi, zi, -detJ.x, -detJ.y, -detJ.z);
		//*****************************************************face5
		coordinate_st[0] = node[0];
		coordinate_st[1] = node[2];
		coordinate_st[2] = node[6];
		coordinate_st[3] = node[4];
		detJ = make_double3(0, 0, 0);
#pragma unroll
		for (s = ksiini; s <= ksiend; s += step) {
#pragma unroll
			for (n = ksiini; n <= ksiend; n += step) {
				t = -1;
				dNsnt_st[0][0] = -(1 - n) * (1 - t) / 8; dNsnt_st[0][1] = -(1 - s) * (1 - t) / 8;
				dNsnt_st[1][0] = -(1 + n) * (1 - t) / 8; dNsnt_st[1][1] = (1 - s) * (1 - t) / 8;
				dNsnt_st[2][0] = (1 + n) * (1 - t) / 8; dNsnt_st[2][1] = (1 + s) * (1 - t) / 8;
				dNsnt_st[3][0] = (1 - n) * (1 - t) / 8; dNsnt_st[3][1] = -(1 + s) * (1 - t) / 8;
				J_s[0] = make_double3(0, 0, 0);
				J_s[1] = make_double3(0, 0, 0);
#pragma unroll
				for (int i = 0; i < 4; i++) {
					J_s[0] += coordinate_st[i] * dNsnt_st[i][0];
					J_s[1] += coordinate_st[i] * dNsnt_st[i][1];
				}
				detJ += cross(J_s[0], J_s[1]);
			}
		}
		detJ *= k_s;
		nmSm[gi_nmSm + 4] = -detJ;
		//printf("%d, %d, %d, %f, %f, %f\n", xi, yi, zi, -detJ.x, -detJ.y, -detJ.z);
		//*****************************************************face6
		coordinate_st[0] = node[1];
		coordinate_st[1] = node[5];
		coordinate_st[2] = node[7];
		coordinate_st[3] = node[3];
		detJ = make_double3(0, 0, 0);
#pragma unroll
		for (s = ksiini; s <= ksiend; s += step) {
#pragma unroll
			for (n = ksiini; n <= ksiend; n += step) {
				t = 1;
				dNsnt_st[0][0] = -(1 - n) * (1 + t) / 8; dNsnt_st[0][1] = -(1 - s) * (1 + t) / 8;
				dNsnt_st[1][0] = (1 - n) * (1 + t) / 8; dNsnt_st[1][1] = -(1 + s) * (1 + t) / 8;
				dNsnt_st[2][0] = (1 + n) * (1 + t) / 8; dNsnt_st[2][1] = (1 + s) * (1 + t) / 8;
				dNsnt_st[3][0] = -(1 + n) * (1 + t) / 8; dNsnt_st[3][1] = (1 - s) * (1 + t) / 8;
				J_s[0] = make_double3(0, 0, 0);
				J_s[1] = make_double3(0, 0, 0);
#pragma unroll
				for (int i = 0; i < 4; i++) {
					J_s[0] += coordinate_st[i] * dNsnt_st[i][0];
					J_s[1] += coordinate_st[i] * dNsnt_st[i][1];
				}
				detJ += cross(J_s[0], J_s[1]);
			}
		}
		detJ *= k_s;
		nmSm[gi_nmSm + 5] = detJ;
		//printf("%d, %d, %d, %f, %f, %f\n", xi, yi, zi, detJ.x, detJ.y, detJ.z);
		//*****************************************************volume
		double vol = 0;
		double detJ_v = 0;
		double dNsnt_vt[8][3];
		double J_v[3][3];
#pragma unroll
		for (s = ksiini; s <= ksiend; s += step) {
#pragma unroll
			for (n = ksiini; n <= ksiend; n += step) {
#pragma unroll
				for (t = ksiini; t <= ksiend; t += step) {
					dNsnt_vt[0][0] = -(1 - n) * (1 - t) / 8; dNsnt_vt[0][1] = -(1 - s) * (1 - t) / 8; dNsnt_vt[0][2] = -(1 - s) * (1 - n) / 8;
					dNsnt_vt[1][0] = -(1 - n) * (1 + t) / 8; dNsnt_vt[1][1] = -(1 - s) * (1 + t) / 8; dNsnt_vt[1][2] = (1 - s) * (1 - n) / 8;
					dNsnt_vt[2][0] = -(1 + n) * (1 - t) / 8; dNsnt_vt[2][1] = (1 - s) * (1 - t) / 8; dNsnt_vt[2][2] = -(1 - s) * (1 + n) / 8;
					dNsnt_vt[3][0] = -(1 + n) * (1 + t) / 8; dNsnt_vt[3][1] = (1 - s) * (1 + t) / 8; dNsnt_vt[3][2] = (1 - s) * (1 + n) / 8;
					dNsnt_vt[4][0] = (1 - n) * (1 - t) / 8; dNsnt_vt[4][1] = -(1 + s) * (1 - t) / 8; dNsnt_vt[4][2] = -(1 + s) * (1 - n) / 8;
					dNsnt_vt[5][0] = (1 - n) * (1 + t) / 8; dNsnt_vt[5][1] = -(1 + s) * (1 + t) / 8; dNsnt_vt[5][2] = (1 + s) * (1 - n) / 8;
					dNsnt_vt[6][0] = (1 + n) * (1 - t) / 8; dNsnt_vt[6][1] = (1 + s) * (1 - t) / 8; dNsnt_vt[6][2] = -(1 + s) * (1 + n) / 8;
					dNsnt_vt[7][0] = (1 + n) * (1 + t) / 8; dNsnt_vt[7][1] = (1 + s) * (1 + t) / 8; dNsnt_vt[7][2] = (1 + s) * (1 + n) / 8;
#pragma unroll
					for (int i = 0; i < 3; i++) {
#pragma unroll
						for (int j = 0; j < 3; j++) {
							J_v[i][j] = 0;
						}
					}
#pragma unroll
					for (int i = 0; i < 8; i++) {
						J_v[0][0] += dNsnt_vt[i][0] * node[i].x;
						J_v[0][1] += dNsnt_vt[i][0] * node[i].y;
						J_v[0][2] += dNsnt_vt[i][0] * node[i].z;
						J_v[1][0] += dNsnt_vt[i][1] * node[i].x;
						J_v[1][1] += dNsnt_vt[i][1] * node[i].y;
						J_v[1][2] += dNsnt_vt[i][1] * node[i].z;
						J_v[2][0] += dNsnt_vt[i][2] * node[i].x;
						J_v[2][1] += dNsnt_vt[i][2] * node[i].y;
						J_v[2][2] += dNsnt_vt[i][2] * node[i].z;
					}
					detJ_v += (J_v[0][0] * (J_v[1][1] * J_v[2][2] - J_v[2][1] * J_v[1][2]) - J_v[0][1] * (J_v[1][0] * J_v[2][2] - J_v[2][0] * J_v[1][2]) + J_v[0][2] * (J_v[1][0] * J_v[2][1] - J_v[2][0] * J_v[1][1]));
				}
			}
		}
		vol = detJ_v * k_v;
		volume[gi] = vol;
		wmp[gi] = wm[gi];
		wm[gi] = params.dx * params.dy * params.dz * params.FA0 / vol;
	}
	else {
		double vol = 0;
		int map[6][4] = { {0, 1, 3, 2}, {4, 6, 7, 5}, {0, 4, 5, 1}, {2, 3, 7, 6}, {0, 2, 6, 4}, {1, 5, 7, 3} };
#pragma unroll
		for (int i = 0; i < 6; i++)
			nmSm[gi_nmSm + i] = cal_face_properties(node[map[i][0]], node[map[i][1]], node[map[i][2]], node[map[i][3]], vol);
		volume[gi] = vol;
		wmp[gi] = wm[gi];
		wm[gi] = params.dx * params.dy * params.dz * params.FA0 / vol;
	}
}

__global__ static void calPressureD(double* pm, double* vm, double* wm)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int yi = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int zi = threadIdx.z + blockIdx.z * blockDim.z + 1;
	if (xi > params.LX - 1 || yi > params.LY - 1 || zi > params.LZ - 1) {
		return;
	}
	int gi = get_index(xi, yi, zi, 1);
	double wmt = wm[gi];
	pm[gi] = -(wmt + log(1.0 - wmt) + (params.CH0 + params.CH1 * wmt) * wmt * wmt) + params.C0 * wmt / (2.0 * params.FA0) + params.CHS * wmt * vm[gi];
}

__global__ static void calNodesVelocityD(double3* rn, double3* ven, double3* ves, double3* Fn, double3* nmSm, double* pm, double* wm)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int yi = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int zi = threadIdx.z + blockIdx.z * blockDim.z + 1;
	int LX = params.LX;
	int LY = params.LY;
	int LZ = params.LZ;
	if (xi > LX || yi > LY || zi > LZ) {
		return;
	}
	int gi = get_index(xi, yi, zi, 2);
	//calculate mobility
	double wn = 0;
#pragma unroll
	for (int i = 0; i < 2; i++)
#pragma unroll
		for (int j = 0; j < 2; j++)
#pragma unroll
			for (int k = 0; k < 2; k++)
				wn += wm[get_index(xi - i, yi - j, zi - k, 1)];
	wn /= 8;
	double Mn = 8. * params.AZ0 * sqrt(params.FA0 / wn) * (1 - wn) / (params.dx * params.dy * params.dz);
	//calculate force
	double3 f1n = make_double3(0., 0., 0.);
	double3 f2n = make_double3(0., 0., 0.);
	double3 rn_m = rn[gi];
	int gi_nmSm;
	if (xi < LX && yi < LY && zi < LZ) {
		gi_nmSm = get_index_nmSm(xi, yi, zi, 0);
		f1n += rn[get_index(xi + 1, yi + 1, zi + 1, 2)] + rn[get_index(xi, yi + 1, zi + 1, 2)] + rn[get_index(xi + 1, yi, zi + 1, 2)] + rn[get_index(xi + 1, yi + 1, zi, 2)] - 4 * rn_m;
		f2n += pm[get_index(xi, yi, zi, 1)] * (nmSm[gi_nmSm] + nmSm[gi_nmSm + 2] + nmSm[gi_nmSm + 4]);
	}
	if (xi < LX && yi < LY && zi > 1) {
		gi_nmSm = get_index_nmSm(xi, yi, zi - 1, 0);
		f1n += rn[get_index(xi + 1, yi + 1, zi - 1, 2)] + rn[get_index(xi, yi + 1, zi - 1, 2)] + rn[get_index(xi + 1, yi, zi - 1, 2)] + rn[get_index(xi + 1, yi + 1, zi, 2)] - 4 * rn_m;
		f2n += pm[get_index(xi, yi, zi - 1, 1)] * (nmSm[gi_nmSm] + nmSm[gi_nmSm + 2] + nmSm[gi_nmSm + 5]);
	}
	if (xi < LX && yi > 1 && zi < LZ) {
		gi_nmSm = get_index_nmSm(xi, yi - 1, zi, 0);
		f1n += rn[get_index(xi + 1, yi - 1, zi + 1, 2)] + rn[get_index(xi, yi - 1, zi + 1, 2)] + rn[get_index(xi + 1, yi, zi + 1, 2)] + rn[get_index(xi + 1, yi - 1, zi, 2)] - 4 * rn_m;
		f2n += pm[get_index(xi, yi - 1, zi, 1)] * (nmSm[gi_nmSm] + nmSm[gi_nmSm + 3] + nmSm[gi_nmSm + 4]);
	}
	if (xi < LX && yi > 1 && zi > 1) {
		gi_nmSm = get_index_nmSm(xi, yi - 1, zi - 1, 0);
		f1n += rn[get_index(xi + 1, yi - 1, zi - 1, 2)] + rn[get_index(xi, yi - 1, zi - 1, 2)] + rn[get_index(xi + 1, yi, zi - 1, 2)] + rn[get_index(xi + 1, yi - 1, zi, 2)] - 4 * rn_m;
		f2n += pm[get_index(xi, yi - 1, zi - 1, 1)] * (nmSm[gi_nmSm] + nmSm[gi_nmSm + 3] + nmSm[gi_nmSm + 5]);
	}
	if (xi > 1 && yi < LY && zi < LZ) {
		gi_nmSm = get_index_nmSm(xi - 1, yi, zi, 0);
		f1n += rn[get_index(xi - 1, yi + 1, zi + 1, 2)] + rn[get_index(xi, yi + 1, zi + 1, 2)] + rn[get_index(xi - 1, yi, zi + 1, 2)] + rn[get_index(xi - 1, yi + 1, zi, 2)] - 4 * rn_m;
		f2n += pm[get_index(xi - 1, yi, zi, 1)] * (nmSm[gi_nmSm + 1] + nmSm[gi_nmSm + 2] + nmSm[gi_nmSm + 4]);
	}
	if (xi > 1 && yi < LY && zi > 1) {
		gi_nmSm = get_index_nmSm(xi - 1, yi, zi - 1, 0);
		f1n += rn[get_index(xi - 1, yi + 1, zi - 1, 2)] + rn[get_index(xi, yi + 1, zi - 1, 2)] + rn[get_index(xi - 1, yi, zi - 1, 2)] + rn[get_index(xi - 1, yi + 1, zi, 2)] - 4 * rn_m;
		f2n += pm[get_index(xi - 1, yi, zi - 1, 1)] * (nmSm[gi_nmSm + 1] + nmSm[gi_nmSm + 2] + nmSm[gi_nmSm + 5]);
	}
	if (xi > 1 && yi > 1 && zi < LZ) {
		gi_nmSm = get_index_nmSm(xi - 1, yi - 1, zi, 0);
		f1n += rn[get_index(xi - 1, yi - 1, zi + 1, 2)] + rn[get_index(xi, yi - 1, zi + 1, 2)] + rn[get_index(xi - 1, yi, zi + 1, 2)] + rn[get_index(xi - 1, yi - 1, zi, 2)] - 4 * rn_m;
		f2n += pm[get_index(xi - 1, yi - 1, zi, 1)] * (nmSm[gi_nmSm + 1] + nmSm[gi_nmSm + 3] + nmSm[gi_nmSm + 4]);
	}
	if (xi > 1 && yi > 1 && zi > 1) {
		gi_nmSm = get_index_nmSm(xi - 1, yi - 1, zi - 1, 0);
		f1n += rn[get_index(xi - 1, yi - 1, zi - 1, 2)] + rn[get_index(xi, yi - 1, zi - 1, 2)] + rn[get_index(xi - 1, yi, zi - 1, 2)] + rn[get_index(xi - 1, yi - 1, zi, 2)] - 4 * rn_m;
		f2n += pm[get_index(xi - 1, yi - 1, zi - 1, 1)] * (nmSm[gi_nmSm + 1] + nmSm[gi_nmSm + 3] + nmSm[gi_nmSm + 5]);
	}

	f1n *= pow(params.dx * params.dy * params.dz, 1 / 3) * params.C0 / 12.;
	f2n /= 4.;
	//if (xi == 1 && yi == 1 && zi == 1) {
	//	printf("%f, %f, %f\n", Fn[gi].x, Fn[gi].y, Fn[gi].z);
	//}
	Fn[gi] += f1n + f2n;
	ven[gi] = Mn * Fn[gi];
	ves[gi] = -wn * ven[gi] / (1 - wn);
	//if (xi == 1 && yi == 1 && zi == 1) {
	//	printf("%f, %f, %f\n", f1n.x + f2n.x, f1n.y + f2n.y, f1n.z + f2n.z);
	//}
	Fn[gi] = make_double3(0, 0, 0);
}

__global__ static void calInternalNodesPositionD(double3* rn, double3* ven)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int yi = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int zi = threadIdx.z + blockIdx.z * blockDim.z + 1;
	if (xi > params.LX || yi > params.LY || zi > params.LZ) {
		return;
	}
	int gi = get_index(xi, yi, zi, 2);
	rn[gi] += params.dtx * ven[gi];
}

__global__ static void calServiceNodesPositionD(double3* rn, int* map_node)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x;
	int yi = threadIdx.y + blockIdx.y * blockDim.y;
	int zi = threadIdx.z + blockIdx.z * blockDim.z;
	int gi = get_index(xi, yi, zi, 2);
	int node_type = map_node[gi];
	if (node_type == 0 || xi > params.LX + 1 || yi > params.LY + 1 || zi > params.LZ + 1) {
		return;
	}
	int gi1 = get_index(xi + params.rn_offset[node_type].x, yi + params.rn_offset[node_type].y, zi + params.rn_offset[node_type].z, 2);
	int gi2 = get_index(xi + 2 * params.rn_offset[node_type].x, yi + 2 * params.rn_offset[node_type].y, zi + 2 * params.rn_offset[node_type].z, 2);
	rn[gi] = 2 * rn[gi1] - rn[gi2];
}

__global__ static void calTermsD(double* T0, double* T1, double* T2, double* wm, double* wmp, double3* ven, double3* nmSm, double* volume, double3* rm_loc, double* un_norm, double* um_norm, double3* rm)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int yi = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int zi = threadIdx.z + blockIdx.z * blockDim.z + 1;
	if (xi > params.LX - 1 || yi > params.LY - 1 || zi > params.LZ - 1) {
		return;
	}
	int gi = get_index(xi, yi, zi, 1);

	double wm_m = wm[gi];
	double um_m = um_norm[gi];
	double3 rm_m = rm[gi];
	//cal T0

	T0[gi] = (1 - wm_m / wmp[gi]) / params.dtx;

	//cal T1
	double dT1 = 0;
	double3 vmum = make_double3(0, 0, 0);
	int count = 0;
	int gi_rm_loc = get_index_rm_loc(xi, yi, zi, 0);
#pragma unroll
	for (int i = 0; i < 2; i++) {
#pragma unroll
		for (int j = 0; i < 2; i++) {
#pragma unroll
			for (int k = 0; i < 2; i++) {
				int id1 = gi_rm_loc + count;
				count++;
				int id2 = get_index(xi + i, yi + j, zi + k, 2);
				double N = 0.125 * (1 + (2 * double(i) - 1) * rm_loc[id1].x) * (1 + (2 * double(j) - 1) * rm_loc[id1].y) * (1 + (2 * double(k) - 1) * rm_loc[id1].z);
				vmum += N * ven[id2] * un_norm[id2];
			}
		}
	}
	int gi_nmSm = get_index_nmSm(xi, yi, zi, 0);
#pragma unroll
	for (int i = 0; i < 6; i++) {
		dT1 += nmSm[gi_nmSm + i] * vmum;
	}
	T1[gi] = dT1 / volume[gi];

	//cal T2
	int gi1 = get_index(xi + 1, yi, zi, 1);
	int gi2 = get_index(xi - 1, yi, zi, 1);
	int gi3 = get_index(xi, yi + 1, zi, 1);
	int gi4 = get_index(xi, yi - 1, zi, 1);
	int gi5 = get_index(xi, yi, zi + 1, 1);
	int gi6 = get_index(xi, yi, zi - 1, 1);

	double3 a_add = make_double3(distance(rm[gi1], rm_m), distance(rm[gi3], rm_m), distance(rm[gi5], rm_m));
	double3 a_sub = make_double3(distance(rm[gi2], rm_m), distance(rm[gi4], rm_m), distance(rm[gi6], rm_m));

	double3 a_add2 = pow(a_add, 2);
	double3 a_sub2 = pow(a_sub, 2);
	double3 a_addsub = a_add ^ a_sub ^ (a_add + a_sub);

	double3 wm_add = make_double3(wm[gi1], wm[gi3], wm[gi5]);
	double3 wm_sub = make_double3(wm[gi2], wm[gi4], wm[gi6]);
	double3 um_add = make_double3(um_norm[gi1], um_norm[gi3], um_norm[gi5]);
	double3 um_sub = make_double3(um_norm[gi2], um_norm[gi4], um_norm[gi6]);

	double3 dwm = (wm_m * (a_add2 - a_sub2) + (wm_add ^ a_sub2) - (wm_sub ^ a_add2)) / a_addsub;
	double3 dum = (um_m * (a_add2 - a_sub2) + (um_add ^ a_sub2) - (um_sub ^ a_add2)) / a_addsub;
	double3 d2um = 2 * ((um_add ^ a_sub) + (um_sub ^ a_add) - um_m * (a_add + a_sub)) / a_addsub;
	T2[gi] = -(dwm * dum) + (1 - wm_m) * (d2um.x + d2um.y + d2um.z);
}

__global__ static void calChemD(double* vm, double* um, double* wm, double* T0, double* T1, double* T2, double3* rm, int time)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int yi = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int zi = threadIdx.z + blockIdx.z * blockDim.z + 1;
	if (xi > params.LX - 1 || yi > params.LY - 1 || zi > params.LZ - 1) {
		return;
	}
	int gi = get_index(xi, yi, zi, 1);
	//double I = params.I * xi / params.LX;
	double I = 0;
	if (time > 10000) {
		I = 0.01 * zi / params.LZ;
	}

	double dvm = vm[gi];
	double dum = um[gi];
	double dwm = wm[gi];
	//Forward Euler method
	//vm[gi] += params.dt * (-dvm * T0[gi] + params.ep * fv(dum, dvm, dwm, I));
	//um[gi] += params.dt * (-dum * T0[gi] + T1[gi] + T2[gi] + fu(dum, dvm, dwm, I));


	//Fourth order Runge Kutta method
	double k1_vm, k2_vm, k3_vm, k4_vm;
	double k1_um, k2_um, k3_um, k4_um;

	// k1
	k1_vm = params.dt * (-dvm * T0[gi] + params.ep * fv(dum, dvm, dwm, I));
	k1_um = params.dt * (-dum * T0[gi] + T1[gi] + T2[gi] + fu(dum, dvm, dwm, I));

	// k2
	k2_vm = params.dt * (-dvm * (T0[gi] + 0.5 * k1_vm) + params.ep * fv(dum + 0.5 * k1_um, dvm + 0.5 * k1_vm, dwm, I));
	k2_um = params.dt * (-dum * (T0[gi] + 0.5 * k1_vm) + (T1[gi] + 0.5 * k1_um) + (T2[gi] + 0.5 * k1_um) + fu(dum + 0.5 * k1_um, dvm + 0.5 * k1_vm, dwm, I));

	// k3
	k3_vm = params.dt * (-dvm * (T0[gi] + 0.5 * k2_vm) + params.ep * fv(dum + 0.5 * k2_um, dvm + 0.5 * k2_vm, dwm, I));
	k3_um = params.dt * (-dum * (T0[gi] + 0.5 * k2_vm) + (T1[gi] + 0.5 * k2_um) + (T2[gi] + 0.5 * k2_um) + fu(dum + 0.5 * k2_um, dvm + 0.5 * k2_vm, dwm, I));

	// k4
	k4_vm = params.dt * (-dvm * (T0[gi] + k3_vm) + params.ep * fv(dum + k3_um, dvm + k3_vm, dwm, I));
	k4_um = params.dt * (-dum * (T0[gi] + k3_vm) + (T1[gi] + k3_um) + (T2[gi] + k3_um) + fu(dum + k3_um, dvm + k3_vm, dwm, I));

	vm[gi] += (k1_vm + 2 * k2_vm + 2 * k3_vm + k4_vm) / 6;
	um[gi] += (k1_um + 2 * k2_um + 2 * k3_um + k4_um) / 6;
	//if (time == 1000 && yi < params.LY / 2) {
	//	//printf("%d,%d,%d,%d,%f,%f\n", time,xi,yi,zi, params.vss, params.uss);
	//	vm[gi] = params.vss;
	//	um[gi] = params.uss;
	//}
}

__global__ static void calChemBoundaryD(double* um, double* um_norm, double* vm, double* vm_norm, double* wm, int* map_element, int time)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x;
	int yi = threadIdx.y + blockIdx.y * blockDim.y;
	int zi = threadIdx.z + blockIdx.z * blockDim.z;
	int gi = get_index(xi, yi, zi, 1);
	int element_type = map_element[gi];
	if (xi > params.LX || yi > params.LY || zi > params.LZ) {
		return;
	}
	//Periodic Boundary
	//int gi1 = get_index(xi + params.um_offset_periodic[element_type].x, yi + params.um_offset_periodic[element_type].y, zi + params.um_offset_periodic[element_type].z, 1);
	//No-flux Boundary
	int gi1 = get_index(xi + params.um_offset_noflux[element_type].x, yi + params.um_offset_noflux[element_type].y, zi + params.um_offset_noflux[element_type].z, 1);
	um[gi] = um[gi1];
	vm[gi] = vm[gi1];
	//if (element_type == 1) {
	//	um[gi] = 0.4;
	//}
	//if (xi > 28 && xi < 32) {
	//	um[gi] = 0.4;
	//}
	um_norm[gi] = um[gi] / (1 - wm[gi]);
	vm_norm[gi] = vm[gi] / (1 - wm[gi]);
}

__global__ static void calUnnormD(double* un_norm, double* um_norm, double* vn_norm, double* vm_norm)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int yi = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int zi = threadIdx.z + blockIdx.z * blockDim.z + 1;
	if (xi > params.LX || yi > params.LY || zi > params.LZ) {
		return;
	}
	double dun_norm = 0;
	double dvn_norm = 0;
#pragma unroll 
	for (int i = 0; i < 2; i++) {
#pragma unroll 
		for (int j = 0; j < 2; j++) {
#pragma unroll 
			for (int k = 0; k < 2; k++) {
				dun_norm += um_norm[get_index(xi - i, yi - j, zi - k, 1)];
				dvn_norm += vm_norm[get_index(xi - i, yi - j, zi - k, 1)];
			}
		}
	}
	un_norm[get_index(xi, yi, zi, 2)] = dun_norm / 8;
	vn_norm[get_index(xi, yi, zi, 2)] = dvn_norm / 8;
}

__global__ static void recordCenterElementD(double* vm_center, double* wm_center, double3* rm_center, double3* Fn_center, double3* Veln_center, double* vm, double* wm, double3* rn, double3* Fn, double3* Veln, int time)
{
	int gi = get_index(params.LX / 2, params.LY / 2, params.LZ / 2, 1);
	int hi = get_index((params.LX + 1) / 2, (params.LY + 1) / 2, (params.LZ + 1) / 2, 2);
	vm_center[time] = vm[gi];
	wm_center[time] = wm[gi];
	rm_center[time] = rn[hi];
	Fn_center[time] = Fn[hi];
	Veln_center[time] = Veln[hi];
}

__global__ static void calFilamentD(double* vn_norm, double* un_norm, double3* filament, int time, unsigned int* hitCnt)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int yi = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int zi = threadIdx.z + blockIdx.z * blockDim.z + 1;
	if (xi > params.LX - 1 || yi > params.LY - 1 || zi > params.LZ - 1) {
		return;
	}
	double Viso = 0.15;
	// 读取 8 结点值
	double u00 = un_norm[get_index(xi, yi, zi, 2)];
	double u10 = un_norm[get_index(xi + 1, yi, zi, 2)];
	double u01 = un_norm[get_index(xi, yi + 1, zi, 2)];
	double u11 = un_norm[get_index(xi + 1, yi + 1, zi, 2)];
	double v00 = vn_norm[get_index(xi, yi, zi, 2)];
	double v10 = vn_norm[get_index(xi + 1, yi, zi, 2)];
	double v01 = vn_norm[get_index(xi, yi + 1, zi, 2)];
	double v11 = vn_norm[get_index(xi + 1, yi + 1, zi, 2)];

	double xb, yb;
	bool ok = solveBilinear(u00, u10, u01, u11, v00, v10, v01, v11, Viso, xb, yb);
	if (ok) {
		// 获得一个全局唯一的下标，并把计数器 +1
		unsigned int gi = atomicAdd(hitCnt, 1u);
		filament[time * params.maxFilamentlen + gi].x = xi + xb;
		filament[time * params.maxFilamentlen + gi].y = yi + yb;
		filament[time * params.maxFilamentlen + gi].z = zi;
	}
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

__global__ static void k_set_force(float3* F) {
	int i = blockDim.x * blockIdx.x + threadIdx.x; 
	if (i < p.N) { 
		F[i] = p.F_const; 
	}
}

__global__ static void k_init(float* f, float* rho, float3* u) {
	int i = blockDim.x * blockIdx.x + threadIdx.x; 
	if (i >= p.N) 
		return;
	rho[i] = 1.f; 
	u[i] = make_float3(0.f, 0.f, 0.f);
#pragma unroll
	for (int q = 0; q < 19; ++q) 
		f[i + q * (size_t)p.N] = p.w[q];
}

__global__ static void k_macros(float* f, float* rho, float3* u, float3* F) {
	int i = blockDim.x * blockIdx.x + threadIdx.x; 
	if (i >= p.N) return;
	float rh = 0.f;
	float3 s = make_float3 (0.f, 0.f, 0.f);
#pragma unroll
	for (int q = 0; q < 19; ++q) {
		float fi = f[i + q * (size_t)p.N];
		rh += fi; 
		s += fi * p.c[q];
	}
	rho[i] = rh;
	u[i] = (s + 0.5f * F[i]) / rh;
}

__global__ static void k_zero(float3* a) {
	int i = blockDim.x * blockIdx.x + threadIdx.x; 
	if (i < p.N) 
		a[i] = make_float3(0.f, 0.f, 0.f);
}

__global__ static void k_ibm_interpolate(float3* u, float3* Ul, float3* lag) {
	int l = blockDim.x * blockIdx.x + threadIdx.x;
	if (l >= p.M) return;

	const float h = p.h;
	const int Nx = p.L.x, Ny = p.L.y, Nz = p.L.z;

	const float gx = lag[l].x / h;
	const float gy = lag[l].y / h;
	const float gz = lag[l].z / h;

	const int ix = (int)floor(gx);
	const int iy = (int)floor(gy);
	const int iz = (int)floor(gz);

	float3 ul = make_float3(0.0, 0.0, 0.0);
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
				wsum += w;
			}
		}
	}
	Ul[l] = ul / wsum;
}

__global__ static void k_scale_negbeta(float3* Ul, float3* Vl, float3* Fl, float beta_eff) {
	int l = blockDim.x * blockIdx.x + threadIdx.x; 
	if (l >= p.M) return;
	//Fl[l] = p.beta * (Vl[l] - Ul[l]);
	Fl[l] = beta_eff * (Vl[l] - Ul[l]);
	//if (l == 1) {
	//	printf("%f, %f\n", Fl[l].x, Fl[l].y);
	//}
}

__global__ static void k_add_reaction_to_gel(int* bIndex, double3* Fn, float3* Fl)
{
	int l = blockDim.x * blockIdx.x + threadIdx.x;
	if (l >= p.M) return;

	int id = bIndex[l];
	float3 f = Fl[l];

	Fn[id].x = -(double)f.x;
	Fn[id].y = -(double)f.y;
	Fn[id].z = -(double)f.z;
	//if (l == 0) {
	//	printf("%f, %f， %f\n", Fl[l].x, Fl[l].y, Fl[l].z);
	//}
}

__global__ static void k_ibm_spread(float3* Fl, float3* lag, float3* F_ibm, float* dA) {
	int l = blockDim.x * blockIdx.x + threadIdx.x;
	if (l >= p.M) return;

	const float h = p.h; 
	const int Nx = p.L.x, Ny = p.L.y, Nz = p.L.z;

	const float gx = lag[l].x / h;
	const float gy = lag[l].y / h;
	const float gz = lag[l].z / h;

	const int ix = (int)floor(gx);
	const int iy = (int)floor(gy);
	const int iz = (int)floor(gz);

	const float3 fL = Fl[l];

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
			}
		}
	}
}

__global__ static void k_collide(float* f, float* fpost, float* rho, float3* u, float3* F) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= p.N) return;
	float rh = rho[i];
	float inv_tau = 1.f / p.tau;
	float uu = u[i] * u[i];
#pragma unroll
	for (int q = 0; q < 19; ++q) {
		int3 c = p.c[q];
		float cu = c * u[i];
		float feq = p.w[q] * rh * (1.f + (cu / p.cs2) + 0.5f * (cu * cu) / (p.cs2 * p.cs2) - 0.5f * (uu / p.cs2));
		float ciF = c * F[i];
		float uF = u[i] * F[i];
		float Fterm = p.w[q] * (1.f - 0.5f * inv_tau) * ((ciF / p.cs2) + (cu * ciF) / (p.cs2 * p.cs2) - (uF / p.cs2));
		float fi = f[i + q * (size_t)p.N];
		fpost[i + q * (size_t)p.N] = fi - inv_tau * (fi - feq) + Fterm;
	}
}

__global__ static void k_stream_bounce(float* fpost, float* fnext) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int N = p.N; if (i >= N) return;
	const int Nx = p.L.x, Ny = p.L.y, Nz = p.L.z;
	int tmp = i / Nx;
	int3 r = make_int3(i % Nx, tmp % Ny, tmp / Ny);
#pragma unroll
	for (int q = 0; q < 19; ++q) {
		int3 rn = r + p.c[q];
		bool inside = (0 <= rn.x && rn.x < Nx &&
			0 <= rn.y && rn.y < Ny &&
			0 <= rn.z && rn.z < Nz);
		if (inside) {
			int idn = id3(rn.x, rn.y, rn.z, Nx, Ny);
			fnext[idn + q * (size_t)N] = fpost[i + q * (size_t)N];
		}
		else {
			int qo = p.opp[q];
			fnext[i + qo * (size_t)N] = fpost[i + q * (size_t)N];
		}
	}
}

__global__ static void k_vec_add(float3* A, float3* B, float3* C) {
	int i = blockDim.x * blockIdx.x + threadIdx.x; 
	if (i < p.N) 
		C[i] = A[i] + B[i];
}

// 推荐的窄化工具（最近偶数舍入）
__device__ __forceinline__ float3 to_float3(const double3 a) {
	return make_float3(__double2float_rn(a.x),
		__double2float_rn(a.y),
		__double2float_rn(a.z));
}

// 你的 kernel：double3 → float3 显式转换
__global__ static void k_gather_boundary(int* bIndex, double3* rn, double3* vn, float3* lag, float3* Vl)
{
	int l = blockDim.x * blockIdx.x + threadIdx.x;
	if (l >= p.M) return;

	int id = bIndex[l];

	// 读一次寄存器，再转换
	const double3 r = rn[id];
	const double3 v = vn[id];

	lag[l] = to_float3(r);
	Vl[l] = to_float3(v);
}