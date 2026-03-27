#include "stdio.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include "gel_kernels.cuh"

__device__ static double3 operator+(double3 a, double3 b)
{
	return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ static double3 operator-(double3 a, double3 b)
{
	return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ static double3 operator-(double3 a)
{
	return make_double3(-a.x, -a.y, -a.z);
}

__device__ static double3 operator*(double a, double3 b)
{
	return make_double3(a * b.x, a * b.y, a * b.z);
}

__device__ static double3 operator*(double3 a, double b)
{
	return make_double3(a.x * b, a.y * b, a.z * b);
}

__device__ static double operator*(double3 a, double3 b)
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

__device__ static int get_index(int xi, int yi, int zi, int size, int LX, int LY)
{
        return xi + yi * (LX + size) + zi * (LY + size) * (LX + size);
}

// Tube mask helper
// Returns true if element (xi,yi,zi) is inside the hollow cavity.
// tube_mode: 0=solid, 1=square, 2=cylinder.
// Uses element 1-based indices directly from the kernel.
__device__ static bool isVoidElement(int yi, int zi, GelParams* gp)
{
    if (gp->tube_mode == 0) return false;   // solid gel
    float dy = (float)yi - gp->tube_cy;
    float dz = (float)zi - gp->tube_cz;
    if (gp->tube_mode == 1) {               // square tube
        return (fabsf(dy) < gp->tube_inner_hy && fabsf(dz) < gp->tube_inner_hz);
    }
    // cylinder (mode 2)
    float rhy = gp->tube_inner_hy, rhz = gp->tube_inner_hz;
    if (rhy < 0.5f || rhz < 0.5f) return false;
    float r2 = (dy*dy)/(rhy*rhy) + (dz*dz)/(rhz*rhz);
    return (r2 < 1.0f);
}

__device__ static int get_index_rm_loc(int xi, int yi, int zi, int num, int LX, int LY)
{
	return num + xi * 8 + yi * (LX + 1) * 8 + zi * (LY + 1) * (LX + 1) * 8;
}

__device__ static int get_index_nmSm(int xi, int yi, int zi, int num, int LX, int LY)
{
	return num + xi * 6 + yi * (LX + 1) * 6 + zi * (LY + 1) * (LX + 1) * 6;
}

__device__ static double fv(double u, double v, double w, double I = 0)
{
	double P1 = 0.0124;
	double P2 = 0.77;
	return (1.0 - w) * (1.0 - w) * u - (1.0 - w) * v + I * (0.5 * P1 + P2);
	//return (1.0 - w) * (1.0 - w) * u - (1.0 - w) * v;
}

__device__ static double fu(double u, double v, double w, double f, double I = 0)
{
	double P1 = 0.0124;
	double P2 = 0.77;
	double q = 1e-4;
	double ww = (1.0 - w) * (1.0 - w);
	return ww * u - u * u - (1.0 - w) * (f * v + I * P1) * (u - q * ww) / (u + q * ww) + I * P2;
	//return ww * u - u * u - (1.0 - w) * (f * v + I) * (u - q * ww) / (u + q * ww);
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

__device__ bool solveBilinear(double A, double B, double C, double D, double E, double F, double G, double H, double vIso, double& xb, double& yb, double eps = 1e-12)
{
	// ---------- ϵ�� ----------
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

	// ---------- ����� / һ�η��� ----------
	if (fabs(c2) < eps) {               // ����
		if (fabs(c1) < eps) return false;
		roots[n++] = -c0 / c1;
	}
	else {
		double disc = c1 * c1 - 4.0 * c2 * c0;
		if (disc < 0.0) return false;    // ��ʵ��
		double s = sqrt(disc);
		roots[n++] = (-c1 + s) / (2.0 * c2);
		roots[n++] = (-c1 - s) / (2.0 * c2);
	}

	// ---------- �ش��� y����ɸѡ�Ϸ��� ----------
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
	return false;                        // û�н������ڵ�Ԫ��
}

__global__ void calElementPropertiesD(double3* rn, double3* rm, double3* rm_loc, double3* nmSm, double* volume, double* wm, double* wmp, GelParams* gp)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x;
	int yi = threadIdx.y + blockIdx.y * blockDim.y;
	int zi = threadIdx.z + blockIdx.z * blockDim.z;
	int LX = gp->LX;
	int LY = gp->LY;
	int LZ = gp->LZ;
	if (xi > LX || yi > LY || zi > LZ) {
		return;
	}

	int gi = get_index(xi, yi, zi, 1, LX, LY);
	int gi_rm_loc = get_index_rm_loc(xi, yi, zi, 0, LX, LY);
	int gi_nmSm = get_index_nmSm(xi, yi, zi, 0, LX, LY);
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
				node[count] = rn[get_index(xi + i, yi + j, zi + k, 2, LX, LY)];
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
		wm[gi] = gp->dx * gp->dy * gp->dz * gp->FA0 / vol;
	}
	else {
		double vol = 0;
		int map[6][4] = { {0, 1, 3, 2}, {4, 6, 7, 5}, {0, 4, 5, 1}, {2, 3, 7, 6}, {0, 2, 6, 4}, {1, 5, 7, 3} };
#pragma unroll
		for (int i = 0; i < 6; i++)
			nmSm[gi_nmSm + i] = cal_face_properties(node[map[i][0]], node[map[i][1]], node[map[i][2]], node[map[i][3]], vol);
		volume[gi] = vol;
		wmp[gi] = wm[gi];
		wm[gi] = gp->dx * gp->dy * gp->dz * gp->FA0 / vol;
	}
}

__global__ void calPressureD(double* pm, double* vm, double* wm, GelParams* gp)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int yi = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int zi = threadIdx.z + blockIdx.z * blockDim.z + 1;
	int LX = gp->LX;
	int LY = gp->LY;
	int LZ = gp->LZ;
	if (xi > LX - 1 || yi > LY - 1 || zi > LZ - 1) {
		return;
	}
	int gi = get_index(xi, yi, zi, 1, LX, LY);
	double wmt = wm[gi];
	if (gp->gelType == 1) {
		pm[gi] = -(wmt + log(1.0 - wmt) + (gp->CH0 + gp->CH1 * wmt) * wmt * wmt) + gp->C0 * wmt / (2.0 * gp->FA0) + gp->CHS * wmt * vm[gi];
	}
	else if (gp->gelType == 2) {
		double c0 = gp->c0_bis + gp->b * vm[gi];
		pm[gi] = -(wmt + log(1.0 - wmt) + (gp->CH0 + gp->CH1 * wmt) * wmt * wmt) + c0 * wmt / (2.0 * gp->FA0);
	}
}

__global__ void calNodesVelocityD(double3* rn, double3* ven, double3* ves, double3* Fn, double3* Fn_robin, double3* Fdrag_robin, double3* nmSm, double* pm, double* wm, double* vn_norm, GelParams* gp)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int yi = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int zi = threadIdx.z + blockIdx.z * blockDim.z + 1;
	int LX = gp->LX;
	int LY = gp->LY;
	int LZ = gp->LZ;
	if (xi > LX || yi > LY || zi > LZ) {
		return;
	}
	int gi = get_index(xi, yi, zi, 2, LX, LY);
	//calculate mobility
	double wn = 0;
#pragma unroll
	for (int i = 0; i < 2; i++)
#pragma unroll
		for (int j = 0; j < 2; j++)
#pragma unroll
			for (int k = 0; k < 2; k++)
				wn += wm[get_index(xi - i, yi - j, zi - k, 1, LX, LY)];
	wn /= 8;
	double Mn = 8. * gp->AZ0 * sqrt(gp->FA0 / wn) * (1 - wn) / (gp->dx * gp->dy * gp->dz);
	//calculate force
	double3 f1n = make_double3(0., 0., 0.);
	double3 f2n = make_double3(0., 0., 0.);
	double3 rn_m = rn[gi];
	int gi_nmSm;
	if (xi < LX && yi < LY && zi < LZ) {
		gi_nmSm = get_index_nmSm(xi, yi, zi, 0, LX, LY);
		f1n += rn[get_index(xi + 1, yi + 1, zi + 1, 2, LX, LY)] + rn[get_index(xi, yi + 1, zi + 1, 2, LX, LY)] + rn[get_index(xi + 1, yi, zi + 1, 2, LX, LY)] + rn[get_index(xi + 1, yi + 1, zi, 2, LX, LY)] - 4 * rn_m;
		f2n += pm[get_index(xi, yi, zi, 1, LX, LY)] * (nmSm[gi_nmSm] + nmSm[gi_nmSm + 2] + nmSm[gi_nmSm + 4]);
	}
	if (xi < LX && yi < LY && zi > 1) {
		gi_nmSm = get_index_nmSm(xi, yi, zi - 1, 0, LX, LY);
		f1n += rn[get_index(xi + 1, yi + 1, zi - 1, 2, LX, LY)] + rn[get_index(xi, yi + 1, zi - 1, 2, LX, LY)] + rn[get_index(xi + 1, yi, zi - 1, 2, LX, LY)] + rn[get_index(xi + 1, yi + 1, zi, 2, LX, LY)] - 4 * rn_m;
		f2n += pm[get_index(xi, yi, zi - 1, 1, LX, LY)] * (nmSm[gi_nmSm] + nmSm[gi_nmSm + 2] + nmSm[gi_nmSm + 5]);
	}
	if (xi < LX && yi > 1 && zi < LZ) {
		gi_nmSm = get_index_nmSm(xi, yi - 1, zi, 0, LX, LY);
		f1n += rn[get_index(xi + 1, yi - 1, zi + 1, 2, LX, LY)] + rn[get_index(xi, yi - 1, zi + 1, 2, LX, LY)] + rn[get_index(xi + 1, yi, zi + 1, 2, LX, LY)] + rn[get_index(xi + 1, yi - 1, zi, 2, LX, LY)] - 4 * rn_m;
		f2n += pm[get_index(xi, yi - 1, zi, 1, LX, LY)] * (nmSm[gi_nmSm] + nmSm[gi_nmSm + 3] + nmSm[gi_nmSm + 4]);
	}
	if (xi < LX && yi > 1 && zi > 1) {
		gi_nmSm = get_index_nmSm(xi, yi - 1, zi - 1, 0, LX, LY);
		f1n += rn[get_index(xi + 1, yi - 1, zi - 1, 2, LX, LY)] + rn[get_index(xi, yi - 1, zi - 1, 2, LX, LY)] + rn[get_index(xi + 1, yi, zi - 1, 2, LX, LY)] + rn[get_index(xi + 1, yi - 1, zi, 2, LX, LY)] - 4 * rn_m;
		f2n += pm[get_index(xi, yi - 1, zi - 1, 1, LX, LY)] * (nmSm[gi_nmSm] + nmSm[gi_nmSm + 3] + nmSm[gi_nmSm + 5]);
	}
	if (xi > 1 && yi < LY && zi < LZ) {
		gi_nmSm = get_index_nmSm(xi - 1, yi, zi, 0, LX, LY);
		f1n += rn[get_index(xi - 1, yi + 1, zi + 1, 2, LX, LY)] + rn[get_index(xi, yi + 1, zi + 1, 2, LX, LY)] + rn[get_index(xi - 1, yi, zi + 1, 2, LX, LY)] + rn[get_index(xi - 1, yi + 1, zi, 2, LX, LY)] - 4 * rn_m;
		f2n += pm[get_index(xi - 1, yi, zi, 1, LX, LY)] * (nmSm[gi_nmSm + 1] + nmSm[gi_nmSm + 2] + nmSm[gi_nmSm + 4]);
	}
	if (xi > 1 && yi < LY && zi > 1) {
		gi_nmSm = get_index_nmSm(xi - 1, yi, zi - 1, 0, LX, LY);
		f1n += rn[get_index(xi - 1, yi + 1, zi - 1, 2, LX, LY)] + rn[get_index(xi, yi + 1, zi - 1, 2, LX, LY)] + rn[get_index(xi - 1, yi, zi - 1, 2, LX, LY)] + rn[get_index(xi - 1, yi + 1, zi, 2, LX, LY)] - 4 * rn_m;
		f2n += pm[get_index(xi - 1, yi, zi - 1, 1, LX, LY)] * (nmSm[gi_nmSm + 1] + nmSm[gi_nmSm + 2] + nmSm[gi_nmSm + 5]);
	}
	if (xi > 1 && yi > 1 && zi < LZ) {
		gi_nmSm = get_index_nmSm(xi - 1, yi - 1, zi, 0, LX, LY);
		f1n += rn[get_index(xi - 1, yi - 1, zi + 1, 2, LX, LY)] + rn[get_index(xi, yi - 1, zi + 1, 2, LX, LY)] + rn[get_index(xi - 1, yi, zi + 1, 2, LX, LY)] + rn[get_index(xi - 1, yi - 1, zi, 2, LX, LY)] - 4 * rn_m;
		f2n += pm[get_index(xi - 1, yi - 1, zi, 1, LX, LY)] * (nmSm[gi_nmSm + 1] + nmSm[gi_nmSm + 3] + nmSm[gi_nmSm + 4]);
	}
	if (xi > 1 && yi > 1 && zi > 1) {
		gi_nmSm = get_index_nmSm(xi - 1, yi - 1, zi - 1, 0, LX, LY);
		f1n += rn[get_index(xi - 1, yi - 1, zi - 1, 2, LX, LY)] + rn[get_index(xi, yi - 1, zi - 1, 2, LX, LY)] + rn[get_index(xi - 1, yi, zi - 1, 2, LX, LY)] + rn[get_index(xi - 1, yi - 1, zi, 2, LX, LY)] - 4 * rn_m;
		f2n += pm[get_index(xi - 1, yi - 1, zi - 1, 1, LX, LY)] * (nmSm[gi_nmSm + 1] + nmSm[gi_nmSm + 3] + nmSm[gi_nmSm + 5]);
	}
	if (gp->gelType == 1) {
		f1n *= pow(gp->dx * gp->dy * gp->dz, 1 / 3) * gp->C0 / 12.;
	}
	else if (gp->gelType == 2) {
		double c0 = gp->c0_bis + gp->b * vn_norm[gi] * (1 - wn);
		f1n *= pow(gp->dx * gp->dy * gp->dz, 1 / 3) * c0 / 12.;
	}
	f2n /= 4.;
	Fn[gi] = f1n + f2n;
	//Fn[gi] = f1n + f2n + Fn_robin[gi];
	ven[gi] = Mn * Fn[gi];
	ves[gi] = -wn * ven[gi] / (1 - wn);
}

__global__ void calInternalNodesPositionD(double3* rn, double3* ven, GelParams* gp)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int yi = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int zi = threadIdx.z + blockIdx.z * blockDim.z + 1;
	int LX = gp->LX;
	int LY = gp->LY;
	int LZ = gp->LZ;
	if (xi > LX || yi > LY || zi > LZ) {
		return;
	}
	int gi = get_index(xi, yi, zi, 2, LX, LY);
	rn[gi] += gp->dtx * ven[gi];
}

__global__ void calServiceNodesPositionD(double3* rn, int* map_node, GelParams* gp)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x;
	int yi = threadIdx.y + blockIdx.y * blockDim.y;
	int zi = threadIdx.z + blockIdx.z * blockDim.z;
	int LX = gp->LX;
	int LY = gp->LY;
	int LZ = gp->LZ;
	int gi = get_index(xi, yi, zi, 2, LX, LY);
	int node_type = map_node[gi];
	if (node_type == 0 || xi > LX + 1 || yi > LY + 1 || zi > LZ + 1) {
		return;
	}
	int gi1 = get_index(xi + gp->rn_offset[node_type].x, yi + gp->rn_offset[node_type].y, zi + gp->rn_offset[node_type].z, 2, LX, LY);
	int gi2 = get_index(xi + 2 * gp->rn_offset[node_type].x, yi + 2 * gp->rn_offset[node_type].y, zi + 2 * gp->rn_offset[node_type].z, 2, LX, LY);
	rn[gi] = 2 * rn[gi1] - rn[gi2];
}

__global__ void calTermsD(double* T0, double* T1, double* T2, double* wm, double* wmp, double3* ven, double3* nmSm, double* volume, double3* rm_loc, double* un_norm, double* um_norm, double3* rm, GelParams* gp)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int yi = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int zi = threadIdx.z + blockIdx.z * blockDim.z + 1;
	int LX = gp->LX;
	int LY = gp->LY;
	int LZ = gp->LZ;
	if (xi > LX - 1 || yi > LY - 1 || zi > LZ - 1) {
		return;
	}
	int gi = get_index(xi, yi, zi, 1, LX, LY);

	double wm_m = wm[gi];
	double um_m = um_norm[gi];
	double3 rm_m = rm[gi];
	//cal T0

	T0[gi] = (1 - wm_m / wmp[gi]) / gp->dtx;

	//cal T1
	double dT1 = 0;
	double3 vmum = make_double3(0, 0, 0);
	int count = 0;
	int gi_rm_loc = get_index_rm_loc(xi, yi, zi, 0, LX, LY);
#pragma unroll
	for (int i = 0; i < 2; i++) {
#pragma unroll
		for (int j = 0; j < 2; j++) {
#pragma unroll
			for (int k = 0; k < 2; k++) {
				int id1 = gi_rm_loc + count;
				count++;
				int id2 = get_index(xi + i, yi + j, zi + k, 2, LX, LY);
				double N = 0.125 * (1 + (2 * double(i) - 1) * rm_loc[id1].x) * (1 + (2 * double(j) - 1) * rm_loc[id1].y) * (1 + (2 * double(k) - 1) * rm_loc[id1].z);
				vmum += N * ven[id2] * un_norm[id2];
			}
		}
	}
	int gi_nmSm = get_index_nmSm(xi, yi, zi, 0, LX, LY);
#pragma unroll
	for (int i = 0; i < 6; i++) {
		dT1 += nmSm[gi_nmSm + i] * vmum;
	}
	T1[gi] = dT1 / volume[gi];

	//cal T2
	int gi1 = get_index(xi + 1, yi, zi, 1, LX, LY);
	int gi2 = get_index(xi - 1, yi, zi, 1, LX, LY);
	int gi3 = get_index(xi, yi + 1, zi, 1, LX, LY);
	int gi4 = get_index(xi, yi - 1, zi, 1, LX, LY);
	int gi5 = get_index(xi, yi, zi + 1, 1, LX, LY);
	int gi6 = get_index(xi, yi, zi - 1, 1, LX, LY);

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

__global__ void calChemD(double* vm, double* um, double* wm, double* T0, double* T1, double* T2, double3* rm, int time, GelParams* gp)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int yi = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int zi = threadIdx.z + blockIdx.z * blockDim.z + 1;
	int LX = gp->LX;
	int LY = gp->LY;
	int LZ = gp->LZ;
	if (xi > LX - 1 || yi > LY - 1 || zi > LZ - 1) {
		return;
	}
	int gi = get_index(xi, yi, zi, 1, LX, LY);
	if (isVoidElement(yi, zi, gp)) {
		vm[gi] = gp->vss;
		um[gi] = gp->uss;
		wm[gi] = gp->wss;
		return;
	}
	//double I = gp->I * xi / gp->LX;
	double I = 0;

	double dvm = vm[gi];
	double dum = um[gi];
	double dwm = wm[gi];
	//Forward Euler method
	vm[gi] += gp->dt * (-dvm * T0[gi] + gp->ep * fv(dum, dvm, dwm, I));
	um[gi] += gp->dt * (-dum * T0[gi] + T1[gi] + T2[gi] + fu(dum, dvm, dwm, gp->f, I));


	//Fourth order Runge Kutta method
	//double k1_vm, k2_vm, k3_vm, k4_vm;
	//double k1_um, k2_um, k3_um, k4_um;

	//// k1
	//k1_vm = gp->dt * (-dvm * T0[gi] + gp->ep * fv(dum, dvm, dwm, I));
	//k1_um = gp->dt * (-dum * T0[gi] + T1[gi] + T2[gi] + fu(dum, dvm, dwm, I));

	//// k2
	//k2_vm = gp->dt * (-dvm * (T0[gi] + 0.5 * k1_vm) + gp->ep * fv(dum + 0.5 * k1_um, dvm + 0.5 * k1_vm, dwm, I));
	//k2_um = gp->dt * (-dum * (T0[gi] + 0.5 * k1_vm) + (T1[gi] + 0.5 * k1_um) + (T2[gi] + 0.5 * k1_um) + fu(dum + 0.5 * k1_um, dvm + 0.5 * k1_vm, dwm, I));

	//// k3
	//k3_vm = gp->dt * (-dvm * (T0[gi] + 0.5 * k2_vm) + gp->ep * fv(dum + 0.5 * k2_um, dvm + 0.5 * k2_vm, dwm, I));
	//k3_um = gp->dt * (-dum * (T0[gi] + 0.5 * k2_vm) + (T1[gi] + 0.5 * k2_um) + (T2[gi] + 0.5 * k2_um) + fu(dum + 0.5 * k2_um, dvm + 0.5 * k2_vm, dwm, I));

	//// k4
	//k4_vm = gp->dt * (-dvm * (T0[gi] + k3_vm) + gp->ep * fv(dum + k3_um, dvm + k3_vm, dwm, I));
	//k4_um = gp->dt * (-dum * (T0[gi] + k3_vm) + (T1[gi] + k3_um) + (T2[gi] + k3_um) + fu(dum + k3_um, dvm + k3_vm, dwm, I));
	//vm[gi] += (k1_vm + 2 * k2_vm + 2 * k3_vm + k4_vm) / 6;
	//um[gi] += (k1_um + 2 * k2_um + 2 * k3_um + k4_um) / 6;
}

__global__ void calChemBoundaryD(double* um, double* um_norm, double* vm, double* vm_norm, double* wm, int* map_element, int time, GelParams* gp)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x;
	int yi = threadIdx.y + blockIdx.y * blockDim.y;
	int zi = threadIdx.z + blockIdx.z * blockDim.z;
	int LX = gp->LX;
	int LY = gp->LY;
	int LZ = gp->LZ;
	int gi = get_index(xi, yi, zi, 1, LX, LY);
	int element_type = map_element[gi];
	if (xi > LX || yi > LY || zi > LZ) {
		return;
	}
	//Periodic Boundary
	//int gi1 = get_index(xi + gp->um_offset_periodic[element_type].x, yi + gp->um_offset_periodic[element_type].y, zi + gp->um_offset_periodic[element_type].z, 1);
	//No-flux Boundary
	int gi1 = get_index(xi + gp->um_offset_noflux[element_type].x, yi + gp->um_offset_noflux[element_type].y, zi + gp->um_offset_noflux[element_type].z, 1, LX, LY);
	um[gi] = um[gi1];
	vm[gi] = vm[gi1];
	um_norm[gi] = um[gi] / (1 - wm[gi]);
	vm_norm[gi] = vm[gi] / (1 - wm[gi]);
}

__global__ void setZero(double* un_robin, double3* Fn_robin, double3* Fdrag_robin, GelParams* gp)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int yi = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int zi = threadIdx.z + blockIdx.z * blockDim.z + 1;
	int LX = gp->LX;
	int LY = gp->LY;
	int LZ = gp->LZ;
	if (xi > LX || yi > LY || zi > LZ) {
		return;
	}
	un_robin[get_index(xi, yi, zi, 2, LX, LY)] = 0;
	Fn_robin[get_index(xi, yi, zi, 2, LX, LY)] = make_double3(0., 0., 0.);
}

__global__ void calUnnormD(double* un_norm, double* un_robin, double* um_norm, double* vn_norm, double* vm_norm, GelParams* gp)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int yi = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int zi = threadIdx.z + blockIdx.z * blockDim.z + 1;
	int LX = gp->LX;
	int LY = gp->LY;
	int LZ = gp->LZ;
	if (xi > LX || yi > LY || zi > LZ) {
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
				dun_norm += um_norm[get_index(xi - i, yi - j, zi - k, 1, LX, LY)];
				dvn_norm += vm_norm[get_index(xi - i, yi - j, zi - k, 1, LX, LY)];
			}
		}
	}
	un_norm[get_index(xi, yi, zi, 2, LX, LY)] = dun_norm / 8 + un_robin[get_index(xi, yi, zi, 2, LX, LY)];
	vn_norm[get_index(xi, yi, zi, 2, LX, LY)] = dvn_norm / 8;
}

__global__ void recordCenterElementD(double* vm_center, double* wm_center, double3* rm_center, double3* Fn_center, double3* Veln_center, double* vm, double* wm, double3* rn, double3* Fn, double3* Veln, int time, GelParams* gp)
{
	int LX = gp->LX;
	int LY = gp->LY;
	int LZ = gp->LZ;
	int gi = get_index(LX / 2, LY / 2, LZ / 2, 1, LX, LY);
	int hi = get_index((LX + 1) / 2, (LY + 1) / 2, (LZ + 1) / 2, 2, LX, LY);
	vm_center[time] = vm[gi];
	wm_center[time] = wm[gi];
	rm_center[time] = rn[hi];
	Fn_center[time] = Fn[hi];
	Veln_center[time] = Veln[hi];
}

__global__ void calFilamentD(double* vn_norm, double* un_norm, double3* filament, int time, unsigned int* hitCnt, GelParams* gp)
{
	int xi = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int yi = threadIdx.y + blockIdx.y * blockDim.y + 1;
	int zi = threadIdx.z + blockIdx.z * blockDim.z + 1;
	int LX = gp->LX;
	int LY = gp->LY;
	int LZ = gp->LZ;
	if (xi > LX - 1 || yi > LY - 1 || zi > LZ - 1) {
		return;
	}
	double Viso = 0.15;
	double u00 = un_norm[get_index(xi, yi, zi, 2, LX, LY)];
	double u10 = un_norm[get_index(xi + 1, yi, zi, 2, LX, LY)];
	double u01 = un_norm[get_index(xi, yi + 1, zi, 2, LX, LY)];
	double u11 = un_norm[get_index(xi + 1, yi + 1, zi, 2, LX, LY)];
	double v00 = vn_norm[get_index(xi, yi, zi, 2, LX, LY)];
	double v10 = vn_norm[get_index(xi + 1, yi, zi, 2, LX, LY)];
	double v01 = vn_norm[get_index(xi, yi + 1, zi, 2, LX, LY)];
	double v11 = vn_norm[get_index(xi + 1, yi + 1, zi, 2, LX, LY)];

	double xb, yb;
	bool ok = solveBilinear(u00, u10, u01, u11, v00, v10, v01, v11, Viso, xb, yb);
	if (ok) {
		unsigned int gi = atomicAdd(hitCnt, 1u);
		filament[time * gp->maxFilamentlen + gi].x = xi + xb;
		filament[time * gp->maxFilamentlen + gi].y = yi + yb;
		filament[time * gp->maxFilamentlen + gi].z = zi;
	}
}
// ── anchorBottomNodesD ────────────────────────────────────────────────────────
// Pins the bottom interior layer (zi==1) of gel nodes: zeros their velocity
// and restores the reference z position so the base cannot drift.
__global__ void anchorBottomNodesD(double3* rn, double3* ven, double anchor_z, GelParams* gp)
{
    int xi = blockIdx.x * blockDim.x + threadIdx.x;
    int yi = blockIdx.y * blockDim.y + threadIdx.y;
    int zi = blockIdx.z * blockDim.z + threadIdx.z;
    int LX = gp->LX, LY = gp->LY, LZ = gp->LZ;
    if (xi < 1 || xi > LX || yi < 1 || yi > LY || zi != 1) return;
    int gi = xi + yi * (LX + 2) + zi * (LX + 2) * (LY + 2);
    ven[gi].x = 0.0; ven[gi].y = 0.0; ven[gi].z = 0.0;
    rn[gi].z = anchor_z;
}
