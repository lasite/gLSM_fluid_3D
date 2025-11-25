#include <cmath>
#include "fluid.h"
#include "fluid_kernels.cuh"
#include <memory.h>
#include <tuple>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <functional>
#include <cuda_runtime.h>
using namespace std;

int Fluid::idx3(int x, int y, int z, int Nx, int Ny)
{
	return x + y * Nx + z * Nx * Ny;
}

void Fluid::allocateHostStorage()
{
	cudaHostAlloc((void**)&h_u, N * sizeof(float3), cudaHostAllocPortable);
	memset(h_u, 0, N * sizeof(float3));
	cudaHostAlloc((void**)&h_c1, N * sizeof(float), cudaHostAllocPortable);
	memset(h_c1, 0, N * sizeof(float));
}

void Fluid::allocateDeviceStorage()
{
	cudaMalloc((void**)&d_fp, sizeof(FluidParams));
	cudaMalloc(&d_f, Nd * sizeof(float));
	cudaMalloc(&d_fpost, Nd * sizeof(float));
	cudaMalloc(&d_fnext, Nd * sizeof(float));
	cudaMalloc(&d_rho, N * sizeof(float));
	cudaMalloc(&d_u, N * sizeof(float3));
	cudaMalloc(&d_c1, N * sizeof(float));
	cudaMalloc(&d_c2, N * sizeof(float));
	cudaMalloc(&d_source, N * sizeof(float));
	cudaMalloc(&d_F, N * sizeof(float3));
	cudaMalloc(&d_F_ibm, N * sizeof(float3));
	cudaMalloc(&d_F_tot, N * sizeof(float3));
}

void Fluid::copyDataToDevice()
{
	cudaMemcpy(d_fp, h_fp, sizeof(FluidParams), cudaMemcpyHostToDevice);
}

void Fluid::setInitValue()
{
	k_init << <blocksN, threads, 0, fluid_stream >> > (d_f, d_rho, d_u, d_c1, d_c2, d_fp);
}

void Fluid::copyDataToHost()
{
	cudaMemcpyAsync(h_u, d_u, sizeof(float3) * N, cudaMemcpyDeviceToHost, fluid_stream);
	cudaMemcpyAsync(h_c1, d_c1, sizeof(float) * N, cudaMemcpyDeviceToHost, fluid_stream);
}

void Fluid::freeHostMemory()
{
	cudaFreeHost(h_u);
	cudaFreeHost(h_c1);
	delete h_fp;
}

void Fluid::freeDeviceMemory()
{
	cudaFree(d_f);
	cudaFree(d_fpost);
	cudaFree(d_fnext);
	cudaFree(d_rho);
	cudaFree(d_u);
	cudaFree(d_c1);
	cudaFree(d_c2);
	cudaFree(d_source);
	cudaFree(d_F);
	cudaFree(d_F_ibm);
	cudaFree(d_F_tot);
	cudaFree(d_fp);
}

void Fluid::recordData(int time)
{
	if (time % 1 == 0) {
		ofstream fVelb;
		string  str_Velb;
		str_Velb = "Velb" + to_string(time) + ".dat";
		fVelb.open(str_Velb);
		ofstream fC1;
		string str_C1 = "Conc" + to_string(time) + ".dat";
		fC1.open(str_C1);
		int gi;
		for (int zi = 0; zi < fluidNodeGrid.z; zi++) {
			for (int yi = 0; yi < fluidNodeGrid.y; yi++) {
				for (int xi = 0; xi < fluidNodeGrid.x; xi++) {
					gi = idx3(xi, yi, zi, fluidNodeGrid.x, fluidNodeGrid.y);
					fVelb << setw(9) << to_string(h_u[gi].x) << "      " << setw(9) << to_string(h_u[gi].y) << "      " << setw(9) << to_string(h_u[gi].z) << "\n";
					fC1 << setw(9) << to_string(h_c1[gi]) << "\n";
				}
			}
		}
		fVelb.close();
		fC1.close();
	}
}

void Fluid::writeFiles(int iter)
{
	double time = iter * dt;
	if (iter % 1000 == 0) {
		if (file_writer_thread.joinable()) {
			file_writer_thread.join();
		}
		copyDataToHost();
		file_writer_thread = thread(&Fluid::recordData, this, int(time));
	}
}

Fluid::Fluid(int3 fluidSize, int time, Coupler* coupler):
fluidSize(fluidSize),
coupler(coupler),
h_u(0),
h_c1(0),
d_f(0),
d_fpost(0),
d_fnext(0),
d_rho(0),
d_u(0),
d_F(0),
d_F_ibm(0),
d_F_tot(0),
d_A(0)
{
	dt = 1e-3f;
	h = 0.5f;
	fluidNodeGrid.x = int(fluidSize.x / h) + 1;
	fluidNodeGrid.y = int(fluidSize.y / h) + 1;
	fluidNodeGrid.z = int(fluidSize.z / h) + 1;
	N = fluidNodeGrid.x * fluidNodeGrid.y * fluidNodeGrid.z;
	Nd = N * 19;
	threads = coupler->threads;
	blocksN = (N + threads - 1) / threads;
	blocksM = coupler->blocksM;
	h_fp = new FluidParams;
	memset(h_fp, 0, sizeof(FluidParams));
	h_fp->dt = dt;
	h_fp->perX = 0;
	h_fp->perY = 0;
	h_fp->perZ = 0;
	h_fp->cs2 = 1.f / 3.f;
	h_fp->tau = 0.8f;
	h_fp->nu = h_fp->cs2 * (h_fp->tau - 0.5f);
	h_fp->N = N;
	h_fp->M = coupler->sumGelBoundaryCount;
	h_fp->L = fluidNodeGrid;
	h_fp->h = h;
	h_fp->dx = 1;
	h_fp->dy = 1;
	h_fp->dz = 1;
	h_fp->D = 1;
	niu = 1e-6f;
	dx_fluid = 40e-6f * h_fp->h;
	dt_fluid = (h_fp->tau - 0.5f) * dx_fluid * dx_fluid / (3 * niu);
	Nsub = int(dt / dt_fluid);
	h_fp->beta = 1.0f;
	int3 c[19] = {
	{0,0,0},
	{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
	{1,1,0},{-1,1,0},{-1,-1,0},{1,-1,0},
	{1,0,1},{-1,0,1},{-1,0,-1},{1,0,-1},
	{0,1,1},{0,1,-1},{0,-1,-1},{0,-1,1}
	};
	float w[19] = {
		1.f / 3.f,
		1.f / 18.f,1.f / 18.f,1.f / 18.f,1.f / 18.f,1.f / 18.f,1.f / 18.f,
		1.f / 36.f,1.f / 36.f,1.f / 36.f,1.f / 36.f,1.f / 36.f,1.f / 36.f,
		1.f / 36.f,1.f / 36.f,1.f / 36.f,1.f / 36.f,1.f / 36.f,1.f / 36.f
	};
	int opp[19] = { 0,2,1, 4,3, 6,5, 9,10,7,8, 13,14,11,12, 17,18,15,16 };
	memcpy(h_fp->c, c, sizeof(c));
	memcpy(h_fp->w, w, sizeof(w));
	memcpy(h_fp->opp, opp, sizeof(opp));
	h_fp->F_const = make_float3(0, 0, 0);
	_initialize(time);
}

Fluid::~Fluid()
{
	_finalize();
}

void Fluid::_initialize(int time)
{
	allocateHostStorage();
	allocateDeviceStorage();
	cudaStreamCreate(&fluid_stream);
	copyDataToDevice();
	setInitValue();
}

void Fluid::stepConcentration(int iter)
{
	if (iter % 1 == 0) {
		k_ibm_sample_concentration << <blocksM, threads, 0, fluid_stream >> > (d_c1, coupler->d_Dl_all_, coupler->d_lag_all_, d_fp);
		k_robin_boundary << <blocksM, threads, 0, fluid_stream >> > (coupler->d_Cl_all_, coupler->d_Dl_all_, coupler->d_Sl_all_, d_fp);
		k_zero_scalar << <blocksN, threads, 0, fluid_stream >> > (d_source, d_fp);
		k_ibm_spread_concentration << <blocksM, threads, 0, fluid_stream >> > (d_source, coupler->d_Sl_all_, coupler->d_lag_all_, d_A, d_fp);
		k_convection_diffusion << <blocksN, threads, 0, fluid_stream >> > (d_u, d_c1, d_c2, d_source, d_fp);
		swap(d_c1, d_c2);
	}
}

void Fluid::stepVelocity(int iter)
{
	float ramp = fminf(1, (iter + 1) / 2000.0f);
	float beta_eff = h_fp->beta * ramp;
	double time = iter * dt;
	for (int kk = 0; kk < Nsub; kk++) {
		k_set_force << <blocksN, threads, 0, fluid_stream >> > (d_F, d_fp);
		k_ibm_interpolate_velocity << <blocksM, threads, 0, fluid_stream >> > (d_u, coupler->d_Ul_all_, coupler->d_lag_all_, d_fp);
		k_scale_negbeta << <blocksM, threads, 0, fluid_stream >> > (coupler->d_Ul_all_, coupler->d_Vl_all_, coupler->d_Fl_all_, beta_eff, d_fp);
		k_ibm_spread_forces << <blocksM, threads, 0, fluid_stream >> > (d_F_ibm, coupler->d_Fl_all_, coupler->d_lag_all_, d_A, d_fp);
		k_vec_add << <blocksN, threads, 0, fluid_stream >> > (d_F, d_F_ibm, d_F_tot, d_fp);
		k_macros << <blocksN, threads, 0, fluid_stream >> > (d_f, d_rho, d_u, d_F_tot, d_fp);
		k_collide << <blocksN, threads, 0, fluid_stream >> > (d_f, d_fpost, d_rho, d_u, d_F_tot, d_fp);
		k_stream_bounce << <blocksN, threads, 0, fluid_stream >> > (d_fpost, d_fnext, d_fp);
		swap(d_f, d_fnext);
		k_zero_vector << <blocksN, threads, 0, fluid_stream >> > (d_F_ibm, d_fp);
	}
}

void Fluid::_finalize()
{
	if (file_writer_thread.joinable()) {
		file_writer_thread.join();
	}
	cudaStreamDestroy(fluid_stream);
	freeHostMemory();
	freeDeviceMemory();
}