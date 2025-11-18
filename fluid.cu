#include <cmath>
#include "fluid.h"
#include "fluid_kernels.cuh"
#include <memory.h>
#include <tuple>
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
	h_u = new float3[N];
	memset(h_u, 0, N * sizeof(float3));
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
	cudaStreamSynchronize(fluid_stream);
}

void Fluid::freeHostMemory()
{
	free(h_u);
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
	cudaFree(d_F);
	cudaFree(d_F_ibm);
	cudaFree(d_F_tot);
}

void Fluid::recordData(int time)
{
	if (time % 1 == 0) {
		ofstream fVelb;
		string  str_Velb;
		str_Velb = "Velb" + to_string(time) + ".dat";
		fVelb.open(str_Velb);
		int gi;
		for (int zi = 0; zi < fluidSize.z; zi++) {
			for (int yi = 0; yi < fluidSize.y; yi++) {
				for (int xi = 0; xi < fluidSize.x; xi++) {
					gi = idx3(xi, yi, zi, fluidSize.x, fluidSize.y);
					fVelb << setw(9) << to_string(h_u[gi].x) << "      " << setw(9) << to_string(h_u[gi].y) << "      " << setw(9) << to_string(h_u[gi].z) << "\n";
				}
			}
		}
		fVelb.close();
	}
}

void Fluid::writeFiles(double time)
{
	if (int(time * 1) % 1 == 0) {
		recordData(int(time * 1));
	}
}

Fluid::Fluid(int3 fluidSize, int time):
	fluidSize(fluidSize),
	d_f(0),
	d_fpost(0),
	d_fnext(0),
	d_rho(0),
	d_u(0),
	d_F(0),
	d_F_ibm(0),
	d_F_tot(0)
{
	dt = 1e-3f;
	N = fluidSize.x * fluidSize.y * fluidSize.z;
	Nd = N * 19;
	threads = 256;
	blocksN = (N + threads - 1) / threads;
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
	h_fp->L = fluidSize;
	h_fp->h = 0.5f;
	h_fp->dx = 1;
	h_fp->dy = 1;
	h_fp->dz = 1;
	h_fp->D = 1;
	niu = 1e-6f;
	dx_fluid = 40e-6 * h_fp->h;
	dt_fluid = (h_fp->tau - 0.5f) * dx_fluid * dx_fluid / (3 * niu);
	Nsub = int(dt / dt_fluid);
	h_fp->beta = 0.01;
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

cudaStream_t Fluid::stream() const
{
        return fluid_stream;
}

void Fluid::convectionAndDiffusion()
{
        k_convection_diffusion << <blocksN, threads, 0, fluid_stream >> > (d_u, d_c1, d_c2, d_fp);
        swap(d_c1, d_c2);
}

void Fluid::update(long long int solverIterations)
{
	double time = solverIterations * dt;
	k_set_force << <blocksN, threads, 0, fluid_stream >> > (d_F, d_fp);
	k_vec_add << <blocksN, threads, 0, fluid_stream >> > (d_F, d_F_ibm, d_F_tot, d_fp);
	k_macros << <blocksN, threads, 0, fluid_stream >> > (d_f, d_rho, d_u, d_F_tot, d_fp);
	k_collide << <blocksN, threads, 0, fluid_stream >> > (d_f, d_fpost, d_rho, d_u, d_F_tot, d_fp);
	k_stream_bounce << <blocksN, threads, 0, fluid_stream >> > (d_fpost, d_fnext, d_fp);
	swap(d_f, d_fnext);
	k_zero << <blocksN, threads, 0, fluid_stream >> > (d_F_ibm, d_fp);
	if (solverIterations % 1000 == 0) {
		if (file_writer_thread.joinable()) {
			file_writer_thread.join();
		}
		copyDataToHost();
		file_writer_thread = thread(mem_fn(&Fluid::writeFiles), this, time);
	}
}

void Fluid::_finalize()
{
	cudaStreamDestroy(fluid_stream);
	if (file_writer_thread.joinable()) {
		file_writer_thread.join();
	}
	freeHostMemory();
	freeDeviceMemory();
}