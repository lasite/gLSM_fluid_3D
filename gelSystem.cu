#include <assert.h>
#include <memory.h>
#include <cmath>
#include <tuple>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <sstream>
#include <thread>
#include <functional>
#include <cuda_runtime.h>

#include"gelSystem.h"
#include"gel_kernel.cu"


void GelSystem::allocateHostStorage()
{
	//chemical variables
	m_hum = new double[m_numGelElements];
	memset(m_hum, 0, m_numGelElements * sizeof(double));

	m_hvm = new double[m_numGelElements];
	memset(m_hvm, 0, m_numGelElements * sizeof(double));

	m_hwm = new double[m_numGelElements];
	memset(m_hwm, 0, m_numGelElements * sizeof(double));

	m_hfilament = new double3[1000 * m_params.maxFilamentlen];
	memset(m_hfilament, 0, 1000 * m_params.maxFilamentlen * sizeof(bool));

	//dynamics variables
	m_hrn = new double3[m_numGelNodes];
	memset(m_hrn, 0, m_numGelNodes * sizeof(double3));

	m_hrm = new double3[m_numGelElements];
	memset(m_hrm, 0, m_numGelElements * sizeof(double3));

	m_hFn = new double3[m_numGelNodes];
	memset(m_hFn, 0, m_numGelNodes * sizeof(double3));

	m_hVeln = new double3[m_numGelNodes];
	memset(m_hVeln, 0, m_numGelNodes * sizeof(double3));

	//
	m_hmap_element = new int[m_numGelElements];
	memset(m_hmap_element, 0, m_numGelElements * sizeof(int));

	m_hmap_node = new int[m_numGelNodes];
	memset(m_hmap_node, 0, m_numGelNodes * sizeof(int));

	m_hvm_center = new double[1000];
	memset(m_hvm_center, 0, 1000 * sizeof(double));

	m_hvm_center_z = new double[1000 * (m_gelSize.z + 1)];
	memset(m_hvm_center_z, 0, 1000 * (m_gelSize.z + 1) * sizeof(double));

	m_hwm_center = new double[1000];
	memset(m_hwm_center, 0, 1000 * sizeof(double));

	m_hrm_center = new double3[1000];
	memset(m_hrm_center, 0, 1000 * sizeof(double3));

	m_hFn_center = new double3[1000];
	memset(m_hFn_center, 0, 1000 * sizeof(double3));

	m_hVeln_center = new double3[1000];
	memset(m_hVeln_center, 0, 1000 * sizeof(double3));

	h_u = new float3[N];
	memset(h_u, 0, N * sizeof(float3));

	h_bIndex = new int[M];
	memset(h_bIndex, 0, M * sizeof(int));
}

void GelSystem::allocateDeviceStorage()
{
	//chemical variables
	cudaMalloc((void**)&m_dum, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_dum_norm, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_dun_norm, m_numGelNodes * sizeof(double));

	cudaMalloc((void**)&m_dvm, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_dvm_norm, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_dvn_norm, m_numGelNodes * sizeof(double));

	cudaMalloc((void**)&m_dwm, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_dwmp, m_numGelElements * sizeof(double));

	cudaMalloc((void**)&m_dT0m, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_dT1m, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_dT2m, m_numGelElements * sizeof(double));

	cudaMalloc((void**)&m_dfilament, 1000 * m_params.maxFilamentlen * sizeof(double3));
	//dynamics variables
	cudaMalloc((void**)&m_drn, m_numGelNodes * sizeof(double3));
	cudaMalloc((void**)&m_drm, m_numGelElements * sizeof(double3));

	cudaMalloc((void**)&m_dFn, m_numGelNodes * sizeof(double3));
	cudaMalloc((void**)&m_dVeln, m_numGelNodes * sizeof(double3));
	cudaMalloc((void**)&m_dVels, m_numGelNodes * sizeof(double3));

	cudaMalloc((void**)&m_drm_loc, 8 * m_numGelElements * sizeof(double3));
	cudaMalloc((void**)&m_dnmSm, 6 * m_numGelElements * sizeof(double3));

	cudaMalloc((void**)&m_dVolm, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_dPrem, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_dc0, m_numGelElements * sizeof(double));

	cudaMalloc((void**)&m_dmap_element, m_numGelElements * sizeof(int));
	cudaMalloc((void**)&m_dmap_node, m_numGelNodes * sizeof(int));

	cudaMalloc((void**)&m_dvm_center, 1000 * sizeof(double));
	cudaMalloc((void**)&m_dwm_center, 1000 * sizeof(double));
	cudaMalloc((void**)&m_drm_center, 1000 * sizeof(double3));
	cudaMalloc((void**)&m_dFn_center, 1000 * sizeof(double3));
	cudaMalloc((void**)&m_dVeln_center, 1000 * sizeof(double3));

	cudaMalloc((void**)&m_dtime, sizeof(double));
	cudaMalloc(&d_hitCnt, sizeof(unsigned int));

	cudaMalloc(&d_f, Nd * sizeof(float));
	cudaMalloc(&d_fpost, Nd * sizeof(float));
	cudaMalloc(&d_fnext, Nd * sizeof(float));
	cudaMalloc(&d_rho, N * sizeof(float));
	cudaMalloc(&d_u, N * sizeof(float3));
	cudaMalloc(&d_F, N * sizeof(float3));
	cudaMalloc(&d_F_ibm, N * sizeof(float3));
	cudaMalloc(&d_F_tot, N * sizeof(float3));

	cudaMalloc(&d_lag, M * sizeof(float3));
	cudaMalloc(&d_Ul, M * sizeof(float3));
	cudaMalloc(&d_Vl, M * sizeof(float3));
	cudaMalloc(&d_Fl, M * sizeof(float3));
	cudaMalloc(&d_bIndex, M * sizeof(int));
}

int GelSystem::get_index(int xi, int yi, int zi, int size)
{
	return xi + yi * (m_gelSize.x + size) + zi * (m_gelSize.y + size) * (m_gelSize.x + size);
}

void GelSystem::setChemicalWave(int wave_type)
{
	int LX = m_gelSize.x;
	int LY = m_gelSize.y;
	int LZ = m_gelSize.z;
	double uss_max = 0.4;
	double vss_max = 0.4;
	double cuty = 0.3;
	double band = 0.4;
	switch (wave_type) {
	case 0:
		break;
	case 1:
		for (int zi = 1; zi <= LZ; zi++) {
			for (int yi = 1; yi <= LY * (1 - cuty); yi++) {
				for (int xi = int(LX * (cuty - 0.5 * band)); xi <= int(LX * (cuty + 0.5 * band)); xi++) {
					int gi = get_index(xi, yi, zi, 1);
					m_hum[gi] = uss_max * (LX * (cuty + 0.5 * band) - xi) / (LX * band);
				}
			}
		}
		for (int zi = 1; zi <= LZ; zi++) {
			for (int yi = 1; yi <= LY * (1 - cuty); yi++) {
				for (int xi = int(LX * cuty); xi <= int(LX * (cuty + band)); xi++) {
					int gi = get_index(xi, yi, zi, 1);
					m_hvm[gi] = vss_max * (LX * (cuty + band) - xi) / (LX * band);
				}
			}
		}
		break;
	case 2:
		break;
	}
}

int GelSystem::idx3(int x, int y, int z, int Nx, int Ny)
{
	return x + y * Nx + z * Nx * Ny;
}

void GelSystem::buildBoundaryIndex()
{
	int LX = m_gelSize.x;
	int LY = m_gelSize.y;
	int LZ = m_gelSize.z;
	int count = 0;
	for (int k = 1; k < LZ + 1; ++k) {
		for (int j = 1; j < LY + 1; ++j) {
			for (int i = 1; i < LX + 1; ++i) {
				if (i == 1 || i == LX || j == 1 || j == LY || k == 1 || k == LZ) {
					h_bIndex[count] = idx3(i, j, k, LX + 2, LY + 2);
					count++;
				}
			}
		}
	}
}

void GelSystem::setInitValue()
{
	//Init chemical variables value
	//double phi = 0;
	bool random = false;
	for (int zi = 1; zi < m_gelSize.z; zi++) {
		for (int yi = 1; yi < m_gelSize.y; yi++) {
			for (int xi = 1; xi < m_gelSize.x; xi++) {
				int gi = get_index(xi, yi, zi, 1);
				if (random) {
					m_hum[gi] = m_params.uss * (1 + 0.5 * (2 * double(rand()) / RAND_MAX - 1));
					m_hvm[gi] = m_params.vss * (1 + 0.5 * (2 * double(rand()) / RAND_MAX - 1));
					m_hwm[gi] = m_params.wss;
				}
				else {
					m_hum[gi] = m_params.uss;
					m_hvm[gi] = m_params.vss;
					m_hwm[gi] = m_params.wss;
				}
			}
		}
	}
	setChemicalWave(0);
	//Init gel coords
	double lam = pow(m_params.FA0 / m_params.wss, 1 / 3.0);
	double x_offset = ((double)fluidSize.x - 1) * fp.h / 2;
	double y_offset = ((double)fluidSize.y - 1) * fp.h / 2;
	double z_offset = ((double)fluidSize.z - 1) * fp.h / 2;
	for (int zi = 0; zi < m_gelSize.z + 2; zi++) {
		for (int yi = 0; yi < m_gelSize.y + 2; yi++) {
			for (int xi = 0; xi < m_gelSize.x + 2; xi++) {
				int hi = get_index(xi, yi, zi, 2);
				m_hrn[hi].x = x_offset + (2 * double(xi) - double(m_gelSize.x) - 1) * m_params.dx * lam / 2;
				m_hrn[hi].y = y_offset + (2 * double(yi) - double(m_gelSize.y) - 1) * m_params.dy * lam / 2;
				m_hrn[hi].z = z_offset + (2 * double(zi) - double(m_gelSize.z) - 1) * m_params.dz * lam / 2;
			}
		}
	}
	//set node type
	setType(m_hmap_node, 2);
	setType(m_hmap_element, 1);
	
	k_init << <blocksN, threads, 0, fluid_stream >> > (d_f, d_rho, d_u);
	buildBoundaryIndex();
}

void GelSystem::setGoonValue(int time)
{
	ifstream frn, fum, fvm, fwm;
	string str_rn, str_um, str_vm, str_wm;
	str_rn = "../rn" + to_string(time) + ".dat";
	str_um = "../um" + to_string(time) + ".dat";
	str_vm = "../vm" + to_string(time) + ".dat";
	str_wm = "../wm" + to_string(time) + ".dat";
	//str_rn = "rn" + to_string(time) + ".dat";
	//str_um = "um" + to_string(time) + ".dat";
	//str_vm = "vm" + to_string(time) + ".dat";
	//str_wm = "wm" + to_string(time) + ".dat";
	frn.open(str_rn, ios::in);
	fvm.open(str_vm, ios::in);
	fum.open(str_um, ios::in);
	fwm.open(str_wm, ios::in);
	string line;
	for (int zi = 1; zi < m_gelSize.z; zi++) {
		for (int yi = 1; yi < m_gelSize.y; yi++) {
			for (int xi = 1; xi < m_gelSize.x; xi++) {
				int gi = get_index(xi, yi, zi, 1);
				double um, vm, wm;
				getline(fum, line);
				istringstream iss1(line);
				iss1 >> um;
				m_hum[gi] = um;

				getline(fvm, line);
				istringstream iss2(line);
				iss2 >> vm;
				m_hvm[gi] = vm;

				getline(fwm, line);
				istringstream iss3(line);
				iss3 >> wm;
				m_hwm[gi] = wm;
			}
		}
	}

	//for (int zi = 0; zi < m_gelSize.z + 2; zi++) {
	//	for (int yi = 0; yi < m_gelSize.y + 2; yi++) {
	//		for (int xi = 0; xi < m_gelSize.x + 2; xi++) {
	for (int zi = 1; zi < m_gelSize.z + 1; zi++) {
		for (int yi = 1; yi < m_gelSize.y + 1; yi++) {
			for (int xi = 1; xi < m_gelSize.x + 1; xi++) {
				int gi = get_index(xi, yi, zi, 2);
				double rnx, rny, rnz;
				getline(frn, line);
				istringstream iss4(line);
				iss4 >> rnx >> rny >> rnz;
				m_hrn[gi] = make_double3(rnx, rny, rnz);
			}
		}
	}

	frn.close();
	fvm.close();
	fum.close();
	fwm.close();

	//set node type
	setType(m_hmap_node, 2);
	setType(m_hmap_element, 1);
}

void GelSystem::setType(int* a, int size)
{
	int xi, yi, zi;
	int LX = m_gelSize.x + size - 1;
	int LY = m_gelSize.y + size - 1;
	int LZ = m_gelSize.z + size - 1;
	for (xi = 1; xi < LX; xi++) {
		for (yi = 1; yi < LY; yi++) {
			for (zi = 1; zi < LZ; zi++) {
				a[get_index(xi, yi, zi, size)] = 0;
			}
		}
	}

	for (yi = 1; yi < LY; yi++) {
		for (zi = 1; zi < LZ; zi++) {
			a[get_index(0, yi, zi, size)] = 1;
			a[get_index(LX, yi, zi, size)] = 2;
		}
	}
	for (xi = 1; xi < LX; xi++) {
		for (zi = 1; zi < LZ; zi++) {
			a[get_index(xi, 0, zi, size)] = 3;
			a[get_index(xi, LY, zi, size)] = 4;
		}
	}
	for (xi = 1; xi < LX; xi++) {
		for (yi = 1; yi < LY; yi++) {
			a[get_index(xi, yi, 0, size)] = 5;
			a[get_index(xi, yi, LZ, size)] = 6;
		}
	}

	for (xi = 1; xi < LX; xi++) {
		a[get_index(xi, 0, 0, size)] = 7;
		a[get_index(xi, 0, LZ, size)] = 8;
		a[get_index(xi, LY, 0, size)] = 9;
		a[get_index(xi, LY, LZ, size)] = 10;
	}
	for (yi = 1; yi < LY; yi++) {
		a[get_index(0, yi, 0, size)] = 11;
		a[get_index(0, yi, LZ, size)] = 12;
		a[get_index(LX, yi, 0, size)] = 13;
		a[get_index(LX, yi, LZ, size)] = 14;
	}
	for (zi = 1; zi < LZ; zi++) {
		a[get_index(0, 0, zi, size)] = 15;
		a[get_index(0, LY, zi, size)] = 16;
		a[get_index(LX, 0, zi, size)] = 17;
		a[get_index(LX, LY, zi, size)] = 18;
	}

	a[get_index(0, 0, 0, size)] = 19;
	a[get_index(0, 0, LZ, size)] = 20;
	a[get_index(0, LY, 0, size)] = 21;
	a[get_index(0, LY, LZ, size)] = 22;
	a[get_index(LX, 0, 0, size)] = 23;
	a[get_index(LX, 0, LZ, size)] = 24;
	a[get_index(LX, LY, 0, size)] = 25;
	a[get_index(LX, LY, LZ, size)] = 26;
}

void GelSystem::copyDataToDevice()
{
	cudaMemcpy(m_dum, m_hum, sizeof(double) * m_numGelElements, cudaMemcpyHostToDevice);
	cudaMemcpy(m_dvm, m_hvm, sizeof(double) * m_numGelElements, cudaMemcpyHostToDevice);
	cudaMemcpy(m_dwm, m_hwm, sizeof(double) * m_numGelElements, cudaMemcpyHostToDevice);
	cudaMemcpy(m_drn, m_hrn, sizeof(double3) * m_numGelNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(m_dmap_element, m_hmap_element, sizeof(int) * m_numGelElements, cudaMemcpyHostToDevice);
	cudaMemcpy(m_dmap_node, m_hmap_node, sizeof(int) * m_numGelNodes, cudaMemcpyHostToDevice);

	cudaMemcpy(d_bIndex, h_bIndex, sizeof(int) * M, cudaMemcpyHostToDevice);
}

void GelSystem::copyDataToHost()
{
	cudaMemcpyAsync(m_hum, m_dum, sizeof(double) * m_numGelElements, cudaMemcpyDeviceToHost, m_gel_stream);
	cudaMemcpyAsync(m_hvm, m_dvm, sizeof(double) * m_numGelElements, cudaMemcpyDeviceToHost, m_gel_stream);
	cudaMemcpyAsync(m_hwm, m_dwm, sizeof(double) * m_numGelElements, cudaMemcpyDeviceToHost, m_gel_stream);
	cudaMemcpyAsync(m_hfilament, m_dfilament, sizeof(double3) * 1000 * m_params.maxFilamentlen, cudaMemcpyDeviceToHost, m_gel_stream);
	cudaMemcpyAsync(m_hrm, m_drm, sizeof(double3) * m_numGelElements, cudaMemcpyDeviceToHost, m_gel_stream);
	cudaMemcpyAsync(m_hrn, m_drn, sizeof(double3) * m_numGelNodes, cudaMemcpyDeviceToHost, m_gel_stream);
	cudaMemcpyAsync(m_hFn, m_dFn, sizeof(double3) * m_numGelNodes, cudaMemcpyDeviceToHost, m_gel_stream);
	cudaMemcpyAsync(m_hVeln, m_dVeln, sizeof(double3) * m_numGelNodes, cudaMemcpyDeviceToHost, m_gel_stream);
	cudaMemcpyAsync(m_hvm_center, m_dvm_center, sizeof(double) * 1000, cudaMemcpyDeviceToHost, m_gel_stream);
	cudaMemcpyAsync(m_hwm_center, m_dwm_center, sizeof(double) * 1000, cudaMemcpyDeviceToHost, m_gel_stream);
	cudaMemcpyAsync(m_hrm_center, m_drm_center, sizeof(double3) * 1000, cudaMemcpyDeviceToHost, m_gel_stream);
	cudaMemcpyAsync(m_hFn_center, m_dFn_center, sizeof(double3) * 1000, cudaMemcpyDeviceToHost, m_gel_stream);
	cudaMemcpyAsync(m_hVeln_center, m_dVeln_center, sizeof(double3) * 1000, cudaMemcpyDeviceToHost, m_gel_stream);
	cudaStreamSynchronize(m_gel_stream);

	cudaMemcpyAsync(h_u, d_u, sizeof(float3) * N, cudaMemcpyDeviceToHost, fluid_stream);
	cudaStreamSynchronize(fluid_stream);
}

void GelSystem::freeHostMemory()
{
	delete[] m_hum;
	delete[] m_hvm;
	delete[] m_hwm;
	delete[] m_hfilament;
	delete[] m_hrn;
	delete[] m_hrm;
	delete[] m_hFn;
	delete[] m_hVeln;
	delete[] m_hmap_element;
	delete[] m_hmap_node;

	free(h_u);
	free(h_bIndex);
}

void GelSystem::freeDeviceMemory()
{

	cudaFree(m_dum);
	cudaFree(m_dum_norm);
	cudaFree(m_dun_norm);
	cudaFree(m_dvm);
	cudaFree(m_dvm_norm);
	cudaFree(m_dvn_norm);
	cudaFree(m_dwm);
	cudaFree(m_dwmp);
	cudaFree(m_dfilament);
	cudaFree(m_dT0m);
	cudaFree(m_dT1m);
	cudaFree(m_dT2m);
	cudaFree(m_drn);
	cudaFree(m_drm);
	cudaFree(m_dFn);
	cudaFree(m_dVeln);
	cudaFree(m_dVels);
	cudaFree(m_drm_loc);
	cudaFree(m_dnmSm);
	cudaFree(m_dVolm);
	cudaFree(m_dPrem);
	cudaFree(m_dmap_element);
	cudaFree(m_dmap_node);

	cudaFree(d_f);
	cudaFree(d_fpost);
	cudaFree(d_fnext);
	cudaFree(d_rho);
	cudaFree(d_u);
	cudaFree(d_F);
	cudaFree(d_F_ibm);
	cudaFree(d_F_tot);
	cudaFree(d_lag);
	cudaFree(d_Ul);
	cudaFree(d_Fl);
}

void GelSystem::steadyStateValue(double& um, double& vm, double& wm, double phi)
{
	wm = m_params.FA0;
	double vsm, usm, usa;
	double vsa = 1.0;
	double vsb = 0.0;
	while (abs(vsa - vsb) > 1e-10)
	{
		vsm = 0.5 * (vsa + vsb);
		usm = vsm / (1 - wm);
		usa = vsa / (1 - wm);
		if (fu_h(usm, vsm, wm, phi) * fu_h(usa, vsa, wm, phi) < 0) {
			vsb = vsm;
		}
		else
			vsa = vsm;
	}
	um = vsb / (1.0 - wm);
	vm = vsb;
}

double GelSystem::fu_h(double u, double v, double w, double phi)
{
	double ww = (1.0 - w) * (1.0 - w);
	return ww * u - u * u - (1.0 - w) * m_params.f * v * (u - m_params.q * ww) / (u + m_params.q * ww) + phi * m_params.P2;
	//return ww * u - u * u - (1.0 - w) * (m_params.f * v + phi) * (u - m_params.q * ww) / (u + m_params.q * ww);
}

void GelSystem::recordCenterElement(double time)
{
	int LX = m_gelSize.x;
	int LY = m_gelSize.y;
	int LZ = m_gelSize.z;
	ofstream fbodycenter;
	ofstream ffilament;
	string str_filament;
	fbodycenter.open("bodycenter" + to_string(m_count) + ".dat", ios::app);

	for (int i = 0; i < 1000; i++) {
		fbodycenter
			<< setw(9) << to_string(time - 1000 + i * 1) << "      "
			<< setw(9) << to_string(m_hrm_center[i].x) << "      "
			<< setw(9) << to_string(m_hrm_center[i].y) << "      "
			<< setw(9) << to_string(m_hrm_center[i].z) << "      "
			<< setw(9) << to_string(m_hFn_center[i].x) << "      "
			<< setw(9) << to_string(m_hFn_center[i].y) << "      "
			<< setw(9) << to_string(m_hFn_center[i].z) << "      "
			<< setw(9) << to_string(m_hVeln_center[i].x) << "      "
			<< setw(9) << to_string(m_hVeln_center[i].y) << "      "
			<< setw(9) << to_string(m_hVeln_center[i].z) << "      "
			<< setw(9) << to_string(m_hvm_center[i]) << "      "
			<< setw(9) << to_string(m_hwm_center[i]) << "\n";

		//str_filament = "filament" + to_string(int(time) - 1000 + i * 1) + ".dat";
		//ffilament.open(str_filament);
		//for (int j = 0; j < m_params.maxFilamentlen; j++) {
		//	int gi = m_params.maxFilamentlen * i + j;
		//	if (m_hfilament[gi].x != 0) {
		//		ffilament << setw(9) << to_string(m_hfilament[gi].x) << "      " << setw(9) << to_string(m_hfilament[gi].y) << "      " << setw(9) << to_string(m_hfilament[gi].z) << "\n";
		//	}
		//}
		//ffilament.close();
	}
	fbodycenter.close();
}

void GelSystem::recordData(int time)
{
	if (time % 1 == 0) {
		ofstream frn, frm, fum, fvm, fwm, fFn, fVeln;
		string str_rn, str_rm, str_um, str_vm, str_wm, str_Fn, str_Veln;
		str_rn = "rn" + to_string(time) + ".dat";
		str_rm = "rm" + to_string(time) + ".dat";
		str_um = "um" + to_string(time) + ".dat";
		str_vm = "vm" + to_string(time) + ".dat";
		str_wm = "wm" + to_string(time) + ".dat";
		str_Fn = "Fn" + to_string(time) + ".dat";
		str_Veln = "Veln" + to_string(time) + ".dat";
		frn.open(str_rn);
		frm.open(str_rm);
		fvm.open(str_vm);
		fum.open(str_um);
		fwm.open(str_wm);
		fFn.open(str_Fn);
		fVeln.open(str_Veln);
		int gi;
		for (int zi = 1; zi < m_gelSize.z + 1; zi++) {
			for (int yi = 1; yi < m_gelSize.y + 1; yi++) {
				for (int xi = 1; xi < m_gelSize.x + 1; xi++) {
					gi = get_index(xi, yi, zi, 2);
					frn << setw(9) << to_string(m_hrn[gi].x) << "      " << setw(9) << to_string(m_hrn[gi].y) << "      " << setw(9) << to_string(m_hrn[gi].z) << "\n";
					fFn << setw(9) << to_string(m_hFn[gi].x) << "      " << setw(9) << to_string(m_hFn[gi].y) << "      " << setw(9) << to_string(m_hFn[gi].z) << "\n";
					fVeln << setw(9) << to_string(m_hVeln[gi].x) << "      " << setw(9) << to_string(m_hVeln[gi].y) << "      " << setw(9) << to_string(m_hVeln[gi].z) << "\n";
				}
			}
		}
		for (int zi = 1; zi < m_gelSize.z; zi++) {
			for (int yi = 1; yi < m_gelSize.y; yi++) {
				for (int xi = 1; xi < m_gelSize.x; xi++) {
					gi = get_index(xi, yi, zi, 1);
					frm << setw(9) << to_string(m_hrm[gi].x) << "      " << setw(9) << to_string(m_hrm[gi].y) << "      " << setw(9) << to_string(m_hrm[gi].z) << "\n";
					fum << setw(9) << to_string(m_hum[gi]) << "\n";
					fvm << setw(9) << m_hvm[gi] << "\n";
					fwm << setw(9) << m_hwm[gi] << "\n";
				}
			}
		}
		frn.close();
		frm.close();
		fum.close();
		fvm.close();
		fwm.close();
		fFn.close();
		fVeln.close();
	}

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

void GelSystem::writeFiles(double time)
{
	if (int(time * 1) % 1 == 0) {
		recordData(int(time * 1));
	}
	/*recordSpiralWave(time);*/

	if (flag && int(time) % 1000 == 0) {
		recordCenterElement(time);
	}
	else {
		flag = true;
	}
}

GelSystem::GelSystem(int3 gelSize, int3 FluidSize, int time, int i, int j, int k) :
	m_bInitialized(false),
	m_gelSize(gelSize),
	fluidSize(FluidSize),
	//CPU data
	m_hum(0),
	m_hvm(0),
	m_hwm(0),
	m_hfilament(0),
	m_hrn(0),
	m_hrm(0),
	m_hFn(0),
	m_hVeln(0),
	m_hvm_center(0),
	m_hvm_center_z(0),
	m_hwm_center(0),
	m_hrm_center(0),
	m_hFn_center(0),
	m_hVeln_center(0),
	m_hmap_element(0),
	m_hmap_node(0),
	//GPU data
	m_dum(0),
	m_dum_norm(0),
	m_dun_norm(0),
	m_dvm(0),
	m_dvm_norm(0),
	m_dvn_norm(0),
	m_dwm(0),
	m_dwmp(0),
	m_dT0m(0),
	m_dT1m(0),
	m_dT2m(0),
	m_drn(0),
	m_drm(0),
	m_dFn(0),
	m_dVeln(0),
	m_dVels(0),
	m_drm_loc(0),
	m_dnmSm(0),
	m_dVolm(0),
	m_dPrem(0),
	m_dvm_center(0),
	m_dwm_center(0),
	m_drm_center(0),
	m_dFn_center(0),
	m_dVeln_center(0),
	m_dmap_element(0),
	m_dmap_node(0),
	d_f(0),
	d_fpost(0),
	d_fnext(0),
	d_rho(0),
	d_u(0),
	d_F(0),
	d_F_ibm(0),
	d_F_tot(0),
	d_lag(0),
	d_Ul(0),
	d_Vl(0),
	d_Fl(0),
	d_bIndex(0),
	d_A(0)
{
	m_dt = 1e-3;
	m_df = int(1 / m_dt);
	m_numGelElements = (gelSize.x + 1) * (gelSize.y + 1) * (gelSize.z + 1);
	m_numGelNodes = (gelSize.x + 2) * (gelSize.y + 2) * (gelSize.z + 2);

	m_blockDim = dim3(8, 8, 8);

	m_gridDim_1.x = (m_gelSize.x - 1 + m_blockDim.x - 1) / m_blockDim.x;
	m_gridDim_1.y = (m_gelSize.y - 1 + m_blockDim.y - 1) / m_blockDim.y;
	m_gridDim_1.z = (m_gelSize.z - 1 + m_blockDim.z - 1) / m_blockDim.z;

	m_gridDim0.x = (m_gelSize.x + m_blockDim.x - 1) / m_blockDim.x;
	m_gridDim0.y = (m_gelSize.y + m_blockDim.y - 1) / m_blockDim.y;
	m_gridDim0.z = (m_gelSize.z + m_blockDim.z - 1) / m_blockDim.z;

	m_gridDim1.x = (m_gelSize.x + 1 + m_blockDim.x - 1) / m_blockDim.x;
	m_gridDim1.y = (m_gelSize.y + 1 + m_blockDim.y - 1) / m_blockDim.y;
	m_gridDim1.z = (m_gelSize.z + 1 + m_blockDim.z - 1) / m_blockDim.z;

	m_gridDim2.x = (m_gelSize.x + 2 + m_blockDim.x - 1) / m_blockDim.x;
	m_gridDim2.y = (m_gelSize.y + 2 + m_blockDim.y - 1) / m_blockDim.y;
	m_gridDim2.z = (m_gelSize.z + 2 + m_blockDim.z - 1) / m_blockDim.z;

	m_params.LX = m_gelSize.x;
	m_params.LY = m_gelSize.y;
	m_params.LZ = m_gelSize.z;
	m_params.I = 0.01 * k;
	m_params.f = 0.1 * i;
	m_params.ep = 0.05 * j;
	m_params.q = 1e-4;
	m_params.P1 = 0.0124;
	m_params.P2 = 0.77;
	m_params.dt = m_dt;
	m_params.dtx = 5 * m_dt;
	m_params.dx = 1.0;
	m_params.dy = 1.0;
	m_params.dz = 1.0;
	m_params.CH0 = 0.338;
	m_params.CH1 = 0.518;
	m_params.CHS = 0.1;
	m_params.C0 = 1.3e-3;
	m_params.AZ0 = 100.0;
	m_params.FA0 = 0.139;
	m_params.maxFilamentlen = 300;
	steadyStateValue(m_params.uss, m_params.vss, m_params.wss, 0);
	int LX_ = m_gelSize.x - 1;
	int LY_ = m_gelSize.y - 1;
	int LZ_ = m_gelSize.z - 1;
	int3 offset[27] = {
	{ 0, 0, 0 },
	{ 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 },
	{ 0, 1, 1 }, { 0, 1, -1 }, { 0, -1, 1 }, { 0, -1, -1 }, { 1, 0, 1 }, { 1, 0, -1 }, { -1, 0, 1 }, { -1, 0, -1 }, { 1, 1, 0 }, { 1, -1, 0 }, { -1, 1, 0 }, { -1, -1, 0 },
	{ 1, 1, 1 }, { 1, 1, -1 }, { 1, -1, 1 }, { 1, -1, -1 }, { -1, 1, 1 }, { -1, 1, -1 }, { -1, -1, 1 }, { -1, -1, -1 } };
	memcpy(m_params.rn_offset, offset, sizeof(offset));
	memcpy(m_params.um_offset_noflux, offset, sizeof(offset));

	N = fluidSize.x * fluidSize.y * fluidSize.z;
	Nd = N * 19;
	M = gelSize.x * gelSize.y * gelSize.z - (gelSize.x - 2) * (gelSize.y - 2) * (gelSize.z - 2);
	threads = 256;
	blocksN = (N + threads - 1) / threads;
	blocksM = (M + threads - 1) / threads;
	fp.cs2 = 1.f / 3.f;
	fp.tau = 0.8f;
	fp.nu = fp.cs2 * (fp.tau - 0.5f);
	fp.N = N;
	fp.M = M;
	fp.L = fluidSize;
	fp.h = 0.5f;
	niu = 1e-6f;
	dx_fluid = 40e-6 * fp.h;
	dt_fluid = (fp.tau - 0.5) * dx_fluid * dx_fluid / (3 * niu);
	Nsub = int(m_dt / dt_fluid);
	fp.beta = 0.01;
	int3 c[19] = {
	{0,0,0},
	{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
	{1,1,0},{-1,1,0},{-1,-1,0},{1,-1,0},
	{1,0,1},{-1,0,1},{-1,0,-1},{1,0,-1},
	{0,1,1},{0,1,-1},{0,-1,-1},{0,-1,1}
	};
	float w[19] = {
	1.0 / 3.0,
	1.0 / 18.0,1.0 / 18.0,1.0 / 18.0,1.0 / 18.0,1.0 / 18.0,1.0 / 18.0,
	1.0 / 36.0,1.0 / 36.0,1.0 / 36.0,1.0 / 36.0,1.0 / 36.0,1.0 / 36.0,
	1.0 / 36.0,1.0 / 36.0,1.0 / 36.0,1.0 / 36.0,1.0 / 36.0,1.0 / 36.0
	};
	int opp[19] = { 0,2,1, 4,3, 6,5, 9,10,7,8, 13,14,11,12, 17,18,15,16 };
	memcpy(fp.c, c, sizeof(c));
	memcpy(fp.w, w, sizeof(w));
	memcpy(fp.opp, opp, sizeof(opp));
	fp.F_const = make_float3(0, 0, 0);
	_initialize(time);
}

GelSystem::~GelSystem()
{
	_finalize();
}

void GelSystem::_initialize(int time)
{
	assert(!m_bInitialized);

	allocateHostStorage();
	allocateDeviceStorage();
	cudaMemcpyToSymbol(params, &m_params, sizeof(GelParams));
	cudaMemcpyToSymbol(p, &fp, sizeof(FluidParams));
	cudaStreamCreate(&m_gel_stream);
	cudaStreamCreate(&fluid_stream);
	if (false)
		setGoonValue(time);
	else
		setInitValue();
	copyDataToDevice();
	m_bInitialized = true;
}

void GelSystem::update(long long int solverIterations)
{
	assert(m_bInitialized);
	double time = solverIterations * m_dt;
	float ramp = fmin(1, (solverIterations + 1) / 20000.0);
	float beta_eff = fp.beta * ramp;
	if (solverIterations % 5 == 0) {
		calServiceNodesPositionD << < m_gridDim2, m_blockDim, 0, m_gel_stream >> > (m_drn, m_dmap_node);
		calElementPropertiesD << <m_gridDim1, m_blockDim, 0, m_gel_stream >> > (m_drn, m_drm, m_drm_loc, m_dnmSm, m_dVolm, m_dwm, m_dwmp);
		calPressureD << < m_gridDim_1, m_blockDim, 0, m_gel_stream >> > (m_dPrem, m_dvm, m_dwm);
		k_add_reaction_to_gel << <blocksM, threads, 0, fluid_stream >> > (d_bIndex, m_dFn, d_Fl);
		calNodesVelocityD << < m_gridDim0, m_blockDim, 0, m_gel_stream >> > (m_drn, m_dVeln, m_dVels, m_dFn, m_dnmSm, m_dPrem, m_dwm);
		calInternalNodesPositionD << < m_gridDim0, m_blockDim, 0, m_gel_stream >> > (m_drn, m_dVeln);
		calChemBoundaryD << < m_gridDim1, m_blockDim, 0, m_gel_stream >> > (m_dum, m_dum_norm, m_dvm, m_dvm_norm, m_dwm, m_dmap_element, time);
		calUnnormD << < m_gridDim0, m_blockDim, 0, m_gel_stream >> > (m_dun_norm, m_dum_norm, m_dvn_norm, m_dvm_norm);
		calTermsD << < m_gridDim_1, m_blockDim, 0, m_gel_stream >> > (m_dT0m, m_dT1m, m_dT2m, m_dwm, m_dwmp, m_dVeln, m_dnmSm, m_dVolm, m_drm_loc, m_dun_norm, m_dum_norm, m_drm);
	}
	calChemD << < m_gridDim_1, m_blockDim, 0, m_gel_stream >> > (m_dvm, m_dum, m_dwm, m_dT0m, m_dT1m, m_dT2m, m_drm, time);
	for (int kk = 0; kk < Nsub; kk++) {
		k_set_force << <blocksN, threads, 0, fluid_stream >> > (d_F);
		k_gather_boundary << <blocksM, threads, 0, fluid_stream >> > (d_bIndex, m_drn, m_dVels, d_lag, d_Vl);
		k_ibm_interpolate << <blocksM, threads, 0, fluid_stream >> > (d_u, d_Ul, d_lag);
		k_scale_negbeta << <blocksM, threads, 0, fluid_stream >> > (d_Ul, d_Vl, d_Fl, beta_eff);
		k_zero << <blocksN, threads, 0, fluid_stream >> > (d_F_ibm);
		k_ibm_spread << <blocksM, threads, 0, fluid_stream >> > (d_Fl, d_lag, d_F_ibm, d_A);
		k_vec_add << <blocksN, threads, 0, fluid_stream >> > (d_F, d_F_ibm, d_F_tot);
		k_macros << <blocksN, threads, 0, fluid_stream >> > (d_f, d_rho, d_u, d_F_tot);
		k_collide << <blocksN, threads, 0, fluid_stream >> > (d_f, d_fpost, d_rho, d_u, d_F_tot);
		k_stream_bounce << <blocksN, threads, 0, fluid_stream >> > (d_fpost, d_fnext);
		std::swap(d_f, d_fnext);
	}
	if (solverIterations % 1000 == 0) {
		if (m_file_writer_thread.joinable()) {
			m_file_writer_thread.join();
		}
		copyDataToHost();
		m_file_writer_thread = thread(mem_fn(&GelSystem::writeFiles), this, time);
	}

	if (solverIterations % 1000 == 0) {
		recordCenterElementD << < dim3(1, 1, 1), dim3(1, 1, 1), 0, m_gel_stream >> > (m_dvm_center, m_dwm_center, m_drm_center, m_dFn_center, m_dVeln_center, m_dvm, m_dwm, m_drn, m_dFn, m_dVeln, (solverIterations / 1000) % 1000);
		cudaMemset(d_hitCnt, 0, sizeof(unsigned int));
		calFilamentD << <m_gridDim_1, m_blockDim, 0, m_gel_stream >> > (m_dvn_norm, m_dun_norm, m_dfilament, (solverIterations / 1000) % 1000, d_hitCnt);
	}
}

void GelSystem::_finalize()
{
	assert(m_bInitialized);
	cudaStreamDestroy(m_gel_stream);
	cudaStreamDestroy(fluid_stream);
	if (m_file_writer_thread.joinable()) {
		m_file_writer_thread.join();
	}
	freeHostMemory();
	freeDeviceMemory();
}