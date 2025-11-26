#include <cmath>
#include <tuple>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <functional>
#include <cuda_runtime.h>

#include "gel_kernels.cuh"
#include <memory.h>
using namespace std;
void Gel::allocateHostStorage()
{
	//chemical variables
	m_hum = new double[m_numGelElements];
	memset(m_hum, 0, m_numGelElements * sizeof(double));

	m_hvm = new double[m_numGelElements];
	memset(m_hvm, 0, m_numGelElements * sizeof(double));

	m_hwm = new double[m_numGelElements];
	memset(m_hwm, 0, m_numGelElements * sizeof(double));

	//dynamics variables
	m_hrn = new double3[m_numGelNodes];
	memset(m_hrn, 0, m_numGelNodes * sizeof(double3));

	m_hrm = new double3[m_numGelElements];
	memset(m_hrm, 0, m_numGelElements * sizeof(double3));

	m_hFn = new double3[m_numGelNodes];
	memset(m_hFn, 0, m_numGelNodes * sizeof(double3));

	m_hVeln = new double3[m_numGelNodes];
	memset(m_hVeln, 0, m_numGelNodes * sizeof(double3));

	m_hmap_element = new int[m_numGelElements];
	memset(m_hmap_element, 0, m_numGelElements * sizeof(int));

	m_hmap_node = new int[m_numGelNodes];
	memset(m_hmap_node, 0, m_numGelNodes * sizeof(int));

	m_hbIndex = new int[m_boundaryCount];
	memset(m_hbIndex, 0, m_boundaryCount * sizeof(int));
}

void Gel::allocateDeviceStorage()
{
	cudaMalloc((void**)&m_dgp, sizeof(GelParams));
	//chemical variables
	cudaMalloc((void**)&m_dum, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_dum_norm, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_dun_norm, m_numGelNodes * sizeof(double));
	cudaMalloc((void**)&m_dun_robin, m_numGelNodes * sizeof(double));

	cudaMalloc((void**)&m_dvm, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_dvm_norm, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_dvn_norm, m_numGelNodes * sizeof(double));

	cudaMalloc((void**)&m_dwm, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_dwmp, m_numGelElements * sizeof(double));

	cudaMalloc((void**)&m_dT0m, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_dT1m, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_dT2m, m_numGelElements * sizeof(double));

	//dynamics variables
	cudaMalloc((void**)&m_drn, m_numGelNodes * sizeof(double3));
	cudaMalloc((void**)&m_drm, m_numGelElements * sizeof(double3));

	cudaMalloc((void**)&m_dFn, m_numGelNodes * sizeof(double3));
	cudaMalloc((void**)&m_dFn_robin, m_numGelNodes * sizeof(double3));
	cudaMalloc((void**)&m_dVeln, m_numGelNodes * sizeof(double3));
	cudaMalloc((void**)&m_dVels, m_numGelNodes * sizeof(double3));

	cudaMalloc((void**)&m_drm_loc, 8 * m_numGelElements * sizeof(double3));
	cudaMalloc((void**)&m_dnmSm, 6 * m_numGelElements * sizeof(double3));

	cudaMalloc((void**)&m_dVolm, m_numGelElements * sizeof(double));
	cudaMalloc((void**)&m_dPrem, m_numGelElements * sizeof(double));

	cudaMalloc((void**)&m_dmap_element, m_numGelElements * sizeof(int));
	cudaMalloc((void**)&m_dmap_node, m_numGelNodes * sizeof(int));

	cudaMalloc(&m_dbIndex, m_boundaryCount * sizeof(int));
}

int Gel::get_index(int xi, int yi, int zi, int size)
{
	return xi + yi * (m_gelNodeGrid.x + size) + zi * (m_gelNodeGrid.y + size) * (m_gelNodeGrid.x + size);
}

void Gel::setChemicalWave(int wave_type)
{
	int LX = m_gelNodeGrid.x;
	int LY = m_gelNodeGrid.y;
	int LZ = m_gelNodeGrid.z;
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

int Gel::idx3(int x, int y, int z, int Nx, int Ny)
{
	return x + y * Nx + z * Nx * Ny;
}

void Gel::buildBoundaryIndex()
{
	int LX = m_gelNodeGrid.x;
	int LY = m_gelNodeGrid.y;
	int LZ = m_gelNodeGrid.z;
	int count = 0;
	for (int k = 1; k < LZ + 1; ++k) {
		for (int j = 1; j < LY + 1; ++j) {
			for (int i = 1; i < LX + 1; ++i) {
				if (i == 1 || i == LX || j == 1 || j == LY || k == 1 || k == LZ) {
					m_hbIndex[count] = idx3(i, j, k, LX + 2, LY + 2);
					count++;
				}
			}
		}
	}
}

void Gel::setInitValue()
{
	//Init chemical variables value
	//double phi = 0;
	bool random = false;
	for (int zi = 1; zi < m_gelNodeGrid.z; zi++) {
		for (int yi = 1; yi < m_gelNodeGrid.y; yi++) {
			for (int xi = 1; xi < m_gelNodeGrid.x; xi++) {
				int gi = get_index(xi, yi, zi, 1);
				if (random) {
					m_hum[gi] = m_hgp->uss * (1 + 0.5 * (2 * double(rand()) / RAND_MAX - 1));
					m_hvm[gi] = m_hgp->vss * (1 + 0.5 * (2 * double(rand()) / RAND_MAX - 1));
					m_hwm[gi] = m_hgp->wss;
				}
				else {
					m_hum[gi] = m_hgp->uss;
					m_hvm[gi] = m_hgp->vss;
					m_hwm[gi] = m_hgp->wss;
				}
			}
		}
	}

	setChemicalWave(0);
	//Init gel coords
	double lam = pow(m_hgp->FA0 / m_hgp->wss, 1 / 3.0);
	for (int zi = 0; zi < m_gelNodeGrid.z + 2; zi++) {
		for (int yi = 0; yi < m_gelNodeGrid.y + 2; yi++) {
			for (int xi = 0; xi < m_gelNodeGrid.x + 2; xi++) {
				int hi = get_index(xi, yi, zi, 2);
				m_hrn[hi].x = m_gelPosition.x + (2 * double(xi) - double(m_gelNodeGrid.x) - 1) * m_hgp->dx * lam / 2;
				m_hrn[hi].y = m_gelPosition.y + (2 * double(yi) - double(m_gelNodeGrid.y) - 1) * m_hgp->dy * lam / 2;
				m_hrn[hi].z = m_gelPosition.z + (2 * double(zi) - double(m_gelNodeGrid.z) - 1) * m_hgp->dz * lam / 2;
			}
		}
	}
        //set node type
        setType(m_hmap_node, 2);
        setType(m_hmap_element, 1);

        // build the list of boundary nodes that the coupler needs to sample
        buildBoundaryIndex();
}

void Gel::setType(int* a, int size)
{
	int xi, yi, zi;
	int LX = m_gelNodeGrid.x + size - 1;
	int LY = m_gelNodeGrid.y + size - 1;
	int LZ = m_gelNodeGrid.z + size - 1;
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

void Gel::copyDataToDevice()
{
	cudaMemcpy(m_dum, m_hum, sizeof(double) * m_numGelElements, cudaMemcpyHostToDevice);
	cudaMemcpy(m_dvm, m_hvm, sizeof(double) * m_numGelElements, cudaMemcpyHostToDevice);
	cudaMemcpy(m_dwm, m_hwm, sizeof(double) * m_numGelElements, cudaMemcpyHostToDevice);
	cudaMemcpy(m_drn, m_hrn, sizeof(double3) * m_numGelNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(m_dmap_element, m_hmap_element, sizeof(int) * m_numGelElements, cudaMemcpyHostToDevice);
	cudaMemcpy(m_dmap_node, m_hmap_node, sizeof(int) * m_numGelNodes, cudaMemcpyHostToDevice);
	cudaMemcpy(m_dgp, m_hgp, sizeof(GelParams), cudaMemcpyHostToDevice);
	cudaMemcpy(m_dbIndex, m_hbIndex, sizeof(int) * m_boundaryCount, cudaMemcpyHostToDevice);
}

void Gel::copyDataToHost()
{
	cudaMemcpyAsync(m_hum, m_dum, sizeof(double) * m_numGelElements, cudaMemcpyDeviceToHost, m_gel_stream);
	cudaMemcpyAsync(m_hvm, m_dvm, sizeof(double) * m_numGelElements, cudaMemcpyDeviceToHost, m_gel_stream);
	cudaMemcpyAsync(m_hwm, m_dwm, sizeof(double) * m_numGelElements, cudaMemcpyDeviceToHost, m_gel_stream);
	cudaMemcpyAsync(m_hrm, m_drm, sizeof(double3) * m_numGelElements, cudaMemcpyDeviceToHost, m_gel_stream);
	cudaMemcpyAsync(m_hrn, m_drn, sizeof(double3) * m_numGelNodes, cudaMemcpyDeviceToHost, m_gel_stream);
	cudaMemcpyAsync(m_hFn, m_dFn, sizeof(double3) * m_numGelNodes, cudaMemcpyDeviceToHost, m_gel_stream);
	cudaMemcpyAsync(m_hVeln, m_dVeln, sizeof(double3) * m_numGelNodes, cudaMemcpyDeviceToHost, m_gel_stream);
	cudaStreamSynchronize(m_gel_stream);
}

void Gel::freeHostMemory()
{
	delete[] m_hum;
	delete[] m_hvm;
	delete[] m_hwm;
	delete[] m_hrn;
	delete[] m_hrm;
	delete[] m_hFn;
	delete[] m_hVeln;
	delete[] m_hmap_element;
	delete[] m_hmap_node;
	delete[] m_hbIndex;
	delete m_hgp;
}

void Gel::freeDeviceMemory()
{
	cudaFree(m_dum);
	cudaFree(m_dum_norm);
	cudaFree(m_dun_norm);
	cudaFree(m_dun_robin);
	cudaFree(m_dvm);
	cudaFree(m_dvm_norm);
	cudaFree(m_dvn_norm);
	cudaFree(m_dwm);
	cudaFree(m_dwmp);
	cudaFree(m_dT0m);
	cudaFree(m_dT1m);
	cudaFree(m_dT2m);
	cudaFree(m_drn);
	cudaFree(m_drm);
	cudaFree(m_dFn);
	cudaFree(m_dFn_robin);
	cudaFree(m_dVeln);
	cudaFree(m_dVels);
	cudaFree(m_drm_loc);
	cudaFree(m_dnmSm);
	cudaFree(m_dVolm);
	cudaFree(m_dPrem);
	cudaFree(m_dmap_element);
	cudaFree(m_dmap_node);
	cudaFree(m_dbIndex);
	cudaFree(m_dgp);
}

void Gel::steadyStateValue(double& um, double& vm, double& wm, double phi)
{
	wm = m_hgp->FA0;
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

double Gel::fu_h(double u, double v, double w, double phi)
{
	double ww = (1.0 - w) * (1.0 - w);
	//return ww * u - u * u - (1.0 - w) * m_hgp->f * v * (u - m_hgp->q * ww) / (u + m_hgp->q * ww) + phi * m_hgp->P2;
	return ww * u - u * u - (1.0 - w) * (m_hgp->f * v + phi) * (u - m_hgp->q * ww) / (u + m_hgp->q * ww);
}

void Gel::recordData(int time)
{
	if (time % 1 == 0) {
		ofstream frn, frm, fum, fvm, fwm, fFn, fVeln;
		string str_rn, str_rm, str_um, str_vm, str_wm, str_Fn, str_Veln;
		str_rn = "gel" + to_string(m_gelId) + "rn" + to_string(time) + ".dat";
		str_rm = "gel" + to_string(m_gelId) + "rm" + to_string(time) + ".dat";
		str_um = "gel" + to_string(m_gelId) + "um" + to_string(time) + ".dat";
		str_vm = "gel" + to_string(m_gelId) + "vm" + to_string(time) + ".dat";
		str_wm = "gel" + to_string(m_gelId) + "wm" + to_string(time) + ".dat";
		str_Fn = "gel" + to_string(m_gelId) + "Fn" + to_string(time) + ".dat";
		str_Veln = "gel" + to_string(m_gelId) + "Veln" + to_string(time) + ".dat";
		frn.open(str_rn);
		frm.open(str_rm);
		fvm.open(str_vm);
		fum.open(str_um);
		fwm.open(str_wm);
		fFn.open(str_Fn);
		fVeln.open(str_Veln);
		int gi;
		for (int zi = 1; zi < m_gelNodeGrid.z + 1; zi++) {
			for (int yi = 1; yi < m_gelNodeGrid.y + 1; yi++) {
				for (int xi = 1; xi < m_gelNodeGrid.x + 1; xi++) {
					gi = get_index(xi, yi, zi, 2);
					frn << setw(9) << to_string(m_hrn[gi].x) << "      " << setw(9) << to_string(m_hrn[gi].y) << "      " << setw(9) << to_string(m_hrn[gi].z) << "\n";
					fFn << setw(9) << to_string(m_hFn[gi].x) << "      " << setw(9) << to_string(m_hFn[gi].y) << "      " << setw(9) << to_string(m_hFn[gi].z) << "\n";
					fVeln << setw(9) << to_string(m_hVeln[gi].x) << "      " << setw(9) << to_string(m_hVeln[gi].y) << "      " << setw(9) << to_string(m_hVeln[gi].z) << "\n";
				}
			}
		}
		for (int zi = 1; zi < m_gelNodeGrid.z; zi++) {
			for (int yi = 1; yi < m_gelNodeGrid.y; yi++) {
				for (int xi = 1; xi < m_gelNodeGrid.x; xi++) {
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
}

void Gel::writeFiles(int iter)
{
	double time = iter * m_dt;
	if (iter % 1000 == 0) {
		if (m_file_writer_thread.joinable()) {
			m_file_writer_thread.join();
		}
		copyDataToHost();
		m_file_writer_thread = thread(&Gel::recordData, this, int(time));
	}
}

Gel::Gel(int3 gelSize, double3 gelPosition, int gelType, int gelId, int time):
	m_gelSize(gelSize),
	m_gelId(gelId),
	m_gelPosition(gelPosition),
	m_gelType(gelType),
	//CPU data
	m_hum(0),
	m_hvm(0),
	m_hwm(0),
	m_hrn(0),
	m_hrm(0),
	m_hFn(0),
	m_hVeln(0),
	m_hmap_element(0),
	m_hmap_node(0),
	m_hbIndex(0),
	//GPU data
	m_dum(0),
	m_dum_norm(0),
	m_dun_norm(0),
	m_dun_robin(0),
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
	m_dFn_robin(0),
	m_dVeln(0),
	m_dVels(0),
	m_drm_loc(0),
	m_dnmSm(0),
	m_dVolm(0),
	m_dPrem(0),
	m_dmap_element(0),
	m_dmap_node(0),
	m_dbIndex(0)
{
	m_dt = 1e-3;
	m_df = int(1 / m_dt);
	m_dx = 1;
	m_dy = 1;
	m_dz = 1;
	m_gelNodeGrid.x = int(gelSize.x / m_dx) + 1;
	m_gelNodeGrid.y = int(gelSize.y / m_dy) + 1;
	m_gelNodeGrid.z = int(gelSize.z / m_dz) + 1;
	m_numGelElements = (m_gelNodeGrid.x + 1) * (m_gelNodeGrid.y + 1) * (m_gelNodeGrid.z + 1);
	m_numGelNodes = (m_gelNodeGrid.x + 2) * (m_gelNodeGrid.y + 2) * (m_gelNodeGrid.z + 2);
	m_boundaryCount = m_gelNodeGrid.x * m_gelNodeGrid.y * m_gelNodeGrid.z - (m_gelNodeGrid.x - 2) * (m_gelNodeGrid.y - 2) * (m_gelNodeGrid.z - 2);
	m_blockDim = dim3(8, 8, 8);

	m_gridDim_1.x = (m_gelNodeGrid.x - 1 + m_blockDim.x - 1) / m_blockDim.x;
	m_gridDim_1.y = (m_gelNodeGrid.y - 1 + m_blockDim.y - 1) / m_blockDim.y;
	m_gridDim_1.z = (m_gelNodeGrid.z - 1 + m_blockDim.z - 1) / m_blockDim.z;

	m_gridDim0.x = (m_gelNodeGrid.x + m_blockDim.x - 1) / m_blockDim.x;
	m_gridDim0.y = (m_gelNodeGrid.y + m_blockDim.y - 1) / m_blockDim.y;
	m_gridDim0.z = (m_gelNodeGrid.z + m_blockDim.z - 1) / m_blockDim.z;

	m_gridDim1.x = (m_gelNodeGrid.x + 1 + m_blockDim.x - 1) / m_blockDim.x;
	m_gridDim1.y = (m_gelNodeGrid.y + 1 + m_blockDim.y - 1) / m_blockDim.y;
	m_gridDim1.z = (m_gelNodeGrid.z + 1 + m_blockDim.z - 1) / m_blockDim.z;

	m_gridDim2.x = (m_gelNodeGrid.x + 2 + m_blockDim.x - 1) / m_blockDim.x;
	m_gridDim2.y = (m_gelNodeGrid.y + 2 + m_blockDim.y - 1) / m_blockDim.y;
	m_gridDim2.z = (m_gelNodeGrid.z + 2 + m_blockDim.z - 1) / m_blockDim.z;
	m_hgp = new GelParams;
	memset(m_hgp, 0, sizeof(GelParams));
	m_hgp->LX = m_gelNodeGrid.x;
	m_hgp->LY = m_gelNodeGrid.y;
	m_hgp->LZ = m_gelNodeGrid.z;
	m_hgp->I = 0.0;
	m_hgp->f = 0.9;
	m_hgp->ep = 0.3;
	m_hgp->q = 1e-4;
	m_hgp->P1 = 0.0124;
	m_hgp->P2 = 0.77;
	m_hgp->dt = m_dt;
	m_hgp->dtx = 5 * m_dt;
	m_hgp->dx = m_dx;
	m_hgp->dy = m_dy;
	m_hgp->dz = m_dz;
	m_hgp->CH0 = 0.338;
	m_hgp->CH1 = 0.518;
	m_hgp->CHS = 0.1;
	m_hgp->C0 = 1.3e-3;
	m_hgp->c0_bis = 1e-4;
	m_hgp->b = 0.01;
	m_hgp->AZ0 = 100.0;
	m_hgp->FA0 = 0.139;
	m_hgp->gelType = m_gelType;
	steadyStateValue(m_hgp->uss, m_hgp->vss, m_hgp->wss, 0);
	int3 offset[27] = {
	{ 0, 0, 0 },
	{ 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 },
	{ 0, 1, 1 }, { 0, 1, -1 }, { 0, -1, 1 }, { 0, -1, -1 }, { 1, 0, 1 }, { 1, 0, -1 }, { -1, 0, 1 }, { -1, 0, -1 }, { 1, 1, 0 }, { 1, -1, 0 }, { -1, 1, 0 }, { -1, -1, 0 },
	{ 1, 1, 1 }, { 1, 1, -1 }, { 1, -1, 1 }, { 1, -1, -1 }, { -1, 1, 1 }, { -1, 1, -1 }, { -1, -1, 1 }, { -1, -1, -1 } };
	memcpy(m_hgp->rn_offset, offset, sizeof(offset));
	memcpy(m_hgp->um_offset_noflux, offset, sizeof(offset));
	_initialize(time);
}

Gel::~Gel()
{
	_finalize();
}

void Gel::_initialize(int time)
{
	allocateHostStorage();
	allocateDeviceStorage();
	cudaStreamCreate(&m_gel_stream);
	setInitValue();
	copyDataToDevice();
}

void Gel::stepElasticity(int iter)
{
	int time = int(iter * m_dt);
	if (iter % 5 == 0) {
		calServiceNodesPositionD << < m_gridDim2, m_blockDim, 0, m_gel_stream >> > (m_drn, m_dmap_node, m_dgp);
		calElementPropertiesD << <m_gridDim1, m_blockDim, 0, m_gel_stream >> > (m_drn, m_drm, m_drm_loc, m_dnmSm, m_dVolm, m_dwm, m_dwmp, m_dgp);
		calPressureD << < m_gridDim_1, m_blockDim, 0, m_gel_stream >> > (m_dPrem, m_dvm, m_dwm, m_dgp);
		calNodesVelocityD << < m_gridDim0, m_blockDim, 0, m_gel_stream >> > (m_drn, m_dVeln, m_dVels, m_dFn, m_dFn_robin, m_dnmSm, m_dPrem, m_dwm, m_dvn_norm, m_dgp);
		calInternalNodesPositionD << < m_gridDim0, m_blockDim, 0, m_gel_stream >> > (m_drn, m_dVeln, m_dgp);
		calChemBoundaryD << < m_gridDim1, m_blockDim, 0, m_gel_stream >> > (m_dum, m_dum_norm, m_dvm, m_dvm_norm, m_dwm, m_dmap_element, time, m_dgp);
		calUnnormD << < m_gridDim0, m_blockDim, 0, m_gel_stream >> > (m_dun_norm, m_dun_robin, m_dum_norm, m_dvn_norm, m_dvm_norm, m_dgp);
		calTermsD << < m_gridDim_1, m_blockDim, 0, m_gel_stream >> > (m_dT0m, m_dT1m, m_dT2m, m_dwm, m_dwmp, m_dVeln, m_dnmSm, m_dVolm, m_drm_loc, m_dun_norm, m_dum_norm, m_drm, m_dgp);
		setZero << < m_gridDim0, m_blockDim, 0, m_gel_stream >> > (m_dun_robin, m_dFn_robin, m_dgp);
	}
}

void Gel::stepChemistry(int iter)
{
	int time = int(iter * m_dt);
	calChemD << < m_gridDim_1, m_blockDim, 0, m_gel_stream >> > (m_dvm, m_dum, m_dwm, m_dT0m, m_dT1m, m_dT2m, m_drm, time, m_dgp);
}

void Gel::_finalize()
{
	if (m_file_writer_thread.joinable()) {
		m_file_writer_thread.join();
	}
    cudaStreamDestroy(m_gel_stream);
	freeHostMemory();
	freeDeviceMemory();
}