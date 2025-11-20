#pragma once
#include <thread>
#include <cuda_runtime.h>

#include "coupling.h"

struct FluidParams {
    int3  c[19];
    float w[19];
    int   opp[19];
    float tau, cs2, nu;
    float3 F_const;
    float beta;
    int  N;
    int  M;
    int3   L;
    float  h;
    float dx, dy, dz;
    float dt;
    float D;
    int perX, perY, perZ;
};

class Fluid {
public:
    Fluid(int3 fluidSize, int time, Coupler* coupler);
    ~Fluid();
    void _initialize(int time);
    void stepVelocity(long long int iter);
    void stepConcentration();
    void _finalize();
    void writeFiles(int iter);

public:
    void allocateHostStorage();
    void allocateDeviceStorage();
    void freeHostMemory();
    void freeDeviceMemory();
    void setInitValue();
    void copyDataToDevice();
    void copyDataToHost();
    void recordData(int time);
    int idx3(int x, int y, int z, int Nx, int Ny);
public:  // data
    // CPU data
    float3* h_u;
    float* h_c1;
    int* h_bIndex;
    FluidParams* h_fp;
    // GPU data
    float* d_f;
    float* d_fpost;
    float* d_fnext;
    float* d_rho;
    float3* d_u;
    float3* d_F;
    float* d_c1;
    float* d_c2;
    float3* d_F_ibm;
    float3* d_F_tot;
    FluidParams* d_fp;
public:
    // params
    Coupler* coupler;
    int3 fluidSize;
    int df;
    float dt;
    int N;
    int Nd;
    float dt_fluid;
    int Nsub;
    float* d_A;
    float niu;
    float dx_fluid;
    std::thread file_writer_thread;
    cudaStream_t fluid_stream;
    int threads;
    int blocksN;
    int blocksM;
};
