#include "coupling.h"
#include "gel.h"
#include "fluid.h"
#include "coupling_kernels.cuh"

void Coupler::allocateHostStorage()
{
    h_offsets.assign(numGels, 0);
    h_gelBoundaryCount.assign(numGels, 0);
    h_owner.assign(sumGelBoundaryCount, 0);

}

void Coupler::allocateDeviceStorage()
{
    cudaMalloc((void**)&d_cp, sizeof(CouplerParams));
    cudaMalloc(&d_lag_all_, sizeof(float3) * sumGelBoundaryCount);
    cudaMalloc(&d_Ul_all_, sizeof(float3) * sumGelBoundaryCount);
    cudaMalloc(&d_Vl_all_, sizeof(float3) * sumGelBoundaryCount);
    cudaMalloc(&d_Fl_all_, sizeof(float3) * sumGelBoundaryCount);
    cudaMalloc(&d_Cl_all_, sizeof(float) * sumGelBoundaryCount);
    cudaMalloc(&d_Dl_all_, sizeof(float) * sumGelBoundaryCount);
    cudaMalloc(&d_bIndex_all_, sizeof(float) * sumGelBoundaryCount);
}

void Coupler::copyDataToDevice()
{
    cudaMemcpy(d_cp, h_cp, sizeof(CouplerParams), cudaMemcpyHostToDevice);
    for (int i = 0; i < numGels; ++i) {
        const int Mi = gels[i]->m_boundaryCount;
        const int off = h_offsets[i];
        cudaMemcpyAsync(d_bIndex_all_ + off, gels[i]->m_hbIndex, sizeof(int) * Mi, cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_owner, h_owner.data(), sizeof(int) * sumGelBoundaryCount, cudaMemcpyHostToDevice);
}

void Coupler::setInitValue()
{
    int sumM = 0;
    for (int i = 0; i < numGels; ++i) {
        const int Mi = gels[i]->m_boundaryCount;
        h_gelBoundaryCount[i] = Mi;
        h_offsets[i] = sumM;
        for (int l = 0; l < Mi; ++l) h_owner[sumM + l] = i;
        sumM += h_gelBoundaryCount[i];
    }
}
Coupler::Coupler(std::vector<Gel*>& gels, Fluid* fluid) :
    gels(gels),
    fluid(fluid),
    d_lag_all_(0),
    d_Ul_all_(0),
    d_Vl_all_(0),
    d_Fl_all_(0),
    d_Cl_all_(0),
    d_Dl_all_(0),
    d_bIndex_all_(0)
{
    numGels = (int)gels.size();
    sumGelBoundaryCount = 0;
    for (int i = 0; i < numGels; ++i) {
        sumGelBoundaryCount += gels[i]->m_boundaryCount;
    }
    threads = 256;
    blocksM = (sumGelBoundaryCount + threads - 1) / threads;
    h_cp = new CouplerParams;
    memset(h_cp, 0, sizeof(CouplerParams));
    h_cp->h = fluid->h_fp->h;
    h_cp->L = fluid->h_fp->L;
    h_cp->M = sumGelBoundaryCount;
    h_cp->beta = 0.01;
    h_cp->delta = 1e-4;
    h_cp->gamma = 1.5;
    _initialize();
}

void Coupler::packFromGels() {
    for (int i = 0; i < numGels; ++i) {
        if (!gels[i]->boundaryDirty()) {
            continue;
        }
        const int off = h_offsets[i];
        const int Mi = gels[i]->m_boundaryCount;
        const int blocks = (Mi + threads - 1) / threads;
        // 1.  kernelûзֵ
        k_gather_boundary << <blocks, threads, 0, coupler_stream >> > (
            gels[i]->m_dbIndex,
            gels[i]->m_drn,
            gels[i]->m_dVels,
            gels[i]->m_dun_norm,
            d_lag_all_ + off,
            d_Vl_all_ + off,
            d_Cl_all_ + off,
            Mi);

        gels[i]->markBoundaryClean();

        // 2. Ȼ cudaGetLastError() ȡǷ
        //cudaError_t err = cudaGetLastError();
        //if (err != cudaSuccess) {
        //    fprintf(stderr, "k_gather_boundary launch failed: %s\n",
        //        cudaGetErrorString(err));
        //    // return;  std::abort();
        //}

    }
}

void Coupler::scatterToGels() {
    for (int i = 0; i < numGels; ++i) {
        const int Mi = gels[i]->m_boundaryCount;
        const int off = h_offsets[i];
        const int blocks = (Mi + threads - 1) / threads;
        k_add_reaction_to_gel << <blocks, threads, 0, coupler_stream >> > (gels[i]->m_dbIndex, gels[i]->m_dFn, gels[i]->m_dun_norm, d_Vl_all_ + off, d_Cl_all_ + off, Mi);
    }
}

void Coupler::applyGelRepulsion() {
    k_gel_repulsion << <blocksM, threads, 0, coupler_stream >> > (d_lag_all_, d_owner, d_Fl_all_, d_cp);
}

void Coupler::_initialize()
{
    allocateHostStorage();
    allocateDeviceStorage();
    cudaStreamCreate(&coupler_stream);
    setInitValue();
    copyDataToDevice();
}

void Coupler::update(long long int solverIterations)
{
    float ramp = fmin(1, (solverIterations + 1) / 20000.0);
    float beta_eff = h_cp->beta * ramp;
    k_ibm_interpolate << <blocksM, threads, 0, coupler_stream >> > (fluid->d_u, fluid->d_c1, d_Ul_all_, d_Dl_all_, d_lag_all_, d_cp);
    k_scale_negbeta << <blocksM, threads, 0, coupler_stream >> > (d_Ul_all_, d_Vl_all_, d_Fl_all_, beta_eff, d_cp);
    std::swap(d_Cl_all_, d_Dl_all_);
    k_ibm_spread << <blocksM, threads, 0, coupler_stream >> > (fluid->d_F_ibm, fluid->d_c1, d_Fl_all_, d_Dl_all_, d_lag_all_, d_A, d_cp);
}

void Coupler::freeHostMemory()
{
    free(h_cp);
}

void Coupler::freeDeviceMemory()
{
    cudaFree(d_lag_all_);
    cudaFree(d_Ul_all_);
    cudaFree(d_Vl_all_);
    cudaFree(d_Fl_all_);
    cudaFree(d_Cl_all_);
    cudaFree(d_Dl_all_);
    cudaFree(d_bIndex_all_);
}

void Coupler::_finalize()
{
    cudaStreamDestroy(coupler_stream);
    freeDeviceMemory();
}

Coupler::~Coupler()
{
    _finalize();
}