#include "coupling.h"
#include "gel.h"
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
    cudaMalloc(&d_Fdrag_all_, sizeof(float3) * sumGelBoundaryCount);
    cudaMalloc(&d_Frep_all_, sizeof(float3) * sumGelBoundaryCount);
    cudaMalloc(&d_Cl_all_, sizeof(float) * sumGelBoundaryCount);
    cudaMalloc(&d_Sl_all_, sizeof(float) * sumGelBoundaryCount);
    cudaMalloc(&d_Dl_all_, sizeof(float) * sumGelBoundaryCount);
    cudaMalloc(&d_bIndex_all_, sizeof(int) * sumGelBoundaryCount);
    cudaMalloc(&d_owner, sizeof(int) * sumGelBoundaryCount);
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
Coupler::Coupler(std::vector<Gel*>& gels) :
    gels(gels),
    d_lag_all_(0),
    d_Ul_all_(0),
    d_Vl_all_(0),
    d_Fdrag_all_(0),
    d_Frep_all_(0),
    d_Cl_all_(0),
    d_Sl_all_(0),
    d_Dl_all_(0),
    d_bIndex_all_(0),
    d_owner(0),
    d_cp(0),
    coupler_stream(0)
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
    h_cp->M = sumGelBoundaryCount;
    h_cp->delta = 1e-4f;
    h_cp->gamma = 1.5f;
    h_cp->gel_to_lbm_vel = 1.0f;
    h_cp->lbm_to_gel_vel = 1.0f;
    _initialize();
}

void Coupler::packFromGels() {
    for (int i = 0; i < numGels; ++i) {
        const int off = h_offsets[i];
        const int Mi = gels[i]->m_boundaryCount;
        const int blocks = (Mi + threads - 1) / threads;
        k_gather_boundary << <blocks, threads, 0, gels[i]->m_gel_stream >> > (
            gels[i]->m_dbIndex,
            gels[i]->m_drn,
            gels[i]->m_dVels,
            gels[i]->m_dun_norm,
            d_lag_all_ + off,
            d_Vl_all_ + off,
            d_Cl_all_ + off,
            Mi,
            h_cp->gel_to_lbm_vel);
    }
}

void Coupler::scatterToGels() {
    for (int i = 0; i < numGels; ++i) {
        const int Mi = gels[i]->m_boundaryCount;
        const int off = h_offsets[i];
        const int blocks = (Mi + threads - 1) / threads;
        k_add_reaction_to_gel<<<blocks, threads, 0, gels[i]->m_gel_stream >>>(
            gels[i]->m_dbIndex,
            gels[i]->m_dFn_robin,
            gels[i]->m_dun_robin,
            d_Frep_all_ + off,
            d_Sl_all_ + off,
            Mi);
        k_add_drag_to_gel<<<blocks, threads, 0, gels[i]->m_gel_stream >>>(
            gels[i]->m_dbIndex,
            gels[i]->m_dFdrag_robin,
            d_Fdrag_all_ + off,
            Mi);
    }
}

void Coupler::applyGelRepulsion() {
    cudaMemsetAsync(d_Frep_all_, 0, sizeof(float3) * sumGelBoundaryCount, coupler_stream);
    k_gel_repulsion << <blocksM, threads, 0, coupler_stream >> > (d_lag_all_, d_owner, d_Frep_all_, d_cp);
}

void Coupler::_initialize()
{
    cudaStreamCreate(&coupler_stream);
    allocateHostStorage();
    allocateDeviceStorage();
    setInitValue();
    copyDataToDevice();
}

void Coupler::freeHostMemory()
{
    delete h_cp;
}

void Coupler::freeDeviceMemory()
{
    cudaFree(d_lag_all_);
    cudaFree(d_Ul_all_);
    cudaFree(d_Vl_all_);
    cudaFree(d_Fdrag_all_);
    cudaFree(d_Frep_all_);
    cudaFree(d_Cl_all_);
    cudaFree(d_Sl_all_);
    cudaFree(d_Dl_all_);
    cudaFree(d_bIndex_all_);
    cudaFree(d_owner);
    cudaFree(d_cp);
}

void Coupler::_finalize()
{
    cudaStreamDestroy(coupler_stream);
    freeHostMemory();
    freeDeviceMemory();
}

Coupler::~Coupler()
{
    _finalize();
}