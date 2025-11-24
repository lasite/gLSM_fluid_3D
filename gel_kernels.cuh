#ifndef GEL_KERNEL_CUH
#define GEL_KERNEL_CUH
#include "gel.h"
__global__ void calElementPropertiesD(double3* rn, double3* rm, double3* rm_loc, double3* nmSm, double* volume, double* wm, double* wmp, GelParams* gp);

__global__ void calPressureD(double* pm, double* vm, double* wm, GelParams* gp);

__global__ void calNodesVelocityD(double3* rn, double3* ven, double3* ves, double3* Fn, double3* Fn_robin, double3* nmSm, double* pm, double* wm, double* vn_norm, GelParams* gp);

__global__ void calInternalNodesPositionD(double3* rn, double3* ven, GelParams* gp);

__global__ void calServiceNodesPositionD(double3* rn, int* map_node, GelParams* gp);

__global__ void calTermsD(double* T0, double* T1, double* T2, double* wm, double* wmp, double3* ven, double3* nmSm, double* volume, double3* rm_loc, double* un_norm, double* um_norm, double3* rm, GelParams* gp);

__global__ void calChemD(double* vm, double* um, double* wm, double* T0, double* T1, double* T2, double3* rm, int time, GelParams* gp);

__global__ void calChemBoundaryD(double* um, double* um_norm, double* vm, double* vm_norm, double* wm, int* map_element, int time, GelParams* gp);

__global__ void calUnnormD(double* un_norm, double* un_robin, double* um_norm, double* vn_norm, double* vm_norm, GelParams* gp);

__global__ void setZero(double* un_robin, double3* Fn_robin, GelParams* gp);

#endif
