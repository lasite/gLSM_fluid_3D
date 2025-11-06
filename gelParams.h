#ifndef GELPARAMS_H
#define GELPARAMS_H

// simulation parameters

struct GelParams {
	int LX;
	int LY;
	int LZ;
	double f;
	double q;
	double ep;
	double P1;
	double P2;
	double dt;
	double dtx;
	double dx;
	double dy;
	double dz;
	double CH0;
	double CH1;
	double CHS;
	double C0;
	double AZ0;
	double FA0;
	double uss;
	double vss;
	double wss;
	double I;
	int3 rn_offset[27];
	int3 um_offset_noflux[27];
	int3 um_offset_periodic[27];
	int TargetWave_y;
	int TargetWave_z;
	int maxFilamentlen;
};

struct FluidParams {
	int3 c[19];
	float w[19];
	int opp[19];
	float tau, cs2, nu;
	float3 F_const;
	float beta;
	float N;
	float M;
	int3 L;
	float h;
};

#endif