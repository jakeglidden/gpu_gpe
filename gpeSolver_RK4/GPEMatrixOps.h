#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef __CUDA_RUNTIME_H__
#include <cuda_runtime.h>
#endif

#ifndef _CUFFT_H_
#include <cufft.h>
#endif

#ifndef CUBLAS_V2_H_
#include <cublas_v2.h>
#endif

#ifndef TWOPI
#define TWOPI (real)(8.0 * atan(1.0))
#endif

static void HandleError(cudaError_t err, const char *file, const char *function, int line) {
	if (err != cudaSuccess) {
		printf("!!! (CUDA) %s:%s (line %d) %s\n", strrchr(file, '/')+1, function, line, cudaGetErrorName(err));
		printf("--> %s\n\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__ , __FUNCTION__ , __LINE__ ))

typedef cufftComplex complex;
typedef cufftReal real;

typedef struct {
	real
		AtomNumber,
		ScatteringLength,
		Radius,
		Length,
		Depth,
		Timestep,
		ShakeAmplitude,
		ShakeFrequency,
		SimulationLength,
		ITSimulationLength,
		XRange,
		YRange,
		ZRange,
		BoundaryDepth,
		BoundaryWidth;
	unsigned int
		NX,
		NY,
		NZ,
		NOutputs;
	real
		*OutputTimes = NULL;
} SimArgs;

typedef struct Geometry 
{
	short unsigned int nx;
	real xlim[2];
	short unsigned int ny;
	real ylim[2];
	short unsigned int nz;
	real zlim[2];
	real dx,  dy,  dz;
	real dtau;
	real dkx, dky, dkz;
	real dktau;
	real dt;
	unsigned int npts;
	real oneonnpts;
	unsigned int npar;
	real *x, *y, *z;
	real *d_z;					// We'll use this when evaluating the time-varying potential
	real *kx, *ky, *kz, *ksq, *d_kz, *d_pot, *d_pot_t, *d_bnd;
	real Tprefix, Vprefix, gprefix;
	real rootN;
	complex *d_dpsi1, *d_dpsi2, *d_dpsi3, *d_dpsi4;
	complex *d_kpsi;
	cufftHandle fftPlan;
	cublasHandle_t cbHandle;
} geometry;

void populateGeometry(geometry* g, SimArgs* sa) {
	
	g->nx = sa->NX;
	g->ny = sa->NY;
	g->nz = sa->NZ;
	g->xlim[0] = -sa->XRange / 2.0f;
	g->xlim[1] = +sa->XRange / 2.0f;
	g->ylim[0] = -sa->YRange / 2.0f;
	g->ylim[1] = +sa->YRange / 2.0f;
	g->zlim[0] = -sa->ZRange / 2.0f;
	g->zlim[1] = +sa->ZRange / 2.0f;
	
	/* find steps and volume element */
	g->dx = (g->xlim[1] - g->xlim[0]) / ((real)g->nx);
	g->dy = (g->ylim[1] - g->ylim[0]) / ((real)g->ny);
	g->dz = (g->zlim[1] - g->zlim[0]) / ((real)g->nz);
	g->dtau = (g->dx)*(g->dy)*(g->dz);
	g->dt = sa->Timestep;

	/* find ksteps */
	g->dkx = TWOPI / (g->xlim[1] - g->xlim[0]);
	g->dky = TWOPI / (g->ylim[1] - g->ylim[0]);
	g->dkz = TWOPI / (g->zlim[1] - g->zlim[0]);

	/* total number of voxels */
	g->npts = (g->nx) * (g->ny) * (g->nz);
	g->oneonnpts = 1.0f / ((real)g->npts);
	g->npar = (g->npts) / 128;

	/* position vectors - allocate memory */
	if (g->x != NULL) free(g->x); g->x = (real *)malloc((g->nx) * sizeof(real));
	if (g->x == NULL) printf("!!! %s:%s (line %d) Error allocating CPU memory.\n", strrchr(__FILE__, '/')+1, __FUNCTION__, __LINE__);
	if (g->y != NULL) free(g->y); g->y = (real *)malloc((g->ny) * sizeof(real));
	if (g->y == NULL) printf("!!! %s:%s (line %d) Error allocating CPU memory.\n", strrchr(__FILE__, '/')+1, __FUNCTION__, __LINE__);
	if (g->z != NULL) free(g->z); g->z = (real *)malloc((g->nz) * sizeof(real));
	if (g->z == NULL) printf("!!! %s:%s (line %d) Error allocating CPU memory.\n", strrchr(__FILE__, '/')+1, __FUNCTION__, __LINE__);

	/* k vectors - allocate memory */
	if (g->kx != NULL) free(g->kx); g->kx = (real *)malloc((g->nx) * sizeof(real));
	if (g->kx == NULL) printf("!!! %s:%s (line %d) Error allocating CPU memory.\n", strrchr(__FILE__, '/')+1, __FUNCTION__, __LINE__);
	if (g->ky != NULL) free(g->ky); g->ky = (real *)malloc((g->ny) * sizeof(real));
	if (g->ky == NULL) printf("!!! %s:%s (line %d) Error allocating CPU memory.\n", strrchr(__FILE__, '/')+1, __FUNCTION__, __LINE__);
	if (g->kz != NULL) free(g->kz); g->kz = (real *)malloc((g->nz) * sizeof(real));
	if (g->kz == NULL) printf("!!! %s:%s (line %d) Error allocating CPU memory.\n", strrchr(__FILE__, '/')+1, __FUNCTION__, __LINE__);

	/* Fill position and k-space grids */
	for (short unsigned int ii = 0; ii < (g->nx); ii++) {
		(g->x)[ii] = (g->xlim)[0] + ii*(g->dx);
		(g->kx)[ii] = (g->dkx)*(ii < (g->nx / 2) ? ii : (ii - (g->nx)));
	}
	for (short unsigned int ii = 0; ii < (g->ny); ii++) {
		(g->y)[ii] = (g->ylim)[0] + ii*(g->dy);
		(g->ky)[ii] = (g->dky)*(ii < (g->ny / 2) ? ii : (ii - (g->ny)));
	}
	for (short unsigned int ii = 0; ii < (g->nz); ii++) {
		(g->z)[ii] = (g->zlim)[0] + ii*(g->dz);
		(g->kz)[ii] = (g->dkz)*(ii < (g->nz / 2) ? ii : (ii - (g->nz)));
	}

	/* Create ksq matrix for extracting mu */
	if (g->ksq != NULL) { HANDLE_ERROR(cudaFree(g->ksq)); g->ksq = NULL; }
	if (g->d_kz != NULL) { HANDLE_ERROR(cudaFree(g->d_kz)); g->d_kz = NULL; }

	HANDLE_ERROR(cudaMalloc((void**)&(g->d_kz), (g->nz) * sizeof(real)));
	HANDLE_ERROR(cudaMemcpy(g->d_kz, g->kz, (g->nz) * sizeof(real), cudaMemcpyHostToDevice));

	/* Get coefficients */
	g->Tprefix = -(real)(0.8149609555 * sa->Timestep);
	g->Vprefix = -(real)(0.1309203 * sa->Timestep);
	g->gprefix = (real)(8.278863628e-3 * (sa->ScatteringLength) * (g->Vprefix));
	g->rootN   = sqrt((sa->AtomNumber) / (g->dtau));
	g->d_pot_t = NULL; // just to be safe

	/* Prepare for CUBLAS and CUFFT operations */
	cufftPlan3d(&(g->fftPlan), g->nx, g->ny, g->nz, CUFFT_C2C);
	cublasCreate(&(g->cbHandle));

}

void depopulateGeometry(geometry *g) {
	if (g->x != NULL) { free(g->x); g->x = NULL; }
	if (g->y != NULL) { free(g->y); g->y = NULL; }
	if (g->z != NULL) { free(g->z); g->z = NULL; }

	if (g->kx != NULL) { free(g->kx); g->kx = NULL; }
	if (g->ky != NULL) { free(g->ky); g->ky = NULL; }
	if (g->kz != NULL) { free(g->kz); g->kz = NULL; }

	if (g->d_kz != NULL) { HANDLE_ERROR(cudaFree(g->d_kz)); g->d_kz = NULL; }
	if (g->ksq != NULL) { HANDLE_ERROR(cudaFree(g->ksq)); g->ksq = NULL; }

	if (g->d_dpsi1 != NULL) { HANDLE_ERROR(cudaFree(g->d_dpsi1)); g->d_dpsi1 = NULL; }
	if (g->d_dpsi2 != NULL) { HANDLE_ERROR(cudaFree(g->d_dpsi2)); g->d_dpsi2 = NULL; }
	if (g->d_dpsi3 != NULL) { HANDLE_ERROR(cudaFree(g->d_dpsi3)); g->d_dpsi3 = NULL; }
	if (g->d_dpsi4 != NULL) { HANDLE_ERROR(cudaFree(g->d_dpsi4)); g->d_dpsi4 = NULL; }
	
	if (g->d_kpsi != NULL) { HANDLE_ERROR(cudaFree(g->d_kpsi)); g->d_kpsi = NULL; }
	if (g->d_pot != NULL) { HANDLE_ERROR(cudaFree(g->d_pot)); g->d_pot = NULL; }
	if (g->d_pot_t != NULL) { HANDLE_ERROR(cudaFree(g->d_pot_t)); g->d_pot_t = NULL; }
	if (g->d_bnd != NULL) { HANDLE_ERROR(cudaFree(g->d_bnd)); g->d_bnd = NULL; }
	
	cublasDestroy(g->cbHandle);
	cufftDestroy(g->fftPlan);

}

void renormalise(complex *psi, geometry *g) {
	real calcnorm;
	cublasScnrm2(g->cbHandle, g->npts, psi, 1, &calcnorm);
	calcnorm = (g->rootN) / calcnorm;
	cublasCsscal(g->cbHandle, g->npts, &calcnorm, psi, 1);
}

void __global__ setStartingPsi(complex *psi, real prefactor, complex *dpsi, complex *dest) {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	dest[idx].x = psi[idx].x + prefactor * dpsi[idx].x;
	dest[idx].y = psi[idx].y + prefactor * dpsi[idx].y;
}

void __global__ timesKSqOnNpts(complex *kpsi, real *ksq) {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	kpsi[idx].x *= ksq[idx];
	kpsi[idx].y *= ksq[idx];
}

void __global__ addWithRealPotTerms(complex *dest, const complex *src, const complex *kpsi, const real *pot, const real *pot_t, const real *bnd, const real gprefix) {
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	real tmp = (pot[idx] + pot_t[idx] + gprefix * (src[idx].x * src[idx].x + src[idx].y * src[idx].y));
	real tmp2 = src[idx].x; // need this if src and dest are the same!!!
	dest[idx].x = - kpsi[idx].y - tmp * src[idx].y + bnd[idx] * src[idx].x;
	dest[idx].y = kpsi[idx].x + tmp * tmp2 + bnd[idx] * src[idx].y;
}

void __global__ addWithImagPotTerms(complex *dest, const complex *src, const complex *kpsi, const real *pot, const real gprefix) {
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	real tmp = (pot[idx] + gprefix * (src[idx].x * src[idx].x + src[idx].y * src[idx].y));
	dest[idx].x = kpsi[idx].x + tmp * src[idx].x;
	dest[idx].y = kpsi[idx].y + tmp * src[idx].y;
}

void __global__ RKSum(complex *psi, complex *d1, complex *d2, complex *d3, complex *d4) {
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	psi[idx].x += 0.16666666666666667 * (d1[idx].x + d4[idx].x) + 0.333333333333333 * (d2[idx].x + d3[idx].x);
	psi[idx].y += 0.16666666666666667 * (d1[idx].y + d4[idx].y) + 0.333333333333333 * (d2[idx].y + d3[idx].y);
}

void RKStep(bool isRealTime, short unsigned int stepNo, complex *psi, geometry *g) {
	
	if (g->d_kpsi == NULL) {
		HANDLE_ERROR(cudaMalloc((void**)&(g->d_kpsi), g->npts * sizeof(complex)));
	}

	complex *psi_start = NULL;
	complex *psi_end = NULL;
	switch (stepNo) {
	case 1:
		psi_start = psi;
		psi_end = g->d_dpsi1;
		break;
	case 2:
		setStartingPsi<<<g->npar, 128>>>(psi, 0.5, g->d_dpsi1, g->d_dpsi2);
		psi_start = g->d_dpsi2;
		psi_end = g->d_dpsi2;
		break;
	case 3:
		setStartingPsi<<<g->npar, 128>>>(psi, 0.5, g->d_dpsi2, g->d_dpsi3);
		psi_start = g->d_dpsi3;
		psi_end = g->d_dpsi3;
		break;
	case 4:
		setStartingPsi<<<g->npar, 128>>>(psi, 1.0, g->d_dpsi3, g->d_dpsi4);
		psi_start = g->d_dpsi4;
		psi_end = g->d_dpsi4;
		break;
	default:
		printf("!!! %s:%s (line %d) Invalid parameter passed to function.\n", strrchr(__FILE__, '/') + 1, __FUNCTION__, __LINE__);
		return;
	}

	cufftExecC2C(g->fftPlan, psi_start, g->d_kpsi, CUFFT_FORWARD);
	timesKSqOnNpts<<<g->npar, 128>>>(g->d_kpsi, g->ksq);
	cufftExecC2C(g->fftPlan, g->d_kpsi, g->d_kpsi, CUFFT_INVERSE);
	if (isRealTime) {
		addWithRealPotTerms<<<g->npar, 128>>>(psi_end, psi_start, g->d_kpsi, g->d_pot, g->d_pot_t, g->d_bnd, g->gprefix);
	} else {
		addWithImagPotTerms<<<g->npar, 128>>>(psi_end, psi_start, g->d_kpsi, g->d_pot, g->gprefix);
	}

}

void RKAdd(complex *psi, geometry *g) {
	RKSum<<<g->npar, 128>>>(psi, g->d_dpsi1, g->d_dpsi2, g->d_dpsi3, g->d_dpsi4);
}

void __global__ MakeV_t_RT(real *V_t, const real Vprefix, const real *d_z, const short unsigned int nz, const real t, const SimArgs simArgs) {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int idz = idx % nz;
	V_t[idx] = Vprefix * (TIMEVARYINGPOTENTIAL(d_z[idz], t));
}

__device__ void warpReduce(volatile float *sdata, short int tid) {
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

__global__ void reduce(complex *g_idata, real *g_odata) {

	extern __shared__ real sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(2 * blockDim.x) + tid;

	sdata[tid] = g_idata[i].x + g_idata[i + blockDim.x].x;
	__syncthreads();

	for (unsigned int s = blockDim.x/2; s > 32; s >>= 1)
	{
		if (tid < s) { sdata[tid] += sdata[tid + s]; }
		__syncthreads();
	}

	if (tid < 32) warpReduce(sdata, tid);

	if (tid == 0)
	{
		g_odata[blockIdx.x] = sdata[0];
	}

}

__global__ void reduce(real *g_idata, real *g_odata) {

	extern __shared__ real sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(2 * blockDim.x) + tid;

	sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
	{
		if (tid < s) { sdata[tid] += sdata[tid + s]; }
		__syncthreads();
	}

	if (tid < 32) warpReduce(sdata, tid);

	if (tid == 0)
	{
		g_odata[blockIdx.x] = sdata[0];
	}

}