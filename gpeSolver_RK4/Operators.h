#pragma once

void getDensity(const complex *psi, const geometry *g, real **dens) {
	if (*dens != NULL) free(*dens);
	*dens = (real *)malloc((g->npts) * sizeof(real));
	for (unsigned int idx = 0; idx < g->npts; idx++)
	{
		(*dens)[idx] = MODSQ(psi[idx]);
	}
}

void __global__ MultK2(complex *kpsi, const real *ksq) {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	kpsi[idx].x = (kpsi[idx].x * kpsi[idx].x + kpsi[idx].y * kpsi[idx].y) * ksq[idx];
	kpsi[idx].y = 0.0f;
}

void __global__ GetDens(real *T, const complex *kpsi) {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	T[idx] = kpsi[idx].x * kpsi[idx].x + kpsi[idx].y * kpsi[idx].y;
}

void __global__ mod2(complex *kpsi) {
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
	kpsi[idx].x = kpsi[idx].x * kpsi[idx].x + kpsi[idx].y * kpsi[idx].y;
	kpsi[idx].y = 0.0f;
}

void saveInSituDensity(char *basefilename, real *dens_3d, int integrate_dimension, geometry *g, real t) {
	char outfilenameWithTime[256];
	char timestring[256];
	sprintf(timestring, "\\I_dens_d%c.t_%+010.3e.csv", integrate_dimension==1?'x':'z', t);
	strncpy(outfilenameWithTime, basefilename, strrchr(basefilename, '\\') - basefilename);
	outfilenameWithTime[strrchr(basefilename, '\\') - basefilename] = '\0'; // need to null terminate otherwise the following 'cat' etc. don't work properly
	strcat(outfilenameWithTime, timestring);
	FILE * f = fopen(outfilenameWithTime, "w");

	real dens = 0.0;

	// INTEGRATING OVER X
	if (integrate_dimension == 1) {
		for (unsigned int idy = 0; idy < g->ny; idy++) {
			dens = 0.0;
			for (unsigned int idx = 0; idx < g->nx; idx++) {
				dens += dens_3d[g->nz * (g->ny*idx + idy)];
			}
			fprintf(f, "%+0.7e", dens);
			for (unsigned int idz = 1; idz < g->nz; idz++) {
				dens = 0.0;
				for (unsigned int idx = 0; idx < g->nx; idx++) {
					dens += dens_3d[g->nz * (g->ny*idx + idy) + idz];
				}
				fprintf(f, ",%+0.7e", dens * g->dx);
			}
			fprintf(f, "\n");
		}
	} else {

	// INTEGRATING OVER Z
		for (unsigned int idx = 0; idx < g->nx; idx++) {
			dens = 0.0;
			for (unsigned int idz = 0; idz < g->nz; idz++) {
				dens += dens_3d[g->nz * (g->ny*idx) + idz];
			}
			fprintf(f, "%+0.7e", dens);
			for (unsigned int idy = 1; idy < g->ny; idy++) {
				dens = 0.0;
				for (unsigned int idz = 0; idz < g->nz; idz++) {
					dens += dens_3d[g->nz * (g->ny*idx + idy) + idz];
				}
				fprintf(f, ",%+0.7e", dens * g->dz);
			}
			fprintf(f, "\n");
		}
	}
	fclose(f);
}

void saveKDensity(char *basefilename, real *dens_3d, int integrate_dimension, geometry *g, real t) {
	char outfilenameWithTime[256];
	char timestring[256];
	sprintf(timestring, "\\I_kdens_dk%c.t_%+010.3e.csv", integrate_dimension == 1 ? 'x' : 'z', t);
	strncpy(outfilenameWithTime, basefilename, strrchr(basefilename, '\\') - basefilename);
	outfilenameWithTime[strrchr(basefilename, '\\') - basefilename] = '\0'; // need to null terminate otherwise the following 'cat' etc. don't work properly
	strcat(outfilenameWithTime, timestring);
	FILE * f = fopen(outfilenameWithTime, "w");

	double dens = 0.0;

	// INTEGRATING OVER KX
	// We need to permute indices to move the fourier transform 0 component to the centre (using FFTIDXSHIFT macro)
	if (integrate_dimension == 1) {
		for (unsigned int idy = 0; idy < g->ny; idy++) {
			unsigned int idy2 = FFTIDXSHIFT(idy, g->ny);
			dens = 0.0;
			for (unsigned int idx = 0; idx < g->nx; idx++) {
				unsigned int idx2 = FFTIDXSHIFT(idx, g->nx);
				dens += dens_3d[g->nz * (g->ny*idx2 + idy2) + FFTIDXSHIFT(0, g->nz)];
			}
			fprintf(f, "%+0.7e", dens * g->dx * ((g->dy/g->dky) * (g->dz/g->dkz)));
			for (unsigned int idz = 1; idz < g->nz; idz++) {
				unsigned int idz2 = FFTIDXSHIFT(idz, g->nz);
				dens = 0.0;
				for (unsigned int idx = 0; idx < g->nx; idx++) {
					unsigned int idx2 = FFTIDXSHIFT(idx, g->nx);
					dens += dens_3d[g->nz * (g->ny*idx2 + idy2) + idz2];
				}
				// Scaling factors account for the Jacobian for the transformation between coordinate- and k-space.
				fprintf(f, ",%+0.7e", dens * g->dx * ((g->dy / g->dky) * (g->dz / g->dkz)));
			}
			fprintf(f, "\n");
		}
	} else {
	// INTEGRATING OVER KZ
	// We need to permute indices to move the fourier transform 0 component to the centre (using FFTIDXSHIFT macro)
		for (unsigned int idx = 0; idx < g->nx; idx++) {
			unsigned int idx2 = FFTIDXSHIFT(idx, g->nx);
			dens = 0.0;
			for (unsigned int idz = 0; idz < g->nz; idz++) {
				unsigned int idz2 = FFTIDXSHIFT(idz, g->nz);
				dens += dens_3d[g->nz * (g->ny*idx2 + FFTIDXSHIFT(0, g->nz)) + idz2];
			}
			fprintf(f, "%+0.7e", dens * g->dz * ((g->dy / g->dky) * (g->dx / g->dkx)));
			for (unsigned int idy = 1; idy < g->ny; idy++) {
				unsigned int idy2 = FFTIDXSHIFT(idy, g->ny);
				dens = 0.0;
				for (unsigned int idz = 0; idz < g->nz; idz++) {
					unsigned int idz2 = FFTIDXSHIFT(idz, g->nz);
					dens += dens_3d[g->nz * (g->ny*idx2 + idy2) + idz2];
				}
				// Scaling factors account for the Jacobian for the transformation between coordinate- and k-space.
				fprintf(f, ",%+0.7e", dens * g->dz * ((g->dy / g->dky) * (g->dx / g->dkx)));
			}
			fprintf(f, "\n");
		}
	}
	fclose(f);
}

void saveFullKWavefunction(char *basefilename, complex *kpsi, geometry *g, real t) {
	char outfilenameWithTime[256];
	char timestring[256];
	sprintf(timestring, "\\kpsi.t_%+010.3e.bin", t);
	strncpy(outfilenameWithTime, basefilename, strrchr(basefilename, '\\') - basefilename);
	outfilenameWithTime[strrchr(basefilename, '\\') - basefilename] = '\0'; // need to null terminate otherwise the following 'cat' etc. don't work properly
	strcat(outfilenameWithTime, timestring);
	FILE * f = fopen(outfilenameWithTime, "wb");
	fwrite(kpsi, sizeof(complex), g->npts, f);
	fclose(f);
}