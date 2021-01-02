#define NSIMARGS 20
#include <sys/stat.h>
#include <Windows.h>

const char* opts[] = { "-AtomNumber", "-ScatteringLength", "-Radius", "-Length", "-Depth", "-Timestep",
  "-ShakeAmplitude", "-ShakeFrequency", "-SimulationLength", "-ITSimulationLength", 
  "-NX", "-XRange", "-NY", "-YRange", "-NZ", "-ZRange", "-BoundaryDepth", "-BoundaryWidth",
  "-NOutputs", "-OutputTimes" };

int setArguments(SimArgs *sa, int argc, char **argv) {
	if (argc % 2 != 1) {
		printf("Incorrect number of arguments passed to function. Aborting.\n");
		return -1;
	}
	for (int i = 1; i < argc; i += 2) {
		if (!strcmp(argv[i],  opts[0])) sa->AtomNumber = (real)atof(argv[i + 1]);
		if (!strcmp(argv[i],  opts[1])) sa->ScatteringLength = (real)atof(argv[i + 1]);
		if (!strcmp(argv[i],  opts[2])) sa->Radius = (real)atof(argv[i + 1]);
		if (!strcmp(argv[i],  opts[3])) sa->Length = (real)atof(argv[i + 1]);
		if (!strcmp(argv[i],  opts[4])) sa->Depth = (real)atof(argv[i + 1]);
		if (!strcmp(argv[i],  opts[5])) sa->Timestep = (real)atof(argv[i + 1]);
		if (!strcmp(argv[i],  opts[6])) sa->ShakeAmplitude = (real)atof(argv[i + 1]);
		if (!strcmp(argv[i],  opts[7])) sa->ShakeFrequency = (real)atof(argv[i + 1]);
		if (!strcmp(argv[i],  opts[8])) sa->SimulationLength = (real)atof(argv[i + 1]);
		if (!strcmp(argv[i],  opts[9])) sa->ITSimulationLength = (real)atof(argv[i + 1]);
		if (!strcmp(argv[i], opts[10])) sa->NX = (unsigned int)atoi(argv[i + 1]);
		if (!strcmp(argv[i], opts[11])) sa->XRange = (real)atof(argv[i + 1]);
		if (!strcmp(argv[i], opts[12])) sa->NY = (unsigned int)atoi(argv[i + 1]);
		if (!strcmp(argv[i], opts[13])) sa->YRange = (real)atof(argv[i + 1]);
		if (!strcmp(argv[i], opts[14])) sa->NZ = (unsigned int)atoi(argv[i + 1]);
		if (!strcmp(argv[i], opts[15])) sa->ZRange = (real)atof(argv[i + 1]);
		if (!strcmp(argv[i], opts[16])) sa->BoundaryDepth = (real)atof(argv[i + 1]);
		if (!strcmp(argv[i], opts[17])) sa->BoundaryWidth = (real)atof(argv[i + 1]);
		if (!strcmp(argv[i], opts[18])) sa->NOutputs = (unsigned int)atoi(argv[i + 1]);
		if (!strcmp(argv[i], opts[19])) {
			// reset & reallocate the memory
			if (sa->OutputTimes != NULL) free(sa->OutputTimes);
			sa->OutputTimes = (real*)malloc(sa->NOutputs * sizeof(real));
			// write each time
			char *currpos = NULL;
			char *prevpos = argv[i + 1];
			char tmpstring[256];
			for (unsigned int j = 0; j < sa->NOutputs; j++)
			{
				currpos = strchr(prevpos, ',');
				if (currpos != NULL)
				{
					strncpy(tmpstring, prevpos, currpos - prevpos);
					tmpstring[currpos - prevpos] = '\0';
				} else strcpy(tmpstring, prevpos);
				(sa->OutputTimes)[j] = (real)atof(tmpstring);
				prevpos = currpos + 1;
			}
		}
	}
	return 0;
}

real getArgument(SimArgs *sa, const char *argName) {
	if (!strcmp(argName, opts[0])) return (sa->AtomNumber);
	if (!strcmp(argName, opts[1])) return (sa->ScatteringLength);
	if (!strcmp(argName, opts[2])) return (sa->Radius);
	if (!strcmp(argName, opts[3])) return (sa->Length);
	if (!strcmp(argName, opts[4])) return (sa->Depth);
	if (!strcmp(argName, opts[5])) return (sa->Timestep);
	if (!strcmp(argName, opts[6])) return (sa->ShakeAmplitude);
	if (!strcmp(argName, opts[7])) return (sa->ShakeFrequency);
	if (!strcmp(argName, opts[8])) return (sa->SimulationLength);
	if (!strcmp(argName, opts[9])) return (sa->ITSimulationLength);
	if (!strcmp(argName, opts[10])) return (real)(sa->NX);
	if (!strcmp(argName, opts[11])) return (sa->XRange);
	if (!strcmp(argName, opts[12])) return (real)(sa->NY);
	if (!strcmp(argName, opts[13])) return (sa->YRange);
	if (!strcmp(argName, opts[14])) return (real)(sa->NZ);
	if (!strcmp(argName, opts[15])) return (sa->ZRange);
	if (!strcmp(argName, opts[16])) return (real)(sa->BoundaryDepth);
	if (!strcmp(argName, opts[17])) return (sa->BoundaryWidth);
	if (!strcmp(argName, opts[18])) return (real)(sa->NOutputs);
	return NAN;
}

void setDefaultArguments(SimArgs *sa) {
	sa->AtomNumber			= 1.00e+5f;
	sa->ScatteringLength	= 2.00e+2f;
	sa->Radius				= 1.50e+1f;
	sa->Length				= 5.00e+1f;
	sa->Depth				= 3.00e+2f;
	sa->Timestep			= 1.00e-2f;
	sa->ShakeAmplitude		= 2.00e-1f;
	sa->ShakeFrequency		= 2.00e+1f;
	sa->SimulationLength	= 1.00e+3f;
	sa->ITSimulationLength	= 2.00e+1f;
	sa->NX					= 128;
	sa->NY					= 128;
	sa->NZ					= 128;
	sa->XRange				= 100.0;
	sa->YRange				= 100.0;
	sa->ZRange				= 100.0;
	sa->BoundaryDepth       = 2.00e+1f;
	sa->BoundaryWidth       = 0.05f;
	sa->NOutputs			= 2;
	// reset and reallocate the memory
	if (sa->OutputTimes != NULL) free(sa->OutputTimes);
	sa->OutputTimes = (real*)malloc(sa->NOutputs * sizeof(real));
	// write each time
	(sa->OutputTimes)[0] = 0.0;
	(sa->OutputTimes)[1] = sa->SimulationLength;
}

void generateFilename(char **ofile) {
  time_t currentTime = time(NULL);
  tm *sTime = localtime(&currentTime);
  *ofile = (char *)malloc(256 * sizeof(char));
  if (*ofile == NULL) printf("!!! %s (line %d) Error allocating memory for filename.\n", strrchr(__FILE__, '/')+1, __LINE__);
  struct stat b;
  if (!(stat("Outputs", &b) == 0 && (S_IFDIR & b.st_mode))) CreateDirectory("Outputs", NULL);
  strftime(*ofile, 256, "Outputs\\gpeSolver_RK4-%Y-%m-%dT%H-%M-%S", sTime);
  CreateDirectory(*ofile, NULL);
  strcat(*ofile, "\\");
}

void initialiseVariablesForImaginaryTime(complex **psi, real **pot, real **K, geometry *g, SimArgs *simArgs) {

	*K = (real *)malloc(g->npts * sizeof(real));
	if (*K == NULL) printf("!!! %s:%s (line %d) Error allocating CPU memory.\n", strrchr(__FILE__, '/')+1, __FUNCTION__, __LINE__);
	*psi = (complex *)malloc(g->npts * sizeof(complex));
	if (*psi == NULL) printf("!!! %s:%s (line %d) Error allocating CPU memory.\n", strrchr(__FILE__, '/')+1, __FUNCTION__, __LINE__);
	*pot = (real *)malloc(g->npts * sizeof(real));
	if (*pot == NULL) printf("!!! %s:%s (line %d) Error allocating CPU memory.\n", strrchr(__FILE__, '/')+1, __FUNCTION__, __LINE__);

	for (short unsigned int idx = 0; idx < g->nx; idx++) {
		for (short unsigned int idy = 0; idy < g->ny; idy++) {
			for (short unsigned int idz = 0; idz < g->nz; idz++) {
				(*psi)[g->nz * (g->ny*idx + idy) + idz] = make_cuComplex((real)(PSI0((g->x[idx]), (g->y[idy]), (g->z[idz]))), 0.0f);
				(*pot)[g->nz * (g->ny*idx + idy) + idz] = (real)((g->Vprefix)*(STATICPOTENTIAL((g->x[idx]), (g->y[idy]), (g->z[idz]))));
				  (*K)[(g->nz) * ((g->ny)*idx + idy) + idz] = (real)((g->kx[idx] * g->kx[idx] + g->ky[idy] * g->ky[idy] + g->kz[idz] * g->kz[idz]) * g->Tprefix) * ((real)(1.0 / g->npts));
			}
		}
	}
}

void initialiseVariablesForRealTime(real **bnd, geometry *g, SimArgs *simArgs) {

	if (*bnd != NULL) free(*bnd);
	*bnd = (real *)malloc(g->npts * sizeof(real));
	if (*bnd == NULL) printf("!!! %s:%s (line %d) Error allocating CPU memory.\n", strrchr(__FILE__, '/')+1, __FUNCTION__, __LINE__);

	for (short unsigned int idx = 0; idx < g->nx; idx++) {
		for (short unsigned int idy = 0; idy < g->ny; idy++) {
			for (short unsigned int idz = 0; idz < g->nz; idz++) {
				(*bnd)[g->nz * (g->ny*idx + idy) + idz] = (real)((g->Vprefix)*(ABS_BOUNDARY((g->x[idx]), (g->y[idy]), (g->z[idz]))));
			}
		}
	}
}

void prepareGPUForImaginaryTime(complex **d_psi, const complex *h_psi, const real *h_pot, const real *h_K, geometry *g) {

	/* Allocate GPU memory */
	HANDLE_ERROR(cudaMalloc((void**)&(*d_psi), g->npts * sizeof(complex)));
	HANDLE_ERROR(cudaMalloc((void**)&(g->d_dpsi1), g->npts * sizeof(complex)));
	HANDLE_ERROR(cudaMalloc((void**)&(g->d_dpsi2), g->npts * sizeof(complex)));
	HANDLE_ERROR(cudaMalloc((void**)&(g->d_dpsi3), g->npts * sizeof(complex)));
	HANDLE_ERROR(cudaMalloc((void**)&(g->d_dpsi4), g->npts * sizeof(complex)));
	HANDLE_ERROR(cudaMalloc((void**)&(g->ksq), g->npts * sizeof(real)));
	HANDLE_ERROR(cudaMalloc((void**)&(g->d_pot), g->npts * sizeof(real)));

	printf("Copying data to the GPU...\n");

	/* Copy the host arrays to the GPU */
	HANDLE_ERROR(cudaMemcpy(*d_psi, h_psi, g->npts * sizeof(complex), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(g->ksq, h_K, g->npts * sizeof(real), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(g->d_pot, h_pot, g->npts * sizeof(real), cudaMemcpyHostToDevice));
}

void prepareGPUForRealTime(const real *h_bnd, geometry *g) {

	/* Allocate GPU memory */
	HANDLE_ERROR(cudaMalloc((void**)&(g->d_z), g->nz * sizeof(real)));
	HANDLE_ERROR(cudaMalloc((void**)&(g->d_bnd), g->npts * sizeof(real)));
	HANDLE_ERROR(cudaMalloc((void**)&(g->d_pot_t), g->npts * sizeof(real)));

	printf("Copying new data to the GPU...\n");
	/* Copy the host arrays to the GPU */
	HANDLE_ERROR(cudaMemcpy(g->d_z, g->z, g->nz * sizeof(real), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(g->d_bnd, h_bnd, g->npts * sizeof(real), cudaMemcpyHostToDevice));
}

void getOutputIndices(SimArgs *sa, long unsigned int **outputTimesteps) {
	if (*outputTimesteps != NULL) free(*outputTimesteps);
	*outputTimesteps = (long unsigned int *)malloc((sa->NOutputs) * sizeof(long unsigned int));
	for (unsigned int i = 0; i < sa->NOutputs; i++)
	{
		(*outputTimesteps)[i] = (long unsigned int)fabs(floor((sa->OutputTimes)[i] / (sa->Timestep)));
	}
}

void printToLog(const char *fn, const char *fs, const char *argName, real argVal) {
	FILE * exportf;
	char asciiout[256];
	const char *startoffilename = strrchr(fn, '\\');
	strncpy(asciiout, fn, (startoffilename - fn));
	asciiout[startoffilename-fn] = '\0';
	strcat(asciiout, "\\Log.txt");

	struct stat buffer;
	if (stat(asciiout, &buffer) != 0) {
		exportf = fopen(asciiout, "w");
		fprintf(exportf, "== Parameters\n\n");
	}
	else exportf = fopen(asciiout, "a");
	fprintf(exportf, fs, argName, argVal);
	fclose(exportf);
}

void writeGeometryToLog(char *fn, geometry *g) {
	FILE * exportf;
	char asciiout[256];
	const char *startoffilename = strrchr(fn, '\\');
	strncpy(asciiout, fn, startoffilename-fn);
	asciiout[startoffilename-fn] = '\0';
	strcat(asciiout, "\\Log.txt");
	exportf = fopen(asciiout,"a");
	fprintf(exportf, "\n\n== Geometry\n\n");

	fprintf(exportf, "\t   nx = %u\n", g->nx);
	fprintf(exportf, "\t xmin = %+.6e um\n", g->xlim[0]);
	fprintf(exportf, "\t xmax = %+.6e um\n\n", g->xlim[1]);
	fprintf(exportf, "\t   ny = %u\n", g->ny);
	fprintf(exportf, "\t ymin = %+.6e um\n", g->ylim[0]);
	fprintf(exportf, "\t ymax = %+.6e um\n\n", g->ylim[1]);
	fprintf(exportf, "\t   nz = %u\n", g->nz);
	fprintf(exportf, "\t zmin = %+.6e um\n", g->zlim[0]);
	fprintf(exportf, "\t zmax = %+.6e um\n\n", g->zlim[1]);

	// Print out the vectors into the log file
	fprintf(exportf, "\n\n== Vectors\n\n");

	// Coordinate vectors
	fprintf(exportf, "x\n");
	fprintf(exportf, "%+010.3e", g->x[0]);
	for (unsigned int idx = 1; idx < g->nx; idx++) fprintf(exportf, ",%+010.3e", g->x[idx]);
	fprintf(exportf, "\ny\n");
	fprintf(exportf, "%+010.3e", g->y[0]);
	for (unsigned int idx = 1; idx < g->ny; idx++) fprintf(exportf, ",%+010.3e", g->y[idx]);
	fprintf(exportf, "\nz\n");
	fprintf(exportf, "%+010.3e", g->z[0]);
	for (unsigned int idx = 1; idx < g->nz; idx++) fprintf(exportf, ",%+010.3e", g->z[idx]);

	// K vectors -- need to rotate indices again to put things into a convenient order
	fprintf(exportf, "\n\nkx\n");
	fprintf(exportf, "%+010.3e", g->kx[FFTIDXSHIFT(0,g->nx)]);
	for (unsigned int idx = 1; idx < g->nx; idx++) fprintf(exportf, ",%+010.3e", g->kx[FFTIDXSHIFT(idx, g->nx)]);
	fprintf(exportf, "\nky\n");
	fprintf(exportf, "%+010.3e", g->ky[FFTIDXSHIFT(0, g->ny)]);
	for (unsigned int idx = 1; idx < g->ny; idx++) fprintf(exportf, ",%+010.3e", g->ky[FFTIDXSHIFT(idx, g->ny)]);
	fprintf(exportf, "\nkz\n");
	fprintf(exportf, "%+010.3e", g->kz[FFTIDXSHIFT(0, g->nz)]);
	for (unsigned int idx = 1; idx < g->nz; idx++) fprintf(exportf, ",%+010.3e", g->kz[FFTIDXSHIFT(idx, g->nz)]);

	fclose(exportf);
}