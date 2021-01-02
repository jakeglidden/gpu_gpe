/*****************************************************************\
|	gpeSolver_RK4											      |
|	Author: Jake Glidden (japg2@cam.ac.uk)		                  |
|												                  |
|	A simple code for evaluating the Gross-Pitaevskii equation    |
|	on a CUDA-compatible GPU, with flexible grid parameters and   |
|   easily-modifiable potential terms.							  |
|                                               				  |
|   Variable names preceded by "d_" are memory on the "device"    |
|   (i.e. GPU), whereas variable names preceded by "h_" are on    |
|   the "host" (i.e. CPU). Data must be explicitly copied between |
|   the two and data on the GPU cannot be directly accessed by    |
|   the CPU and vice versa.                                       |
|                                                                 |
|   Usage:                                                        |
|   $ gpeSolver_RK4 [-name value]*                                |
|                                                                 |
|   Arguments:                                                    |
|   -AtomNumber          number of atoms initially                |
|   -ScatteringLength    scattering length in Bohr radii          |
|   -Radius              box radius in micron                     |
|   -Length              box length in micron                     |
|   -Depth               box depth in nK                          |
|   -BoundaryDepth       depth of absorbing boundary in nK        |
|   -BoundaryWidth       fraction of grid used for asborbing      |
|                        boundary, e.g. 0.05 for 5%               |
|   -NX                  number of grid points along x            |
|   -XRange              total span of x grid   [sim for Y,Z]     |
|   -Timestep            computational timestep in millisecond    |
|   -ShakeAmplitude      shake amplitude in nK per micron         |
|   -ShakeFrequency      shake frequency in Hz                    |
|   -SimulationLength    simulation length in ms                  |
|   -ITSimulationLength  imag. time simulation length in ms       |
|   -OutputTimes *       ,-separated list of output times in ms   |
|                        * THESE MUST BE SORTED                   |
|                                                                 |
\*****************************************************************/

#pragma region DEFINES

#define MAX(x,y) \
	((x) > (y) ? (x) : (y))

#define MIN(x,y) \
	((x) < (y) ? (x) : (y))

#define MODSQ(xp) \
	((xp.x)*(xp.x) + (xp.y)*(xp.y))

#define FFTIDXSHIFT(idx, n) \
	(((idx) + (n)/2) % (n))

#define PSI0(x, y, z) \
	exp(-2.0 * (x * x + y * y + z * z) / pow(MIN(simArgs->Radius, 0.5*(simArgs->Length)), 2.0))

#define STATICPOTENTIAL(x, y, z) \
	((simArgs->Depth) * \
	(((x*x + y*y) < ((simArgs->Radius) * (simArgs->Radius)) && \
	(2.0 * abs(z) <= (simArgs->Length))) ? 0.0 : 1.0))

#define TIMEVARYINGPOTENTIAL(z, t) \
	(simArgs.ShakeAmplitude * z * sin(TWOPI * simArgs.ShakeFrequency * t * 1.0e-3))

#define ABS_BOUNDARY(x, y, z) \
	( (simArgs->BoundaryDepth) * \
		MAX(MAX( \
			pow(cosh((1.0f/(simArgs->BoundaryWidth)) * (1.0 - abs(x)/((g->xlim)[1]))), -2.0), \
			pow(cosh((1.0f/(simArgs->BoundaryWidth)) * (1.0 - abs(y)/((g->ylim)[1]))), -2.0)), \
			pow(cosh((1.0f/(simArgs->BoundaryWidth)) * (1.0 - abs(z)/((g->zlim)[1]))), -2.0) \
		) \
	)

#pragma endregion

#include "GPEMatrixOps.h"
#include "kwargparser.h"
#include "Operators.h"

#pragma region GLOBAL VARIABLES

geometry g;

char 
	*outfilename;

real
	*h_K;			  // K^2 matrix

real
	*h_pot,		      // Static trapping potential
	*h_bnd,		      // Absorbing boundaries
	*h_dens;

complex
	*h_psi, *d_psi;   // Wavefunction

char
	timestring[256] = "";

#pragma endregion

int main(int argc, char** argv)
{

#pragma region GENERAL SETUP

	/* Process command-line arguments */
	SimArgs simArgs;
	setDefaultArguments(&simArgs);
	if (setArguments(&simArgs, argc, argv) == -1) {
		printf("!!! %s (line %d) Unable to process command-line arguments. Please check that the arguments have been correctly specified.\n\n", strrchr(__FILE__, '/') + 1, __LINE__);
		exit(-1);
	}

	/* Create filename with timestamp */
	generateFilename(&outfilename);
	printf("Saving output in directory: %s\n\n", outfilename);

	/* Print arguments list in console window and in output file */
	printf("Running simulation with simArgs:\n");
	for (short unsigned int ii = 0; ii < NSIMARGS-1; ii++) {
		printf("\t%20s: %6.3e\n", opts[ii], getArgument(&simArgs, opts[ii]));
		printToLog(outfilename, "\t%20s: %6.3e\n", opts[ii], getArgument(&simArgs, opts[ii]));
	}
	printf("\n\n");
	
	/* Clear GPU in preparation */
	cudaSetDevice(0);
	cudaDeviceReset();

	/* Determine commonly-needed geometrical parameters from grid info and write to output file */
	populateGeometry(&g, &simArgs);
	writeGeometryToLog(outfilename, &g);

#pragma endregion

#pragma region IMAGINARY TIME SETUP

	printf("============================================================\n\n");
	printf("BEGINNING IMAGINARY TIME EVOLUTION\n\n");
	printf("============================================================\n\n");

	/* Initialise psi, K, and potential */
	printf("Initialising variables ...\n");
	initialiseVariablesForImaginaryTime(&h_psi, &h_pot, &h_K, &g, &simArgs);

	/* Prepare GPU */
	printf("Preparing GPU ...\n");
	prepareGPUForImaginaryTime(&d_psi, h_psi, h_pot, h_K, &g);

	/* Renormalise initial state */
	printf("Normalising the starting wavefunction to %7.3e atoms ... ", simArgs.AtomNumber);
	renormalise(d_psi, &g);
	printf("Done!\n\n");

#pragma endregion

#pragma region IMAGINARY TIME EVOLUTION

	/* EVOLVE IN IMAGINARY TIME
	*  I use a procedure called `constant renormalisation', and evolve the GPE with t -> (1.0i)*t.
	*  This means that states decay in time instead of having a rotating global phase, and higher-
	*  energy states decay more rapidly (proportional to their energy). Renormalising gives a
	*  relative boost to the ground state, until we eventually converge. */
	printf("\nBeginning IMAGINARY time evolution ...\n\n");
	for (unsigned long int ii = (unsigned long int) (simArgs.ITSimulationLength / simArgs.Timestep); ii > 0; ii--) {
		
		printf("\r\tt = %+08.1f ms", ii * (-simArgs.Timestep));

		/* RK Step 1 */ RKStep(false, 1, d_psi, &g);
		/* RK Step 2 */ RKStep(false, 2, d_psi, &g);
		/* RK Step 3 */ RKStep(false, 3, d_psi, &g);
		/* RK Step 4 */ RKStep(false, 4, d_psi, &g);
		/* Add Steps */ RKAdd(d_psi, &g);
		/* Renormal. */ renormalise(d_psi, &g);

	}
	printf("\n\nDone!\n\n");

#pragma endregion

#pragma region REAL TIME SETUP

	printf("============================================================\n\n");
	printf("BEGINNING REAL TIME EVOLUTION\n\n");
	printf("============================================================\n\n");

	/* Initialise psi, K, and potential */
	printf("Initialising variables ...\n");
	initialiseVariablesForRealTime(&h_bnd, &g, &simArgs);

	/* Prepare GPU */
	printf("Preparing GPU ...\n");
	prepareGPUForRealTime(h_bnd, &g);

	/* Get first point of time-varying potential */
	MakeV_t_RT<<<g.npar, 128>>>(g.d_pot_t, g.Vprefix, g.d_z, g.nz, 0.0, simArgs);

#pragma endregion

#pragma region REAL TIME EVOLUTION

	/* Evolve in real time */
	long unsigned int *outputTimesteps = NULL;
	unsigned int outputPos = 0;
	long unsigned int nIts = (long unsigned int)(simArgs.SimulationLength / simArgs.Timestep);
	getOutputIndices(&simArgs, &outputTimesteps); // just converting floats to integers to avoid dealing with machine precision differences
	
	printf("\nBeginning REAL time evolution ...\n\n");
	/* LOOP FOR TIMESTEPS */
	for (long unsigned int ii = 0; ii <= nIts; ii++) {

		/* PRINT WHERE WE'RE UP TO */
		printf("\r\tt = %+08.1f ms", ii * simArgs.Timestep);

		/* SAVE IF THE CURRENT TIME IS IN THE LIST OF TIMES TO SAVE */
		if (outputPos < simArgs.NOutputs && ii >= outputTimesteps[outputPos]) {

			// Save in-situ density
			HANDLE_ERROR(cudaMemcpy(h_psi, d_psi, g.npts * sizeof(complex), cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaDeviceSynchronize());
			getDensity(h_psi, &g, &h_dens);
			saveInSituDensity(outfilename, h_dens, 1, &g, ii * simArgs.Timestep); // integrated wrt x
			saveInSituDensity(outfilename, h_dens, 3, &g, ii * simArgs.Timestep); // integrated wrt z

			// Save momentum density
			if (g.d_kpsi == NULL) HANDLE_ERROR(cudaMalloc((void**)&(g.d_kpsi), g.npts * sizeof(complex)));
			cufftExecC2C(g.fftPlan, d_psi, g.d_kpsi, CUFFT_FORWARD);
			real renormfac = (real)sqrt(1.0 / ((double)(g.npts)));
			cublasCsscal(g.cbHandle, g.npts, &renormfac, g.d_kpsi, 1);
			HANDLE_ERROR(cudaMemcpy(h_psi, g.d_kpsi, g.npts * sizeof(complex), cudaMemcpyDeviceToHost));
			HANDLE_ERROR(cudaDeviceSynchronize());
			getDensity(h_psi, &g, &h_dens);
			saveKDensity(outfilename, h_dens, 1, &g, ii * simArgs.Timestep);  // integrated wrt kx
			saveKDensity(outfilename, h_dens, 3, &g, ii * simArgs.Timestep);  // integrated wrt kz

			saveFullKWavefunction(outfilename, h_psi, &g, ii * simArgs.Timestep);

			outputPos++;
			printf("\r\tt = %+08.1f ms\n", ii * simArgs.Timestep);
		}

		/* STOP IF WE HAVE REACHED THE PRE-COMPUTED TOTAL NUMBER OF ITERATIONS */
		if (ii == nIts) break;

		/* USE RUNGE-KUTTA ALGORITHM TO INTEGRATE THE GPE FOR ONE TIMESTEP */
		/* RK Step 1 */ RKStep(true, 1, d_psi, &g);  // ESTIMATE FROM DERIVATIVES/POTENTIAL AT START OF TIMESTEP
		/* Correct V */ MakeV_t_RT<<<g.npar, 128>>>(g.d_pot_t, g.Vprefix, g.d_z, g.nz, ((real)ii + 0.5f) * simArgs.Timestep, simArgs);
		/* RK Step 2 */ RKStep(true, 2, d_psi, &g);  // ESTIMATE FROM DERIVATIVES/POTENTIAL AT MIDPOINT OF TIMESTEP
		/* RK Step 3 */ RKStep(true, 3, d_psi, &g);  // ESTIMATE FROM DERIVATIVES/POTENTIAL AT MIDPOINT OF TIMESTEP, ESTIMATED USING PREVIOUS STEP
		/* Correct V */ MakeV_t_RT<<<g.npar, 128>>>(g.d_pot_t, g.Vprefix, g.d_z, g.nz, ((real)ii + 1.0f) * simArgs.Timestep, simArgs);
		/* RK Step 4 */ RKStep(true, 4, d_psi, &g);  // ESTIMATE FROM DERIVATIVES/POTENTIAL AT END OF TIMESTEP
		/* Add Steps */ RKAdd(d_psi, &g);

	/* END FOR LOOP OVER TIMESTEPS */
	}

	printf("\nDone!\n\n");

#pragma endregion

#pragma region GENERAL CLEANUP

	depopulateGeometry(&g);

	/* Clean up on GPU */
	if (d_psi != NULL) { HANDLE_ERROR(cudaFree(d_psi)); d_psi = NULL; }

	/* Clean up on CPU */
	if (simArgs.OutputTimes != NULL) { free(simArgs.OutputTimes); simArgs.OutputTimes = NULL; }
	if (outfilename != NULL) { free(outfilename); outfilename = NULL; }
	if (outputTimesteps != NULL) { free(outputTimesteps); outputTimesteps = NULL; }
	if (h_K != NULL) { free(h_K); h_K = NULL;  }
	if (h_pot != NULL) { free(h_pot); h_pot = NULL; }
	if (h_psi != NULL) { free(h_psi); h_psi = NULL; }
	if (outfilename != NULL) { free(outfilename); outfilename = NULL; }
	if (h_dens != NULL) { free(h_dens); h_dens = NULL; }

    return 0;

#pragma endregion

}