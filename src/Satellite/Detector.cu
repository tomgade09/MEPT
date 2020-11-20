#include <stdexcept>

//CUDA includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

//Project specific includes
#include "Satellite/Satellite.h"
#include "ErrorHandling/cudaErrorCheck.h"

using std::invalid_argument;

__global__ void satelliteDetector(double** data_d, double** capture_d, double simtime, double dt, double altitude, bool upward)
{
	unsigned int thdInd{ blockIdx.x * blockDim.x + threadIdx.x };

	double* detected_t_d{ capture_d[3] };    //do this first before creating a bunch of pointers

	if (simtime == 0.0) //not sure I fully like this, but it works
	{
		double* detected_ind_d{ capture_d[4] };
		detected_t_d[thdInd] = -1.0;
		detected_ind_d[thdInd] = -1.0;
	}

	//guard to prevent unnecessary variable creation, if condition checks
	if (detected_t_d[thdInd] > -0.1) return; //if the slot in detected_t[thdInd] is filled (gt or equal to 0), return

	const double* v_d{ data_d[0] }; // do not like this way of doing it - no bounds checking due to c-style arrays
	const double* s_d{ data_d[2] };
	const double* s0_d{ data_d[5] };
	const double* v0_d{ data_d[6] };

	if (//no detected particle is in the data array at the thread's index already AND
		((!upward) && (s_d[thdInd] >= altitude) && (s0_d[thdInd] < altitude)) //detector is facing down and particle crosses altitude in dt
		|| //OR
		((upward) && (s_d[thdInd] <= altitude) && (s0_d[thdInd] > altitude)) //detector is facing up and particle crosses altitude in dt
		)
	{
		const double* mu_d{ data_d[1] };

		double* detected_v_d{ capture_d[0] }; double* detected_mu_d{ capture_d[1] };
		double* detected_s_d{ capture_d[2] }; double* detected_ind_d{ capture_d[4] };
		//
		//
		// TEST CODE TO REMOVE UPGOING PARTICLES IN DOWNGOING DETECTOR
		const double s_ratio{ (altitude - s0_d[thdInd]) / (s_d[thdInd] - s0_d[thdInd]) };
		const double interpol_v_d{ v0_d[thdInd] + (v_d[thdInd] - v0_d[thdInd]) * s_ratio }; //interpolates v back to where the particle was detected
		const double interpol_t_d{ simtime + dt * s_ratio };
		//
		//
		//
		detected_v_d[thdInd] = interpol_v_d;//v_d[thdInd];
		detected_mu_d[thdInd] = mu_d[thdInd];
		detected_s_d[thdInd] = altitude;//s_d[thdInd];
		detected_t_d[thdInd] = interpol_t_d;
		detected_ind_d[thdInd] = static_cast<double>(thdInd);
	}//particle not removed from sim
}

void Satellite::iterateDetector(double simtime, double dt, int blockSize)
{
	if (numberOfParticles_m % blockSize != 0)
		throw invalid_argument("Satellite::iterateDetector: numberOfParticles is not a whole multiple of blocksize, some particles will not be checked");

	satelliteDetector<<< numberOfParticles_m / blockSize, blockSize >>>(particleData2D_d, satCaptrData2D_d, simtime, dt, altitude_m, upwardFacing_m);
	CUDA_KERNEL_ERRCHK_WSYNC();
}