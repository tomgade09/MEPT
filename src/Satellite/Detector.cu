#include <stdexcept>

//CUDA includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

//Project specific includes
#include "Satellite/Satellite.h"
#include "ErrorHandling/cudaErrorCheck.h"

using std::invalid_argument;

__global__ void satelliteDetector(float** data_d, float** capture_d, float simtime, float dt, float altitude, bool upward)
{
	unsigned int thdInd{ blockIdx.x * blockDim.x + threadIdx.x };

	float* detected_t_d{ capture_d[3] };    //do this first before creating a bunch of pointers

	if (simtime == 0.0f) //not sure I fully like this, but it works
	{
		float* detected_ind_d{ capture_d[4] };
		detected_t_d[thdInd] = -1.0f;
		detected_ind_d[thdInd] = -1.0f;
	}

	//guard to prevent unnecessary variable creation, if condition checks
	if (detected_t_d[thdInd] > -0.1f) return; //if the slot in detected_t[thdInd] is filled (gt or equal to 0), return

	const float* v_d{ data_d[0] }; // do not like this way of doing it - no bounds checking due to c-style arrays
	const float* s_d{ data_d[2] };
	const float* s0_d{ data_d[5] };
	const float* v0_d{ data_d[6] };

	if (//no detected particle is in the data array at the thread's index already AND
		((!upward) && (s_d[thdInd] >= altitude) && (s0_d[thdInd] < altitude)) //detector is facing down and particle crosses altitude in dt
		|| //OR
		((upward) && (s_d[thdInd] <= altitude) && (s0_d[thdInd] > altitude)) //detector is facing up and particle crosses altitude in dt
		)
	{
		const float* mu_d{ data_d[1] };

		float* detected_v_d{ capture_d[0] }; float* detected_mu_d{ capture_d[1] };
		float* detected_s_d{ capture_d[2] }; float* detected_ind_d{ capture_d[4] };
		//
		//
		// TEST CODE TO REMOVE UPGOING PARTICLES IN DOWNGOING DETECTOR
		const float s_ratio{ (altitude - s0_d[thdInd]) / (s_d[thdInd] - s0_d[thdInd]) };
		const float interpol_v_d{ v0_d[thdInd] + (v_d[thdInd] - v0_d[thdInd]) * s_ratio }; //interpolates v back to where the particle was detected
		const float interpol_t_d{ simtime + dt * s_ratio };
		//
		//
		//
		detected_v_d[thdInd] = interpol_v_d;//v_d[thdInd];
		detected_mu_d[thdInd] = mu_d[thdInd];
		detected_s_d[thdInd] = altitude;//s_d[thdInd];
		detected_t_d[thdInd] = interpol_t_d;
		detected_ind_d[thdInd] = static_cast<float>(thdInd);
	}//particle not removed from sim
}

void Satellite::iterateDetector(float simtime, float dt, int blockSize, int GPUind)
{
	if (numberOfParticles_m % blockSize != 0)
		throw invalid_argument("Satellite::iterateDetector: numberOfParticles is not a whole multiple of blocksize, some particles will not be checked");

	//a;
	//need to adjust pointers in the below kernel call to refer to specific GPU's memory spaces that are set up within each satellite
	//refer to Particles for examples:
	//In Particles.h:
	//members:
	//	size_t numGPUs_m;
	//	vector<size_t> particleCountPerGPU_m;
	//
	//	constructor takes in number of GPUS and distribution of particles per GPU
	//	
	//In Particles.cpp
	//	initializeGPU() - iterates over number of GPUs
	//	copyDataToGPU() - uses offset and end to create a small subset vector of the total data and copy that to GPU
	//	copyDataToHost() - uses offset and end to create a small empty vector that is passed in to copy data back to host
	//	freeGPUMemory() - frees all devices' GPU mem regions
	CUDA_API_ERRCHK(cudaSetDevice(GPUind));
	satelliteDetector<<< numberOfParticles_m / blockSize, blockSize >>>(particleData2D_d.at(GPUind), satCaptrData2D_d.at(GPUind), simtime, dt, altitude_m, upwardFacing_m);
	CUDA_KERNEL_ERRCHK_WSYNC();
}