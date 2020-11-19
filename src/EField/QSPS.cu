#include "EField/QSPS.h"

#include <filesystem>

#include "device_launch_parameters.h"
#include "utils/serializationHelpers.h"
#include "ErrorHandling/cudaErrorCheck.h"
#include "ErrorHandling/cudaDeviceMacros.h"

using std::cerr;
using std::to_string;
using std::invalid_argument;
using namespace utils::fileIO::serialize;

namespace QSPS_d
{
	__global__ void setupEnvironment_d(EModel** qsps, meters* altMin, meters* altMax, double* magnitude, int numRegions)
	{
		ZEROTH_THREAD_ONLY((*qsps) = new QSPS(altMin, altMax, magnitude, numRegions)); //this overloaded constructor is only compiled in the case where __CUDA_ARCH__ is defined
	}

	__global__ void deleteEnvironment_d(EModel** qsps)
	{
		ZEROTH_THREAD_ONLY(delete (*((QSPS**)qsps)));
	}
}

#ifndef __CUDA_ARCH__ //host code
__host__ const vector<meters>& QSPS::altMin() const
{
	return altMin_m;
}

__host__ const vector<meters>& QSPS::altMax() const
{
	return altMax_m;
}

__host__ const vector<double>& QSPS::magnitude() const 
{
	return magnitude_m;
}
#endif


__host__ QSPS::QSPS(meters altMin, meters altMax, Vperm magnitude, int stepUpRegions) : EModel(Type::QSPS)
{
	#ifndef __CUDA_ARCH__ //host code
	altMin_m.push_back(altMin);       //unfortunately this wrapping is necessary
	altMax_m.push_back(altMax);       //as the vectors above also have to be wrapped
	magnitude_m.push_back(magnitude); //in an ifndef/endif block so this will compile

	//step up regions allow the QSPS to gradually step up to full magnitude
	//this avoids a "hard edge" to the QSPS, potentially leading to errors
	if (stepUpRegions != 0)
	{
		constexpr ratio suSize{ 0.05 }; //use step up regions = 5% of QSPS size (arbitrary)

		altMin_m.resize(2 * stepUpRegions + 1); //step up regions on either side of the QSPS, as well as the QSPS itself
		altMax_m.resize(2 * stepUpRegions + 1);
		magnitude_m.resize(2 * stepUpRegions + 1);

		altMin_m.at(stepUpRegions) = altMin; //middle index
		altMax_m.at(stepUpRegions) = altMax;
		magnitude_m.at(stepUpRegions) = magnitude;

		meters size{ altMax - altMin };
		
		for (int iii = 0; iii < stepUpRegions; iii++)
		{//step up regions starting with the bottom of the QSPS ranging to the top
			altMin_m.at(iii) = altMin - (stepUpRegions - iii) * suSize * size;
			altMax_m.at(iii) = altMin - (stepUpRegions - iii - 1) * suSize * size;
			magnitude_m.at(iii) = magnitude * (iii + 1) / (stepUpRegions + 1);
			altMin_m.at(stepUpRegions + 1 + iii) = altMax + iii * suSize * size;
			altMax_m.at(stepUpRegions + 1 + iii) = altMax + (iii + 1) * suSize * size;
			magnitude_m.at(stepUpRegions + 1 + iii) = magnitude * (stepUpRegions - iii) / (stepUpRegions + 1);
		}
	}

	if (useGPU_m) setupEnvironment();
	#endif /* !__CUDA_ARCH__ */
}

__host__ QSPS::QSPS(ifstream& in) : EModel(Type::QSPS)
{
	deserialize(in);
	if (useGPU_m) setupEnvironment();
}

__device__ QSPS::QSPS(meters* altMin, meters* altMax, Vperm* magnitude, int numRegions) : EModel(Type::QSPS),
	altMin_d{ altMin }, altMax_d{ altMax }, magnitude_d{ magnitude }, numRegions_m{ numRegions }
{

}

__host__ __device__ QSPS::~QSPS()
{
	#ifndef __CUDA_ARCH__ //host code
	if (useGPU_m) deleteEnvironment();
	#endif /* !__CUDA_ARCH__ */
}

__host__ void QSPS::setupEnvironment()
{
	#ifndef __CUDA_ARCH__ //host code
	CUDA_API_ERRCHK(cudaMalloc((void **)&this_d, sizeof(QSPS*))); //malloc for ptr to ptr to GPU QSPS Obj
	CUDA_API_ERRCHK(cudaMalloc((void **)&altMin_d, altMin_m.size() * sizeof(meters))); //array of altitude min bounds
	CUDA_API_ERRCHK(cudaMalloc((void **)&altMax_d, altMax_m.size() * sizeof(meters)));
	CUDA_API_ERRCHK(cudaMalloc((void **)&magnitude_d, magnitude_m.size() * sizeof(Vperm))); //array of E magnitude between above min/max
	CUDA_API_ERRCHK(cudaMemcpy(altMin_d, altMin_m.data(), altMin_m.size() * sizeof(meters), cudaMemcpyHostToDevice));
	CUDA_API_ERRCHK(cudaMemcpy(altMax_d, altMax_m.data(), altMax_m.size() * sizeof(meters), cudaMemcpyHostToDevice));
	CUDA_API_ERRCHK(cudaMemcpy(magnitude_d, magnitude_m.data(), magnitude_m.size() * sizeof(meters), cudaMemcpyHostToDevice));

	QSPS_d::setupEnvironment_d <<< 1, 1 >>> (this_d, altMin_d, altMax_d, magnitude_d, (int)(magnitude_m.size()));
	CUDA_KERNEL_ERRCHK_WSYNC(); //creates GPU instance of QSPS
	#endif /* !__CUDA_ARCH__ */
}

__host__ void QSPS::deleteEnvironment()
{
	QSPS_d::deleteEnvironment_d <<< 1, 1 >>> (this_d);
	CUDA_KERNEL_ERRCHK_WSYNC();

	CUDA_API_ERRCHK(cudaFree(this_d));
	CUDA_API_ERRCHK(cudaFree(altMin_d)); //On device
	CUDA_API_ERRCHK(cudaFree(altMax_d));
	CUDA_API_ERRCHK(cudaFree(magnitude_d));
}

__host__ __device__ Vperm QSPS::getEFieldAtS(const meters s, const seconds t) const
{
	#ifndef __CUDA_ARCH__ //host code
	for (int ind = 0; ind < magnitude_m.size(); ind++)
	{
		if (s >= altMin_m.at(ind) && s <= altMax_m.at(ind))
			return magnitude_m.at(ind);
	}
	#else //device code
	for (int ind = 0; ind < numRegions_m; ind++)
	{
		if (s >= altMin_d[ind] && s <= altMax_d[ind])
			return magnitude_d[ind];
	}
	#endif /* !__CUDA_ARCH__ */

	return 0.0;
}