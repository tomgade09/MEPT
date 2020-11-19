#include "BField/DipoleBLUT.h"
#include "BField/DipoleB.h"

#include <memory>
#include <filesystem>

#include "device_launch_parameters.h"
#include "utils/serializationHelpers.h"
#include "ErrorHandling/cudaErrorCheck.h"
#include "ErrorHandling/cudaDeviceMacros.h"

using std::cerr;
using std::string;
using std::invalid_argument;
using namespace utils::fileIO::serialize;

constexpr double LAMBDAERRTOL{ 1.0e-10 }; //the error tolerance of DipoleB's lambda estimation

//setup CUDA kernels
namespace DipoleBLUT_d
{
	__global__ void setupEnvironmentGPU(BModel** this_d, double ILATDeg, double simMin, double simMax, double ds_gradB, int numMsmts, double* altArray, double* magArray)
	{
		ZEROTH_THREAD_ONLY(
			*this_d = new DipoleBLUT(ILATDeg, simMin, simMax, ds_gradB, numMsmts);
			((DipoleBLUT*)(*this_d))->setAltArray(altArray);
			((DipoleBLUT*)(*this_d))->setMagArray(magArray);
		);
	}

	__global__ void deleteEnvironmentGPU(BModel** this_d)
	{
		ZEROTH_THREAD_ONLY(delete ((DipoleBLUT*)(*this_d)));
	}
}
//end


//DipoleBLUT protected member functions
__host__ void DipoleBLUT::setupEnvironment()
{// consts: [ ILATDeg, L, L_norm, s_max, ds, errorTolerance ]
	CUDA_API_ERRCHK(cudaMalloc((void**)&this_d, sizeof(DipoleBLUT*)));
	CUDA_API_ERRCHK(cudaMalloc((void**)&altitude_d, sizeof(meters) * numMsmts_m));
	CUDA_API_ERRCHK(cudaMalloc((void**)&magnitude_d, sizeof(tesla) * numMsmts_m));

	#ifndef __CUDA_ARCH__ //host code
	CUDA_API_ERRCHK(cudaMemcpy(altitude_d, altitude_m.data(), sizeof(meters) * numMsmts_m, cudaMemcpyHostToDevice));
	CUDA_API_ERRCHK(cudaMemcpy(magnitude_d, magnitude_m.data(), sizeof(meters) * numMsmts_m, cudaMemcpyHostToDevice));
	#endif /* !__CUDA_ARCH__ */
	
	DipoleBLUT_d::setupEnvironmentGPU <<< 1, 1 >>> (this_d, ILAT_m, simMin_m, simMax_m, ds_gradB_m, numMsmts_m, altitude_d, magnitude_d);
	CUDA_KERNEL_ERRCHK_WSYNC();
}

__host__ void DipoleBLUT::deleteEnvironment()
{
	DipoleBLUT_d::deleteEnvironmentGPU <<< 1, 1 >>> (this_d);
	CUDA_KERNEL_ERRCHK_WSYNC();

	CUDA_API_ERRCHK(cudaFree(this_d));
	CUDA_API_ERRCHK(cudaFree(altitude_d));
	CUDA_API_ERRCHK(cudaFree(magnitude_d));
}


//DipoleBLUT public member functions
__host__ __device__ DipoleBLUT::DipoleBLUT(degrees ILAT, meters simMin, meters simMax, meters ds_gradB, int numberOfMeasurements, bool useGPU) :
	BModel(Type::DipoleBLUT), ILAT_m{ ILAT }, simMin_m{ simMin }, simMax_m{ simMax }, ds_gradB_m{ ds_gradB }, numMsmts_m{ numberOfMeasurements }, useGPU_m{ useGPU }
{
	ds_msmt_m = (simMax_m - simMin_m) / (numMsmts_m - 1);

	#ifndef __CUDA_ARCH__ //host code
	std::unique_ptr<DipoleB> dip = std::make_unique<DipoleB>(ILAT_m, LAMBDAERRTOL, ds_gradB_m, false); //destroyed at end of function
	
	for (int msmt = 0; msmt < numMsmts_m; msmt++)
	{
		altitude_m.push_back(simMin_m + msmt * ds_msmt_m);
		magnitude_m.push_back(dip->getBFieldAtS(altitude_m.at(msmt), 0.0));
	}

	if (useGPU_m) setupEnvironment();
	#endif /* !__CUDA_ARCH__ */
}

__host__ DipoleBLUT::DipoleBLUT(ifstream& in) : BModel(Type::DipoleBLUT)
{
	deserialize(in);
	ds_msmt_m = (simMax_m - simMin_m) / (numMsmts_m - 1);

	if (useGPU_m) setupEnvironment();
}

__host__ __device__ DipoleBLUT::~DipoleBLUT()
{
	#ifndef __CUDA_ARCH__ //host code
	if (useGPU_m) deleteEnvironment();
	#endif /* !__CUDA_ARCH__ */
}

__host__ degrees DipoleBLUT::ILAT() const
{
	return ILAT_m;
}

__host__ __device__ tesla DipoleBLUT::getBFieldAtS(const meters s, const seconds simtime) const
{// consts: [ ILATDeg, L, L_norm, s_max, ds, errorTolerance ]
	int startInd{ 0 };
	if (s <= simMin_m)
		startInd = 0;
	else if (s >= simMax_m)
		startInd = numMsmts_m - 2; //if s is above simMax, we interpolate based on the highest indicies ([numMsmts - 2] to [numMsmts - 1])
	else
		startInd = (int)((s - simMin_m) / ds_msmt_m); //c-style cast to int basically == floor()
	
	// deltaB_bin / deltas_bin^3 * (s'(dist up from altBin)) + B@altBin
	#ifndef __CUDA_ARCH__ //host code
	return (s - altitude_m.at(startInd)) * (magnitude_m.at(startInd + 1) - magnitude_m.at(startInd)) / ds_msmt_m + magnitude_m.at(startInd); //B = ms + b(0)
	#else
	return (s - altitude_d[startInd]) * (magnitude_d[startInd + 1] - magnitude_d[startInd]) / ds_msmt_m + magnitude_d[startInd]; //B = ms + b(0)
	#endif /* !__CUDA_ARCH__ */
}

__host__ __device__ double DipoleBLUT::getGradBAtS(const meters s, const seconds simtime) const
{
	return (getBFieldAtS(s + ds_gradB_m, simtime) - getBFieldAtS(s - ds_gradB_m, simtime)) / (2 * ds_gradB_m);
}

__host__ __device__ meters DipoleBLUT::getSAtAlt(const meters alt_fromRe) const
{
	//admittedly, this is a pretty inefficient way of doing this...but it's not used often
	return DipoleB(ILAT_m, 1.0e-4, RADIUS_EARTH / 1000.0, false).getSAtAlt(alt_fromRe);
}

__device__ void DipoleBLUT::setAltArray(double* altArray)
{
	altitude_d = altArray;
}

__device__ void DipoleBLUT::setMagArray(double* magArray)
{
	magnitude_d = magArray;
}

__host__ ratio DipoleBLUT::getErrTol() const
{
	return LAMBDAERRTOL;
}

__host__ meters DipoleBLUT::getds() const
{
	return ds_gradB_m;
}