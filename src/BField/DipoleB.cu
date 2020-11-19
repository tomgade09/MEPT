#include "BField/DipoleB.h"

#include "device_launch_parameters.h"
#include "ErrorHandling/cudaErrorCheck.h"
#include "ErrorHandling/cudaDeviceMacros.h"

using std::string;

constexpr double B0{ 3.12e-5 }; //B_0 for Earth dipole B model

//setup CUDA kernels
namespace DipoleB_d
{
	__global__ void setupEnvironmentGPU(BModel** this_d, degrees ILAT, ratio errTol, double ds)
	{
		ZEROTH_THREAD_ONLY((*this_d) = new DipoleB(ILAT, errTol, ds));
	}

	__global__ void deleteEnvironmentGPU(BModel** this_d)
	{
		ZEROTH_THREAD_ONLY(delete ((DipoleB*)(*this_d)));
	}
}

//DipoleB protected member functions
void DipoleB::setupEnvironment()
{// consts: [ ILATDeg, L, L_norm, s_max, ds, lambdaErrorTolerance ]
	CUDA_API_ERRCHK(cudaMalloc((void**)&this_d, sizeof(DipoleB*)));
	DipoleB_d::setupEnvironmentGPU <<< 1, 1 >>> (this_d, ILAT_m, lambdaErrorTolerance_m, ds_m);
	CUDA_KERNEL_ERRCHK_WSYNC();
}

void DipoleB::deleteEnvironment()
{
	DipoleB_d::deleteEnvironmentGPU <<< 1, 1 >>> (this_d);
	CUDA_KERNEL_ERRCHK_WSYNC();

	CUDA_API_ERRCHK(cudaFree(this_d));
}
//deserialize is in DipoleB.cpp


//DipoleB public member functions
__host__ __device__ DipoleB::DipoleB(degrees ILAT, ratio lambdaErrorTolerance, meters ds, bool useGPU) : BModel(Type::DipoleB),
	ILAT_m{ ILAT }, ds_m{ ds }, lambdaErrorTolerance_m{ lambdaErrorTolerance }, useGPU_m{ useGPU }
{
	L_m = RADIUS_EARTH / (cos(ILAT_m * RADS_PER_DEG) * cos(ILAT_m * RADS_PER_DEG));
	L_norm_m = L_m / RADIUS_EARTH;
	s_max_m = getSAtLambda(ILAT_m);

	#ifndef __CUDA_ARCH__ //host code
	if (useGPU_m) setupEnvironment();
	#endif /* !__CUDA_ARCH__ */
}

__host__ DipoleB::DipoleB(ifstream& in) : BModel(Type::DipoleB)
{
	deserialize(in);
	if (useGPU_m) setupEnvironment();
}

__host__ __device__ DipoleB::~DipoleB()
{
	#ifndef __CUDA_ARCH__ //host code
	if (useGPU_m) deleteEnvironment();
	#endif /* !__CUDA_ARCH__ */
}

__host__ degrees DipoleB::ILAT() const
{
	return ILAT_m;
}

__host__ __device__ meters DipoleB::getSAtLambda(const degrees lambda) const
{
	//double x{ asinh(sqrt(3.0) * sinpi(lambdaDegrees / 180.0)) }; //asinh triggers an odd cuda 8.x bug that is resolved in 9.x+
	double sinh_x{ sqrt(3.0) * sinpi(lambda / 180.0) };
	double x{ log(sinh_x + sqrt(sinh_x * sinh_x + 1)) }; //trig identity for asinh - a bit faster - asinh(x) == ln(x + sqrt(x*x + 1))

	return (0.5 * L_m / sqrt(3.0)) * (x + 0.25 * (exp(2.0*x)-exp(-2.0*x))); /* L */ //0.25 * (exp(2*x)-exp(-2*x)) == sinh(x) * cosh(x) and is faster
}

__host__ __device__ degrees DipoleB::getLambdaAtS(const meters s) const
{
	degrees lambda_tmp{ (-ILAT_m / s_max_m) * s + ILAT_m }; //-ILAT / s_max * s + ILAT
	meters  s_tmp{ s_max_m - getSAtLambda(lambda_tmp) };
	degrees dlambda{ 1.0 };
	bool    over{ 0 };

	while (abs((s_tmp - s) / s) > lambdaErrorTolerance_m)
	{
		while (1)
		{
			over = (s_tmp >= s);
			if (over)
			{
				lambda_tmp += dlambda;
				s_tmp = s_max_m - getSAtLambda(lambda_tmp);
				if (s_tmp < s)
					break;
			}
			else
			{
				lambda_tmp -= dlambda;
				s_tmp = s_max_m - getSAtLambda(lambda_tmp);
				if (s_tmp >= s)
					break;
			}
		}
		if (dlambda < lambdaErrorTolerance_m / 100.0)
			break;
		dlambda /= 5.0; //through trial and error, this reduces the number of calculations usually (compared with 2, 2.5, 3, 4, 10)
	}

	return lambda_tmp;
}

__host__ __device__ tesla DipoleB::getBFieldAtS(const meters s, const seconds simtime) const
{// consts: [ ILATDeg, L, L_norm, s_max, ds, lambdaErrorTolerance ]
	degrees lambda{ getLambdaAtS(s) };
	meters  rnorm{ L_norm_m * cospi(lambda / 180.0) * cospi(lambda / 180.0) };

	return -B0 / (rnorm * rnorm * rnorm) * sqrt(1.0 + 3 * sinpi(lambda / 180.0) * sinpi(lambda / 180.0));
}

__host__ __device__ double DipoleB::getGradBAtS(const meters s, const seconds simtime) const
{
	return (getBFieldAtS(s + ds_m, simtime) - getBFieldAtS(s - ds_m, simtime)) / (2 * ds_m);
}

__host__ __device__ meters DipoleB::getSAtAlt(const meters alt_fromRe) const
{
	degrees lambda{ acos(sqrt((alt_fromRe + RADIUS_EARTH) / L_m)) / RADS_PER_DEG };
	return s_max_m - getSAtLambda(lambda);
}

__host__ ratio  DipoleB::getErrTol() const
{
	return lambdaErrorTolerance_m;
}

__host__ meters DipoleB::getds() const
{
	return ds_m;
}