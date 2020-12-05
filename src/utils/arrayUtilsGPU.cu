#include "utils/arrayUtilsGPU.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "curand_kernel.h"

#include "utils/loopmacros.h"
#include "ErrorHandling/cudaErrorCheck.h"
//#include "ErrorHandling/cudaDeviceMacros.h"

using std::cout;
using std::clog;
using std::invalid_argument;

namespace utils
{
	namespace GPU
	{
		__global__ void setup2DArray(float* array1D, float** array2D, int outerDim, int innerDim)
		{//run once on only one thread
			if (blockIdx.x * blockDim.x + threadIdx.x != 0)
				return;

			for (int out = 0; out < outerDim; out++)
				array2D[out] = &array1D[out * innerDim];
		}

		void setDev(size_t devNum)  //internal function not exposed to outside (through header)
		{
			int numdevs{ 0 };
			int activedev{ -1 };
			cudaGetDeviceCount(&numdevs);
			cudaGetDevice(&activedev);
			int dnumInt{ static_cast<size_t>(devNum) };
			
			if (dnumInt >= numdevs || dnumInt < 0)
				clog << "Invalid device number " << dnumInt << ".  Number of devices " << numdevs << ".  Using default device.";
			else if (activedev != dnumInt)
				cudaSetDevice(dnumInt);
		}
		
		void setup2DArray(float** data1D_d, float*** data2D_d, size_t outerDim, size_t innerDim, size_t devNum)
		{
			setDev(devNum);
			
			CUDA_API_ERRCHK(cudaMalloc((void**)&(*data1D_d), outerDim * innerDim * sizeof(float*)));
			CUDA_API_ERRCHK(cudaMalloc((void**)&(*data2D_d), outerDim * sizeof(float*)));
			
			CUDA_API_ERRCHK(cudaMemset(*data1D_d, 0, outerDim * innerDim * sizeof(float)));

			setup2DArray <<< 1, 1 >>> (*data1D_d, *data2D_d, static_cast<int>(outerDim), static_cast<int>(innerDim));
			CUDA_KERNEL_ERRCHK_WSYNC();
		}

		void copy2DArray(vector<vector<float>>& data, float** data1D_d, bool hostToDev, size_t devNum)
		{
			setDev(devNum);
			
			size_t frontSize{ data.front().size() };
			for (const auto& elem : data)
				if (elem.size() != frontSize)
					throw invalid_argument("utils::GPU::copy2DArray: inner vectors of argument 'data' (2D float vector) are not equally sized.");

			if (hostToDev)
			{
				LOOP_OVER_1D_ARRAY(data.size(), CUDA_API_ERRCHK(cudaMemcpy((*data1D_d) + data.at(0).size() * iii, data.at(iii).data(), data.at(0).size() * sizeof(float), cudaMemcpyHostToDevice)));
			}
			else
			{
				LOOP_OVER_1D_ARRAY(data.size(), CUDA_API_ERRCHK(cudaMemcpy(data.at(iii).data(), (*data1D_d) + data.at(0).size() * iii, data.at(0).size() * sizeof(float), cudaMemcpyDeviceToHost)));
			}
		}

		void free2DArray(float** data1D_d, float*** data2D_d, size_t devNum)
		{
			setDev(devNum);
			
			CUDA_API_ERRCHK(cudaFree(*data1D_d));
			CUDA_API_ERRCHK(cudaFree(*data2D_d));

			*data1D_d = nullptr;
			*data2D_d = nullptr;
		}

		void getGPUMemInfo(size_t* free, size_t* total, size_t devNum)
		{ //use CUDA API to get free and total mem sizes for a specified GPU
			int currDev{ -1 };
			CUDA_API_ERRCHK(cudaGetDevice(&currDev));

			if (currDev != devNum) CUDA_API_ERRCHK(cudaSetDevice(devNum));

			CUDA_API_ERRCHK(cudaMemGetInfo(free, total));

			if (currDev != devNum) CUDA_API_ERRCHK(cudaSetDevice(currDev));
		}

		void getCurrGPUMemInfo(size_t* free, size_t* total)
		{ //difference from above is this just checks the current device
			CUDA_API_ERRCHK(cudaMemGetInfo(free, total));
		}
	}
}