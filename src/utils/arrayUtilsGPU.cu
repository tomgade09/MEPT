#include "utils/arrayUtilsGPU.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

#include "utils/loopmacros.h"
#include "ErrorHandling/cudaErrorCheck.h"

using std::cerr;
using std::clog;
using std::string;
using std::to_string;
using std::invalid_argument;

namespace utils
{
	namespace GPU
	{
		__global__ void setup2DArray(flPt_t* array1D, flPt_t** array2D, int outerDim, int innerDim)
		{//run once on only one thread
			if (blockIdx.x * blockDim.x + threadIdx.x != 0)
				return;

			for (int out = 0; out < outerDim; out++)
				array2D[out] = &array1D[out * innerDim];
		}

		void setDev(size_t devNum)
		{
			int numdevs{ 0 };
			int activedev{ -1 };
			CUDA_API_ERRCHK(cudaGetDeviceCount(&numdevs));
			CUDA_API_ERRCHK(cudaGetDevice(&activedev));
			int dnumInt{ static_cast<int>(devNum) };
			
			if (dnumInt >= numdevs || dnumInt < 0)
				cerr << "Invalid device number " << dnumInt << ".  Number of devices " << numdevs << ".  Using default device.\n";
			else if (activedev != dnumInt)
				CUDA_API_ERRCHK(cudaSetDevice(dnumInt));
		}

		string getDeviceNames()
		{
			cudaDeviceProp prop;
			size_t devcnt{ getDeviceCount() };
			string devnames;
			for (size_t i = 0; i < devcnt; i++)
			{
				cudaGetDeviceProperties(&prop, (int)i);
				devnames += prop.name;
				if (i != devcnt - 1) devnames += ", ";
			}

			return devnames;
		}
		
		size_t getDeviceCount()
		{
			int numdevs{ 0 };
			cudaGetDeviceCount(&numdevs);
			return static_cast<size_t>(numdevs);
		}
		
		void setup2DArray(flPt_t** data1D_d, vector<flPt_t*>& data2D_d, size_t outerDim, size_t innerDim, size_t devNum)
		{   //creates 2D array and returns the pointer to the whole 1D array, as well as the pointers to locations within that 1D array
			//in the vector data2D_d, and the array of pointers on device is discarded - this is different behavior than below
			setDev(devNum);

			if (data2D_d.size() != outerDim)
				data2D_d.resize(outerDim);
			
			flPt_t** tmp_d{ nullptr };
			CUDA_API_ERRCHK(cudaMalloc((void**)&(*data1D_d), outerDim * innerDim * sizeof(flPt_t)));
			CUDA_API_ERRCHK(cudaMalloc((void**)&(tmp_d), outerDim * sizeof(flPt_t*)));
			
			CUDA_API_ERRCHK(cudaMemset(*data1D_d, 0, outerDim * innerDim * sizeof(flPt_t)));
			CUDA_API_ERRCHK(cudaMemset(tmp_d, 0, outerDim * sizeof(flPt_t*)));

			setup2DArray <<< 1, 1 >>> (*data1D_d, tmp_d, static_cast<int>(outerDim), static_cast<int>(innerDim));
			CUDA_KERNEL_ERRCHK_WSYNC();
			
			CUDA_API_ERRCHK(cudaMemcpy(data2D_d.data(), tmp_d, outerDim * sizeof(flPt_t*), cudaMemcpyDeviceToHost));
			CUDA_API_ERRCHK(cudaFree(tmp_d));
		}

		void setup2DArray(flPt_t** data1D_d, flPt_t*** data2D_d, size_t outerDim, size_t innerDim, size_t devNum)
		{   //creates 2D array and returns the pointer to both the 1D and 2D arrays on device, but not the pointers to locations within
			//the 1D array - this is different behavior from the other function overload of the same name immediately above
			setDev(devNum);

			CUDA_API_ERRCHK(cudaMalloc((void**)&(*data1D_d), outerDim * innerDim * sizeof(flPt_t)));
			CUDA_API_ERRCHK(cudaMalloc((void**)&(*data2D_d), outerDim * sizeof(flPt_t*)));

			CUDA_API_ERRCHK(cudaMemset(*data1D_d, 0, outerDim * innerDim * sizeof(flPt_t)));
			CUDA_API_ERRCHK(cudaMemset(*data2D_d, 0, outerDim * sizeof(flPt_t*)));

			setup2DArray <<< 1, 1 >>> (*data1D_d, *data2D_d, static_cast<int>(outerDim), static_cast<int>(innerDim));
			CUDA_KERNEL_ERRCHK_WSYNC();
		}

		void copy2DArray(fp2Dvec& data, flPt_t** data1D_d, bool hostToDev, size_t devNum)
		{
			setDev(devNum);
			
			size_t frontSize{ data.front().size() };
			for (const auto& elem : data)
				if (elem.size() != frontSize)
					throw invalid_argument("utils::GPU::copy2DArray: inner vectors of argument 'data' (2D flPt_t vector) are not equally sized.");

			if (hostToDev)
			{
				LOOP_OVER_1D_ARRAY(data.size(), CUDA_API_ERRCHK(cudaMemcpy((*data1D_d) + data.at(0).size() * iii, data.at(iii).data(), data.at(0).size() * sizeof(flPt_t), cudaMemcpyHostToDevice)));
			}
			else
			{
				LOOP_OVER_1D_ARRAY(data.size(), CUDA_API_ERRCHK(cudaMemcpy(data.at(iii).data(), (*data1D_d) + data.at(0).size() * iii, data.at(0).size() * sizeof(flPt_t), cudaMemcpyDeviceToHost)));
			}
			CUDA_KERNEL_ERRCHK_WSYNC();
		}

		void free2DArray(flPt_t** data1D_d, flPt_t*** data2D_d, size_t devNum)
		{
			setDev(devNum);
			
			CUDA_API_ERRCHK(cudaFree(*data1D_d));
			*data1D_d = nullptr;

			if (data2D_d != nullptr)
			{
				CUDA_API_ERRCHK(cudaFree(*data2D_d));
				*data2D_d = nullptr;
			}
		}

		void getGPUMemInfo(size_t* free, size_t* total, size_t devNum)
		{ //use CUDA API to get free and total mem sizes for a specified GPU
			int currDev{ -1 };
			CUDA_API_ERRCHK(cudaGetDevice(&currDev));

			if (currDev != (int)devNum) CUDA_API_ERRCHK(cudaSetDevice(devNum));

			CUDA_API_ERRCHK(cudaMemGetInfo(free, total));

			if (currDev != devNum) CUDA_API_ERRCHK(cudaSetDevice(currDev));
		}

		void getCurrGPUMemInfo(size_t* free, size_t* total)
		{ //difference from above is this just checks the current device
			CUDA_API_ERRCHK(cudaMemGetInfo(free, total));
		}

		vector<size_t> getSplitSize(size_t numOfParticles, size_t blocksize)
		{   //returns the number of particles a device will receive based on the pct in computeSplit_m
			//returns block aligned numbers except last one, if total is not a multiple of block size

			int gpuCount = 0;
			CUDA_API_ERRCHK(cudaGetDeviceCount(&gpuCount));

			cudaDeviceProp devProp;
			flPt_t computeTotal = 0;

			fp1Dvec computeSplit;

			// Iterate over each GPU and determine how much data it can handle
			for (int gpu = 0; gpu < gpuCount; gpu++)
			{
				// Get the GPU Speed
				CUDA_API_ERRCHK(cudaGetDeviceProperties(&devProp, gpu));

				// For the author's machine, MP count gives a good metric to split tasks evenly
				// In future: either optimize for specific hardware create a more precise equation
				flPt_t compute = static_cast<flPt_t>(devProp.clockRate / 1024.0 * devProp.multiProcessorCount);
				computeTotal += compute;
				computeSplit.push_back(compute);  //need to use floats to get decimal numbers
			}

			// Iterate through computeSplit and get percent ratio work each device will get
			for (size_t i = 0; i < computeSplit.size(); ++i)
			{
				computeSplit.at(i) /= computeTotal;
			}

			vector<size_t> particleSplit;

			auto getBlockAlignedCount = [&](size_t count)
			{
				size_t ret{ count };
				flPt_t bs{ static_cast<flPt_t>(blocksize) };

				if (count % blocksize)
				{
					flPt_t pct(static_cast<flPt_t>(count % blocksize) / bs);  //find percent of blocksize of the remainder
					if (pct >= 0.5)      //if remainder is over 50% of block size, add an extra block
						ret = static_cast<size_t>((count / blocksize + 1) * blocksize);
					else                 //else just return the block aligned size
						ret = static_cast<size_t>((count / blocksize) * blocksize);
				}

				return ret;
			};

			size_t total{ 0 };
			for (const auto& comp : computeSplit)
			{
				size_t bsaln{ getBlockAlignedCount(static_cast<size_t>(comp * static_cast<flPt_t>(numOfParticles))) };
				total += bsaln;
				particleSplit.push_back(bsaln);
			}

			//above code creates particle count in multiples of blocksize
			size_t diff{ 0 };
			if (total < numOfParticles)  //check that we aren't missing particles or have extra blocks
			{
				diff = numOfParticles - total;

				while (diff / blocksize)  //does not execute when diff < blocksize
				{   //if more than one block not accounted for... (shouldn't happen)
					cerr << "Simulation::getSplitSize: total < # parts : Need " + to_string(diff / blocksize) + " more full blocks\n";
					particleSplit.at((diff / blocksize) % gpuCount) += blocksize;  //add one full block to GPU
					diff -= blocksize;
					//overflows are prevented by the fact that when the int div product is 0, this doesn't execute
				}

				//less than one full block of unaccounted for particles remains.  Add diff
				particleSplit.back() += diff;  //add diff to total
				clog << "Simulation::getSplitSize: Adding diff " + to_string(diff) + ".  Total: " + to_string(total + diff) << "\n";

				for (size_t dev = 0; dev < gpuCount; dev++)
					clog << "Simulation::getSplitSize: GPU " + to_string(dev) + ": particle count: " + to_string(particleSplit.at(dev)) << "\n";
			}
			else if (total > numOfParticles)
			{
				diff = total - numOfParticles;

				while (diff / blocksize)
				{   //if one or more whole extra blocks created... (shouldn't happen)
					cerr << "Simulation::getSplitSize: total > # parts " + to_string(diff / blocksize) + " extra blocks to create\n";
					particleSplit.at((diff / blocksize) % gpuCount) -= blocksize;  //remove one full block from GPU
					diff -= blocksize;
				}

				//less than one full block of particles extra - subtract diff
				particleSplit.back() -= diff;  //shrink total by diff
				clog << "Simulation::getSplitSize: Subtracting diff " + to_string(diff) + ".  Total: " + to_string(total - diff) << "\n";

				for (size_t dev = 0; dev < gpuCount; dev++)
					clog << "Simulation::getSplitSize: GPU " + to_string(dev) + ": particle count: " + to_string(particleSplit.at(dev)) << "\n";
			}

			return particleSplit;
		}
	}
}