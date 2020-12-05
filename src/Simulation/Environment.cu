#include <omp.h>
#include <string>
#include "Simulation/Environment.h"
#include "ErrorHandling/cudaErrorCheck.h"

#define cDP cudaDeviceProp
#define Env Environment
#define EnC Environment::CPU
#define EnG Environment::GPU

using std::cout;
using std::cerr;
using std::string;
using std::to_string;
using std::exception;
using std::logic_error;

//
// ================ Environment ================ //
//
Environment::Environment(int numOfElements) : numOfElements_m{ numOfElements }
{
	cpus_m.push_back(CPU());

	int gpuCount{ 0 };
	if (CUDA_API_ERRCHK(cudaGetDeviceCount(&gpuCount)))
	{
		cout << "Error: cannot get GPU count.  Only default GPU used.  Assuming at least one CUDA capable device.\n";
		gpus_m.push_back(GPU());
		return;
	}

	for (int gpu = 0; gpu < gpuCount; gpu++)
	{
		cudaSetDevice(gpu);
		gpus_m.push_back(GPU());
	}

	cudaSetDevice(0);

	taskSplit(); //not as efficient as I would like...
				 //splits tasks evenly between all devices (which default to use_m = true)
				 //later if/when the user decides to manually add speeds calculated themselves
				 //(which should be done for now), this leads to taskSplit() being called twice
				 //This is basically useless because it's recommended to change speeds
				 //but it's the only way to ensure that speeds / indicies are assigned out to each device
				 //- that is - if I remove this and a user doesn't set speeds (invoking taskSplit())
				 //indicies will remain default of -1 - which will throw an exception when used in code
				 //Later, a more robust speed testing needs to be developed, or the user needs to be forced
				 //to supply their own speeds on instantiation of an Environment instance
}

// ---------------- Environment::Private ---------------- //
/*void Env::taskSplit()  //removed in favor of getBlockAlignedSize
{
	for (auto& elem : gpus_m)
		if (elem.use_m && elem.speed_m == 0) elem.speedTest();

	for (auto& elem : cpus_m)
		if (elem.use_m && elem.speed_m == 0) elem.speedTest();


	int sumSpeed{ 0 };
	for (auto& elem : gpus_m)
		if (elem.use_m && elem.speed_m != 0) sumSpeed += elem.speed_m;

	for (auto& elem : cpus_m)
		if (elem.use_m && elem.speed_m != 0) sumSpeed += elem.speed_m;

	auto gpuBlockAlignedTaskSize = [](int numOfElements, int maxThreadsPerBlock, int devSpeed, int totalSpeed)
	{
		float elemCnt{ 1.0f * numOfElements * devSpeed / totalSpeed };
		int ret{ (int)round(elemCnt / maxThreadsPerBlock) * maxThreadsPerBlock };

		return ret;
	};

	int elemIdx{ 0 };
	for (auto& elem : gpus_m)
	{
		if (elem.use_m == false || elem.speed_m == 0) continue;
		elem.dataStartIdx_m = elemIdx;
		elemIdx += gpuBlockAlignedTaskSize(numOfElements_m, elem.gpuProps_m.maxThreadsPerBlock, elem.speed_m, sumSpeed);
		elem.dataEndIdx_m = elemIdx - 1;
	}
	for (auto& elem : cpus_m)
	{
		if (elem.use_m == false || elem.speed_m == 0) continue;
		elem.dataStartIdx_m = elemIdx;
		elemIdx += gpuBlockAlignedTaskSize(numOfElements_m, 1, elem.speed_m, sumSpeed);
		elem.dataEndIdx_m = elemIdx - 1;
	}

	int elemDiff{ numOfElements_m - 1 - elemIdx };
	if (elemDiff != 0)
	{
		cerr << "Assigning (or removing) extras to (from) last cpu\n";
		cerr << "elemIdx: " << elemIdx << "\n";
		cerr << "#elems : " << numOfElements_m << "\n";
		cerr << "Diff   : " << elemDiff << "\n";
		cpus_m.back().dataEndIdx_m = numOfElements_m - 1;
	}
}*/

// ---------------- Environment::Public ---------------- //
void Env::useCPU(bool use, size_t cpunum) //cpunum default = 0
{
	if (cpunum < cpus_m.size() - 1)
	{
		cout << "Invalid CPU num: " << cpunum << " Returning without changing.\n";
		//cerr << "Some Logged Error\n";
		return;
	}
	cout << "CPU use alongside GPU not implemented yet\n";
	cpus_m.at(cpunum).use_m = false;  return;
	
	cpus_m.at(cpunum).use_m = use;
	taskSplit();
}

void Env::useGPU(bool use, size_t gpunum) //gpunum default = 0
{
	if (gpunum < gpus_m.size() - 1)
	{
		cout << "Invalid GPU num: " << gpunum << " Returning without changing.\n";
		//cerr << "Some Logged Error\n";
		return;
	}
	
	gpus_m.at(gpunum).use_m = use;
	taskSplit();
}

void Env::setSpeed(const vector<int>& cpuspd, const vector<int>& gpuspd)
{
	if (gpuspd.size() != gpus_m.size())
	{
		cout << "Invalid GPU speed size.  No changes made to speed.\n";
		//cerr << "Some Logged Error\n";
		return;
	}
	if (cpuspd.size() != cpus_m.size())
	{
		cout << "Invalid CPU speed size.  No changes made to speed.\n";
		//cerr << "Some Logged Error\n";
		return;
	}

	for (int i = 0; i < gpus_m.size(); i++)
		gpus_m.at(i).speed_m = gpuspd.at(i);
	for (int i = 0; i < cpus_m.size(); i++)
		cpus_m.at(i).speed_m = cpuspd.at(i);

	taskSplit();
}

void Env::setBlockSize(size_t blocksize, size_t gpu)
{
	try
	{
		gpus_m.at(gpu).setBlockSize(blocksize);
	}
	catch (exception& e)
	{
		cout << e.what();
		cout << "  No changes made.\n";
	}
}

size_t Env::getBlockAlignedSize(size_t gpuind, size_t totalNumber)
{
	return totalNumber;

	//the below is not complete yet - returning total number to allow for single GPU use
	float sumspeed{ 0.0 };
	for (const auto& gpu : gpus_m)
		sumspeed += (gpu.use() ? static_cast<float>(gpu.speed()) : 0.0);

	vector<size_t> counts;
	size_t idx{ 0 };
	for (const auto& gpu : gpus_m)
	{
		if (gpu.use())
		{
			float pct{ static_cast<float>(gpu.speed()) / sumspeed };
			size_t elems{ static_cast<size_t>(pct * static_cast<float>(totalNumber)) };
			counts.push_back( (elems / gpu.blockSize_m + 
				((elems % gpu.blockSize_m) ? 1 : 0)) * gpu.blockSize_m);
		}
	}

	size_t sum{ 0 };
	for (const auto& count : counts)
		sum += count;

	size_t diff{ 0 };
	if (sum < totalNumber)
	{
		diff = totalNumber - sum;
		
	}

	
	return 0;
}

size_t Env::getCUDAGPUInd(size_t gpuind)
{
	size_t activeind{ 0 };
	for (const auto& gpu : gpus_m)
	{
		if (gpu.use())
		{
			if (activeind == gpuind) return gpu.devnum_m;
			activeind++;
		}
	}

	cerr << "Environment::getCUDAGPUInd: gpuind " + to_string(gpuind) + " not valid.  Using default (0)";
}

size_t Env::numCPUs() const { return cpus_m.size(); }
size_t Env::numGPUs() const { return gpus_m.size(); }

int Env::gpuBlockSize(int gpu) const
{
	try
	{
		return gpus_m.at(gpu).blockSize_m;
	}
	catch(exception& e)
	{
		cout << e.what();
	}

	return 0;
}

EnC Env::getCPU(int cpuind) const
{
	try
	{
		return cpus_m.at(cpuind);
	}
	catch (exception& e)
	{
		cout << e.what();
		cout << " Returning empty CPU\n";
		return CPU();
	}
}

EnG Env::getGPU(int gpuind) const
{
	try
	{
		return gpus_m.at(gpuind);
	}
	catch (exception & e)
	{
		cout << e.what();
		cout << " Returning empty GPU\n";
		return GPU();
	}
}


//
// ================ Environment::CPU ================ //
//

Environment::CPU::CPU() : numberThreads_m{ omp_get_num_threads() }
{
	speedTest();
}

// ---------------- Environment::CPU::Private ---------------- //
void EnC::speedTest()
{
	if (speed_m == 0)
		speed_m = 1; //a robust speed test not yet implemented - for now, set all speeds equal
}

// ---------------- Environment::CPU::Public ---------------- //
int EnC::start() const { return dataStartIdx_m; }
int EnC::end()   const { return dataEndIdx_m; }
int EnC::speed() const { return speed_m; }
bool EnC::use()  const { return use_m; }
int EnC::cores() const { return numberThreads_m; }

//
// ================ Environment::GPU ================ //
//

Environment::GPU::GPU()
{
	if (CUDA_API_ERRCHK(cudaGetDevice(&devnum_m)))
	{
		cout << "Error getting device number.  Setting generic number: 0\n";
		devnum_m = 0;
	}

	if (CUDA_API_ERRCHK(cudaGetDeviceProperties(&gpuProps_m, devnum_m)))
	{
		cout << "Error getting device props for dev num " << devnum_m << ".  Populating generic data.\n";
		gpuProps_m.maxThreadsPerBlock = 256;
	}

	speedTest();
}

// ---------------- Environment::GPU::Private ---------------- //
void EnG::speedTest()
{
	if (speed_m == 0)      //for now speed == clock rate in khz * MP count
		speed_m = gpuProps_m.clockRate / 1024 * gpuProps_m.multiProcessorCount;
}

void EnG::setBlockSize(int blocksize)
{
	if (blocksize <= 0)
		throw logic_error("Environment::GPU::setBlockSize: Invalid blocksize: blocksize is less than or equal to 0.  \
			Block size must be greater than zero and less than max block size.");

	if (blocksize > gpuProps_m.maxThreadsPerBlock)
		throw logic_error("Environment::GPU::setBlockSize: Specified blocksize is greater than the maximum threads per block for " 
			+ string(gpuProps_m.name));
	
	blockSize_m = blocksize;
}

// ---------------- Environment::GPU::Public ---------------- //
int EnG::speed() const { return speed_m; }
bool EnG::use()  const { return use_m; }
cDP EnG::props() const { return gpuProps_m; }