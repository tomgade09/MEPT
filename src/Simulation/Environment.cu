#include <omp.h>

#include "Simulation/Environment.h"
#include "ErrorHandling/cudaErrorCheck.h"

#define cDP cudaDeviceProp
#define SEn Simulation::Environment
#define SEC Simulation::Environment::CPU
#define SEG Simulation::Environment::GPU

using std::cout;
using std::cerr;
using std::string;
using std::exception;
using std::logic_error;

//
// ================ Environment ================ //
//
Simulation::Environment::Environment(int numOfElements) : numOfElements_m{ numOfElements }
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
void SEn::taskSplit()
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
		double elemCnt{ 1.0 * numOfElements * devSpeed / totalSpeed };
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
}

// ---------------- Environment::Public ---------------- //
void SEn::useCPU(bool use, size_t cpunum) //cpunum default = 0
{
	if (cpunum < cpus_m.size() - 1)
	{
		cout << "Invalid CPU num: " << cpunum << " Returning without changing.\n";
		//cerr << "Some Logged Error\n";
		return;
	}

	if (cpus_m.at(cpunum).use_m == use) return;
	cpus_m.at(cpunum).use_m = use;
	taskSplit();
}

void SEn::useGPU(bool use, size_t gpunum) //gpunum default = 0
{
	if (gpunum < gpus_m.size() - 1)
	{
		cout << "Invalid GPU num: " << gpunum << " Returning without changing.\n";
		//cerr << "Some Logged Error\n";
		return;
	}

	if (cpus_m.at(gpunum).use_m == use) return;
	gpus_m.at(gpunum).use_m = use;
	taskSplit();
}

void SEn::setSpeed(const vector<int>& cpuspd, const vector<int>& gpuspd)
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

void SEn::setBlockSize(int blocksize, int gpu)
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

size_t SEn::numCPUs() const { return cpus_m.size(); }
size_t SEn::numGPUs() const { return gpus_m.size(); }

int SEn::gpuBlockSize(int gpu) const
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

SEC SEn::getCPU(int cpuind) const
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

SEG SEn::getGPU(int gpuind) const
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

Simulation::Environment::CPU::CPU() : numberThreads_m{ omp_get_num_threads() }
{
	speedTest();
}

// ---------------- Environment::CPU::Private ---------------- //
void SEC::speedTest()
{
	if (speed_m == 0)
		speed_m = 1; //a robust speed test not yet implemented - for now, set all speeds equal
}

// ---------------- Environment::CPU::Public ---------------- //
int SEC::start() const { return dataStartIdx_m; }
int SEC::end()   const { return dataEndIdx_m; }
int SEC::speed() const { return speed_m; }
bool SEC::use()  const { return use_m; }
int SEC::cores() const { return numberThreads_m; }

//
// ================ Environment::GPU ================ //
//

Simulation::Environment::GPU::GPU()
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
void SEG::speedTest()
{
	if (speed_m == 0)
		speed_m = 1; //a robust speed test not yet implemented - for now, set all speeds equal
}

void SEG::setBlockSize(int blocksize)
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
int SEG::start() const { return dataStartIdx_m; }
int SEG::end()   const { return dataEndIdx_m; }
int SEG::speed() const { return speed_m; }
bool SEG::use()  const { return use_m; }
cDP SEG::props() const { return gpuProps_m; }