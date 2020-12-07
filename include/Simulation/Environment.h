#ifndef ENVIRONMENT_H
#define ENVIRONMENT_H

#include <vector>

using std::vector;

typedef size_t GPURealID;    //Real ID is the CUDA ID of the GPU
typedef size_t GPUEffID;     //Effective ID is the index of the active GPUs
                             //So, if GPUs 0, 2 are active, Eff ID is 0, 1
							 //This is used to index from 0:end over active GPUs

class Environment
{
private:
	class CPU;
	class GPU;

	vector<CPU> cpus_m;
	vector<GPU> gpus_m;

	size_t numParticles_m;
	
	void taskSplit();

public:
	Environment(size_t numParticles);

	void useCPU(bool use, GPURealID cpunum = 0);
	void useGPU(bool use, GPURealID gpunum = 0);
	void setSpeed(const vector<int>& cpuspd, const vector<int>& gpuspd);
	void setBlockSize(size_t blocksize, GPURealID gpu);

	size_t getBlockAlignedSize(GPUEffID gpuind) const;
	GPURealID getRealID(GPUEffID gpuind) const;
	size_t numCPUs() const;    //Returns number of CPUs (GPUs) in use
	size_t numGPUs() const;
	size_t gpuBlockSize(GPURealID gpu) const;

	CPU getCPU(int cpuind) const;
	GPU getGPU(int gpuind) const;
};

class Environment::CPU
{
private:
	int numberThreads_m{ 0 };
	int dataStartIdx_m{ -1 };
	int dataEndIdx_m{ -1 };
	int speed_m{ 0 };
	bool use_m{ false };

	void speedTest();

	friend class Environment;

public:
	CPU(bool use = false);

	int start() const;
	int end()   const;
	int speed() const;
	bool use()  const;
	int cores() const;
};

class Environment::GPU
{
private:
	cudaDeviceProp gpuProps_m;
	size_t blockSize_m{ 256 };
	size_t gridSize_m{ 0 };       //total number of threads / particles responsible for
	
	size_t speed_m{ 0 };
	int devnum_m{ 0 };
	bool use_m{ false };

	void speedTest();
	void setBlockSize(int blocksize);

	friend class Environment;

public:
	GPU(bool use = false);

	int speed() const;
	bool use()  const;
	cudaDeviceProp props() const;
};

#endif //end ENVIRONMENT_SIMULATION_H