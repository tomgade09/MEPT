#ifndef ENVIRONMENT_SIMULATION_H
#define ENVIRONMENT_SIMULATION_H

#include <vector>

using std::vector;

#define cDP cudaDeviceProp

class Environment
{
private:
	class CPU;
	class GPU;

	vector<CPU> cpus_m;
	vector<GPU> gpus_m;

	void taskSplit();

public:
	Environment();

	void useCPU(bool use, size_t cpunum = 0);
	void useGPU(bool use, size_t gpunum = 0);
	void setSpeed(const vector<int>& cpuspd, const vector<int>& gpuspd);
	void setBlockSize(size_t blocksize, size_t gpu);

	size_t getBlockAlignedSize(size_t gpuind, size_t totalNumber);
	size_t getCUDAGPUInd(size_t gpuind);
	size_t numCPUs() const;
	size_t numGPUs() const;
	int    gpuBlockSize(int gpu) const;

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
	CPU();

	int start() const;
	int end()   const;
	int speed() const;
	bool use()  const;
	int cores() const;
};

class Environment::GPU
{
private:
	cDP gpuProps_m;
	int blockSize_m{ 256 };
	int speed_m{ 0 };
	int devnum_m{ 0 };
	bool use_m{ true };

	void speedTest();
	void setBlockSize(int blocksize);

	friend class Environment;

public:
	GPU();

	int speed() const;
	bool use()  const;
	cDP props() const;
};

#undef cDP

#endif //end ENVIRONMENT_SIMULATION_H