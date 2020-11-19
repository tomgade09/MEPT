#ifndef ENVIRONMENT_SIMULATION_H
#define ENVIRONMENT_SIMULATION_H

#include "Simulation/Simulation.h"

using std::vector;

#define cDP cudaDeviceProp

class Simulation::Environment
{
private:
	class CPU;
	class GPU;

	vector<CPU> cpus_m;
	vector<GPU> gpus_m;
	int numOfElements_m;

	void taskSplit();

public:
	Environment(int numOfElements);

	void useCPU(bool use, size_t cpunum = 0);
	void useGPU(bool use, size_t gpunum = 0);
	void setSpeed(const vector<int>& cpuspd, const vector<int>& gpuspd);
	void setBlockSize(int blocksize, int gpu);

	size_t numCPUs() const;
	size_t numGPUs() const;
	int gpuBlockSize(int gpu) const;

	CPU getCPU(int cpuind) const;
	GPU getGPU(int gpuind) const;
};

class Simulation::Environment::CPU
{
private:
	int numberThreads_m{ 0 };
	int dataStartIdx_m{ -1 };
	int dataEndIdx_m{ -1 };
	int speed_m{ 0 };
	bool use_m{ true };

	void speedTest();

	friend class Simulation::Environment;

public:
	CPU();

	int start() const;
	int end()   const;
	int speed() const;
	bool use()  const;
	int cores() const;
};

class Simulation::Environment::GPU
{
private:
	cDP gpuProps_m;
	int blockSize_m{ 1024 };
	int dataStartIdx_m{ -1 };
	int dataEndIdx_m{ -1 };
	int speed_m{ 0 };
	int devnum_m{ 0 };
	bool use_m{ true };

	void speedTest();
	void setBlockSize(int blocksize);

	friend class Simulation::Environment;

public:
	GPU();

	int start() const;
	int end()   const;
	int speed() const;
	bool use()  const;
	cDP props() const;
};

#undef cDP

#endif //end ENVIRONMENT_SIMULATION_H