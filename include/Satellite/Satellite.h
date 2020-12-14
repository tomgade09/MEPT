#ifndef SATELLITE_H
#define SATELLITE_H

#include <vector>
#include <string>

#include "dlldefines.h"
#include "Particles/Particles.h"
#include "utils/unitsTypedefs.h"

using std::vector;
using std::string;
using std::ifstream;
using std::ofstream;

#define STRVEC vector<string>
#define FLT2DV vector<vector<float>>
#define FLT3DV vector<vector<vector<float>>>

class Satellite
{
protected:
	string name_m;
	STRVEC attributeNames_m;
	
	meters altitude_m{ 0.0f };
	bool   upwardFacing_m{ false };
	bool   initializedGPU_m{ false };

	size_t numberOfParticles_m{ 0 };
	size_t numGPUs_m;
	vector<size_t> particleCountPerGPU_m;
	
	FLT2DV  data_m; //[attribute][particle]
	vector<FLT2DV>	data_GPU_m;
	vector<float*>  satCaptrData1D_d; //flattened satellite capture data on GPU
	vector<float**> satCaptrData2D_d; //2D satellite capture data on GPU
	vector<float**> particleData2D_d;

	void   initializeGPU();
	void   freeGPUMemory();
	void   deserialize(ifstream& in);
	size_t getAttrIndByName(string name);

public:
	Satellite(string name, STRVEC attributeNames, meters altitude, bool upwardFacing, size_t numberOfParticles, const std::shared_ptr<Particles>& particle, size_t numGPUs, vector<size_t> particleCountPerGPU_m);
	Satellite(ifstream& in, const std::shared_ptr<Particles> &particle, size_t numGPUs, vector<size_t> particleCountPerGPU);
	~Satellite();
	Satellite(const Satellite&) = delete;
	Satellite& operator=(const Satellite&) = delete;

	//Access functions
	string        name() const;
	meters        altitude() const;
	bool	      upward() const;
	FLT2DV&       __data();
	const FLT2DV& data() const;
	float**       get2DDataGPUPtr(int GPUind) const;
	float*        get1DDataGPUPtr(int GPUind) const;
	size_t        getNumberOfAttributes() const;
	size_t        getNumberOfParticles()  const;

	//Other functions
	void iterateDetector(float simtime, float dt, int blockSize, int GPUind); //increment time, track overall sim time, or take an argument??
	void iterateDetectorCPU(const FLT2DV& particleData, seconds simtime, seconds dt);
	void copyDataToHost(); //some sort of sim time check to verify I have iterated for the current sim time??
	void saveDataToDisk(string folder);
	void loadDataFromDisk(string folder);

	FLT2DV removeZerosData();
	void serialize(ofstream& out) const;
};

#undef STRVEC
#undef FLT2DV
#undef FLT3DV

#endif