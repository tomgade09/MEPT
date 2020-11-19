#ifndef SATELLITE_H
#define SATELLITE_H

#include <vector>
#include <string>

#include "dlldefines.h"
#include "utils/unitsTypedefs.h"

using std::vector;
using std::string;
using std::ifstream;
using std::ofstream;

#define STRVEC vector<string>
#define DBL2DV vector<vector<double>>
#define DBL3DV vector<vector<vector<double>>>

class Satellite
{
protected:
	string name_m;
	STRVEC attributeNames_m;
	
	meters altitude_m{ 0.0 };
	bool   upwardFacing_m{ false };
	bool   initializedGPU_m{ false };

	long numberOfParticles_m{ -1 };

	DBL2DV   data_m; //[attribute][particle]
	double*  satCaptrData1D_d{ nullptr }; //flattened satellite capture data on GPU
	double** satCaptrData2D_d{ nullptr }; //2D satellite capture data on GPU
	double** particleData2D_d{ nullptr };

	void   initializeGPU();
	void   freeGPUMemory();
	void   deserialize(ifstream& in);
	size_t getAttrIndByName(string name);

public:
	Satellite(string name, STRVEC attributeNames, meters altitude, bool upwardFacing, long numberOfParticles, double** partDataGPUPtr);
	Satellite(ifstream& in, double** particleData2D);
	~Satellite();
	Satellite(const Satellite&) = delete;
	Satellite& operator=(const Satellite&) = delete;

	//Access functions
	string        name() const;
	meters        altitude() const;
	bool	      upward() const;
	DBL2DV&       __data();
	const DBL2DV& data() const;
	double**      get2DDataGPUPtr() const;
	double*       get1DDataGPUPtr() const;
	size_t  getNumberOfAttributes() const;
	long    getNumberOfParticles()  const;

	//Other functions
	void iterateDetector(double simtime, double dt, int blockSize); //increment time, track overall sim time, or take an argument??
	void iterateDetectorCPU(const DBL2DV& particleData, seconds simtime, seconds dt);
	void copyDataToHost(); //some sort of sim time check to verify I have iterated for the current sim time??
	void saveDataToDisk(string folder);
	void loadDataFromDisk(string folder);

	DBL2DV removeZerosData();
	void serialize(ofstream& out) const;
};

#undef STRVEC
#undef DBLVEC
#undef DBL2DV
#undef DBL3DV

#endif