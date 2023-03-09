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
using std::shared_ptr;

template <typename T1>
struct SatData
{
	static constexpr int size{ 4 }; //number of elements in this struct
	
	T1 vpara;
	T1 mu;
	T1 s;
	T1 t_detect;

	SatData() = default;
	SatData(const T1& vpara_i, const T1& mu_i, const T1& s_i, const T1& t_i);

	T1& at(size_t loc);
	const T1& __at(size_t loc) const; //the const version, for API access
};

typedef SatData<flPt_t*>        SatDataPtrs;
typedef SatData<vector<flPt_t>> SatDataVecs;

struct Sat_d
{
	SatDataPtrs capture_d;
	meters      altitude;
	bool        upward;
};

class Satellite
{
protected:
	string name_m;
	
	meters altitude_m{ 0.0f };
	bool   upwardFacing_m{ false };
	bool   initializedGPU_m{ false };

	size_t numberOfParticles_m{ 0 };
	size_t numGPUs_m;
	vector<size_t> particleCountPerGPU_m;
	
	SatDataVecs          data_m;         //overall data
	const strvec         names_m{ "vpara", "vperp", "s", "time" }; //data names on GPU
	vector<flPt_t*>      gpuMemRegion_d; //flattened satellite capture data on GPU
	vector<SatDataPtrs>  gpuDataPtrs_d;  //pointers to data arrays on GPU, stored as struct of arrays (SOA)
										 //first pointer (vpara) is also the pointer for the entire GPU mem space

	void   initializeGPU();
	void   freeGPUMemory();
	void   deserialize(ifstream& in);

public:
	Satellite(string& name, meters altitude, bool upwardFacing, size_t numberOfParticles);
	Satellite(ifstream& in);
	~Satellite();
	Satellite(const Satellite&) = delete;
	Satellite& operator=(const Satellite&) = delete;

	//Access functions
	string        name() const;
	meters        altitude() const;
	bool	      upward() const;
	SatDataVecs&       __data();
	const SatDataVecs& data() const;
	size_t        getNumberOfAttributes() const;
	size_t        getNumberOfParticles()  const;
	size_t		  getNumParticlesPerGPU(int GPUind) const;
	Sat_d         getSat_d(int GPUind) const;
	SatDataPtrs   getPtrs_d(int GPUind) const;
	
	//Other functions
	void iterateDetectorCPU(const fp2Dvec& particleData, seconds simtime, seconds dt); //need to remove as well
	void copyDataToHost(); //some sort of sim time check to verify I have iterated for the current sim time??
	void saveDataToDisk(string folder);
	void loadDataFromDisk(string folder);

	SatDataVecs removeZerosData(vector<int>& indices);
	void serialize(ofstream& out) const;
};

#endif