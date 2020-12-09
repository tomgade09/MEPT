#ifndef PARTICLE_H
#define PARTICLE_H

#include <vector>
#include <string>

#include "utils/unitsTypedefs.h"
#include "utils/writeIOclasses.h"

using std::vector;
using std::string;
using std::ifstream;
using std::ofstream;
using utils::fileIO::ParticleDistribution;

#define STRVEC vector<string>
#define FLTVEC vector<float>
#define FLT2DV vector<vector<float>>

class Particles
{
protected:	
	string name_m; //name of particle

	STRVEC attributeNames_m;
	FLT2DV origData_m; //initial data - not modified, but eventually saved to disk
	FLT2DV currData_m; //current data - that which is being updated by iterating the sim

	bool initDataLoaded_m{ false };
	bool initializedGPU_m{ false }; //consider what to do with this with multi GPU - still necessary?

	float mass_m;
	float charge_m;
	size_t numberOfParticles_m;
	size_t numGPUs_m;
	vector<size_t> particleCountPerGPU_m;
	
	//device pointers
	vector<float*>  currData1D_d;
	vector<float**> currData2D_d;

	void initializeGPU(); //need to modify all of these below to account for multi GPU
	void copyDataToGPU(bool origToGPU = true);
	void freeGPUMemory();
	void deserialize(ifstream& in);

public:
	Particles::Particles(string name, vector<string> attributeNames, float mass, float charge, size_t numParts,
		size_t numGPUs, vector<size_t> particleCountPerGPU);
	Particles(ifstream& in);
	~Particles();
	Particles(const Particles&) = delete;
	Particles& operator=(const Particles& otherpart) = delete;

	//Access functions
	string         name()           const;
	const STRVEC&  attributeNames() const;
	FLT2DV&        __data(bool orig);
	const FLT2DV&  data(bool orig) const;
	float          mass()          const;
	float          charge()        const;
	size_t         getNumberOfAttributes() const;
	size_t         getNumberOfParticles()  const;
	size_t         getNumParticlesPerGPU(size_t GPUind) const;
	vector<size_t> getNumParticlesPerGPU() const;
	bool           getInitDataLoaded() const;
	float**        getCurrDataGPUPtr(size_t GPUind) const;

	size_t         getAttrIndByName(string searchName) const;
	string         getAttrNameByInd(size_t searchIndx) const;

	//Other functions
	void setParticlesSource_s(float s_ion, float s_mag);

	void generateDist(size_t numEbins, eV E_min, eV E_max, size_t numPAbins, degrees PA_min, degrees PA_max, meters s_ion, meters s_mag);
	void loadDistFromPD(const ParticleDistribution& pd, meters s_ion, meters s_mag);
	void loadDistFromPD(const ParticleDistribution& pd, vector<meters>& s);
	void loadDistFromDisk(string folder, string distName, meters s_ion, meters s_mag);
	void loadDataFromMem(vector<vector<float>> data, bool orig = true);
	void loadDataFromDisk(string folder, bool orig = true);
	void saveDataToDisk(string folder, bool orig) const;
	void copyDataToHost(); //needs to be public because Particles doesn't know when things are done modifying GPU data
	void serialize(ofstream& out) const;
};

#undef STRVEC
#undef FLTVEC
#undef FLT2DV

#endif