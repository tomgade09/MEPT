#ifndef PARTICLE_H
#define PARTICLE_H

#include "utils/unitsTypedefs.h"
#include "utils/writeIOclasses.h"

using std::vector;
using std::string;
using std::ifstream;
using std::ofstream;
using utils::fileIO::ParticleDistribution;

class Particles
{
protected:	
	string name_m; //name of particle

	strvec  attributeNames_m;
	fp2Dvec origData_m; //initial data - not modified, but eventually saved to disk
	fp2Dvec currData_m; //current data - that which is being updated by iterating the sim

	bool initDataLoaded_m{ false };
	bool initializedGPU_m{ false };

	kg      mass_m;
	coulomb charge_m;
	size_t  numberOfParticles_m;
	size_t  numGPUs_m;
	vector<size_t> particleCountPerGPU_m;
	
	//device pointers
	vector<flPt_t*>  currData1D_d; //pointer to the contiguous region of memory that is subdivided into equal sized subarrays
	vector<flPt_t**> currData2D_d; //pointer to the array of pointers _on device_ - i.e. from host, these ptrs within the array cannot be read

	void initializeGPU(); //need to modify all of these below to account for multi GPU
	void copyDataToGPU(bool currToGPU = false);
	void freeGPUMemory();
	void deserialize(ifstream& in);

public:
	Particles(string name, const strvec& attributeNames, kg mass, coulomb charge, size_t numParts);
	Particles(ifstream& in);
	~Particles();
	Particles(const Particles&) = delete;
	Particles& operator=(const Particles& otherpart) = delete;

	//Access functions
	string         name()           const;
	const strvec&  attributeNames() const;
	fp2Dvec&        __data(bool orig);
	const fp2Dvec& data(bool orig) const;
	kg             mass()          const;
	coulomb        charge()        const;
	size_t         getNumberOfAttributes() const;
	size_t         getNumberOfParticles()  const;
	size_t         getNumParticlesPerGPU(size_t GPUind) const;
	vector<size_t> getNumParticlesPerGPU() const;
	bool           getInitDataLoaded() const;
	flPt_t**       getCurrDataGPUPtr(size_t GPUind) const;

	size_t         getAttrIndByName(string searchName) const;
	string         getAttrNameByInd(size_t searchIndx) const;

	//Other functions
	void setParticlesSource_s(meters s_ion, meters s_mag);

	void generateDist(size_t numEbins, eV E_min, eV E_max, size_t numPAbins, degrees PA_min, degrees PA_max, meters s_ion, meters s_mag);
	void loadDistFromPD(const ParticleDistribution& pd, meters s_ion, meters s_mag);
	void loadDistFromPD(const ParticleDistribution& pd, vector<meters>& s);
	void loadDistFromDisk(string folder, string distName, meters s_ion, meters s_mag);
	void loadDataFromMem(const fp2Dvec& data, bool orig = true);
	void loadDataFromDisk(string folder, bool orig = true);
	void saveDataToDisk(string folder, bool orig) const;
	void copyDataToHost(); //needs to be public because Particles doesn't know when things are done modifying GPU data
	void serialize(ofstream& out) const;
};

#endif