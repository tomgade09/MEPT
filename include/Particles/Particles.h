#ifndef PARTICLE_H
#define PARTICLE_H

#include <vector>
#include <string>

#include "dlldefines.h"
#include "utils/unitsTypedefs.h"
#include "utils/writeIOclasses.h"

using std::vector;
using std::string;
using std::ifstream;
using std::ofstream;
using utils::fileIO::ParticleDistribution;

#define STRVEC vector<string>
#define DBLVEC vector<double>
#define DBL2DV vector<vector<double>>

class Particles
{
protected:
	string name_m; //name of particle

	STRVEC attributeNames_m;
	DBL2DV origData_m; //initial data - not modified, but eventually saved to disk
	DBL2DV currData_m; //current data - that which is being updated by iterating the sim

	bool initDataLoaded_m{ false };
	bool initializedGPU_m{ false }; //consider what to do with this with multi GPU - still necessary?

	long numberOfParticles_m;
	double mass_m;
	double charge_m;

	//device pointers
	double*  currData1D_d{ nullptr }; //make vectors for handling multiple GPUs
	double** currData2D_d{ nullptr };

	void initializeGPU(); //need to modify all of these below to account for multi GPU
	void copyDataToGPU(bool origToGPU = true);
	void freeGPUMemory();
	void deserialize(ifstream& in);

public:
	Particles(string name, vector<string> attributeNames, double mass, double charge, long numParts);
	Particles(ifstream& in);
	~Particles();
	Particles(const Particles&) = delete;
	Particles& operator=(const Particles& otherpart) = delete;

	//Access functions
	string        name()           const;
	const STRVEC& attributeNames() const;
	DBL2DV&       __data(bool orig);
	const DBL2DV& data(bool orig) const;
	double        mass()          const;
	double        charge()        const;
	size_t        getNumberOfAttributes() const;
	long          getNumberOfParticles()  const;
	bool          getInitDataLoaded() const;
	double**      getCurrDataGPUPtr() const;

	size_t        getAttrIndByName(string searchName) const;
	string        getAttrNameByInd(size_t searchIndx) const;

	//Other functions
	void setParticlesSource_s(double s_ion, double s_mag);

	void generateDist(size_t numEbins, eV E_min, eV E_max, size_t numPAbins, degrees PA_min, degrees PA_max, meters s_ion, meters s_mag);
	void loadDistFromPD(const ParticleDistribution& pd, meters s_ion, meters s_mag);
	void loadDistFromPD(const ParticleDistribution& pd, vector<meters>& s);
	void loadDistFromDisk(string folder, string distName, meters s_ion, meters s_mag);
	void loadDataFromMem(vector<vector<double>> data, bool orig = true);
	void loadDataFromDisk(string folder, bool orig = true);
	void saveDataToDisk(string folder, bool orig) const;
	void copyDataToHost(); //needs to be public because Particles doesn't know when things are done modifying GPU data
	void serialize(ofstream& out) const;
};

#undef STRVEC
#undef DBLVEC
#undef DBL2DV

#endif