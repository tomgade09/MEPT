#ifndef SIMULATIONCLASS_H
#define SIMULATIONCLASS_H

#include <vector>
#include <memory> //smart pointers

#include "dlldefines.h"
#include "BField/allBModels.h"
#include "EField/allEModels.h"
#include "Particles/Particles.h"
#include "Satellite/Satellite.h"
#include "Log/Log.h"

using std::string;
using std::vector;
using std::unique_ptr;
using std::shared_ptr;

class Simulation
{
protected:
	//Structs and classes that fill various roles	
	enum class Component
	{
		BField,
		EField,
		Log,
		Particles,
		Satellite
	};

	/*struct TempSat
	{//Struct that holds data to create satellite - allows satellites to be added in any order, but ensure they are created before particles
		size_t particleInd;
		meters altitude;
		bool   upwardFacing;
		string name;

		TempSat(size_t partInd, meters alt, bool upward, string nameStr) :
			particleInd{ partInd }, altitude{ alt }, upwardFacing{ upward }, name{ nameStr } {}
	};

	struct SatandPart
	{//Satellite needs particle-specific data associated with it, so this struct holds a shared_ptr to the particle
		unique_ptr<Satellite> satellite;
		shared_ptr<Particles> particle;

		SatandPart(unique_ptr<Satellite> sat, shared_ptr<Particles> part) :
			satellite{ std::move(sat) }, particle{ std::move(part) } {}
	};*/

	//Simulation Characteristics
	seconds dt_m{ 0.0f };
	meters  simMin_m{ 0.0f };
	meters  simMax_m{ 0.0f };
	string  saveRootDir_m{ "./" };
	seconds simTime_m{ 0.0f };

	//Flags
	bool initialized_m{ false };
	bool dataOnGPU_m{ false };
	bool saveReady_m{ false };
	bool previousSim_m{ false };

	//GPU Data
	size_t  gpuCount_m{ 0 };
	//fp1Dvec computeSplit_m;

	//Simulation-specific classes tracked by Simulation
	unique_ptr<BModel>             BFieldModel_m;
	unique_ptr<EField>             EFieldModel_m;
	unique_ptr<Log>                Log_m;
	vector<unique_ptr<Particles>>  particles_m;
	//vector<unique_ptr<TempSat>>    tempSats_m; //holds data until the GPU data arrays are allocated, allows the user more flexibility of when to call createSatellitesAPI
	//vector<unique_ptr<SatandPart>> satPartPairs_m;
	vector<unique_ptr<Satellite>>  satellites_m;

	//Protected functions
	void incTime();
	void printSimAttributes(size_t numberOfIterations, size_t itersBtwCouts);
	void loadSimulation(string saveRootDir);
	void loadDataFromDisk();

	void setupGPU();

public:
	Simulation(seconds dt, meters simMin, meters simMax, string rootdir);
	Simulation(string saveRootDir); //for loading previous simulation data
	virtual ~Simulation();

	Simulation(const Simulation&) = delete;
	Simulation& operator=(const Simulation&) = delete;

	class Environment;

	//
	//======== Simulation Access Functions ========//
	//
	seconds simtime() const;
	seconds dt()      const;
	meters  simMin()  const;
	meters  simMax()  const;

	//Class data
	int    getNumberOfParticleTypes()         const;
	int    getNumberOfSatellites()            const;
	int    getNumberOfParticles(int partInd)  const;
	int    getNumberOfAttributes(int partInd) const;
	string getParticlesName(int partInd)      const;
	string getSatelliteName(int satInd)       const;
	//int    getParticleIndexOfSat(int satInd)  const;

	//Class pointers
	Particles* particles(int partInd) const;
	Particles* particles(string name) const; //search for name, return particle
	Satellite* satellite(int satInd)  const;
	Satellite* satellite(string name) const; //search for name, return satellite
	BModel*    Bmodel()   const;
	EField*    Efield()   const;
	Log*	   getLog();

	//Simulation data
	const fp2Dvec&     getParticleData(size_t partInd, bool originalData);
	const SatDataVecs& getSatelliteData(size_t satInd);

	//Fields data
	tesla getBFieldAtS(meters s, seconds time) const;
	Vperm getEFieldAtS(meters s, seconds time) const;

	//
	//======== Class Creation Functions ========//
	//
	void createParticlesType(string name, kg mass, coulomb charge, size_t numParts, string loadFilesDir = "");
	//void createTempSat(string partName, meters altitude, bool upwardFacing, string name);
	//void createTempSat(size_t partInd, meters altitude, bool upwardFacing, string name);
	void createSatellite(meters altitude, bool upwardFacing, string name, size_t totalParticleCount);
	void setBFieldModel(string name, fp1Dvec args);
	//void setBFieldModel(unique_ptr<BModel> BModelptr);
	void addEFieldModel(string name, fp1Dvec args);

	//
	//======== Simulation Management Functions ========//
	//
	void initializeSimulation();
	void __iterateSimCPU(size_t numberOfIterations, size_t checkDoneEvery);
	void iterateSimulation(size_t numberOfIterations, size_t itersBtwCouts);
	void saveDataToDisk();
	void freeGPUMemory();
	void resetSimulation(bool fields = false);
	void saveSimulation() const;
};//end class
#endif //end header guard