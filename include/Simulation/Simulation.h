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

#define GPU true// For Now this will tell the compiler to compile .cuda in .cpp files

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

	struct TempSat
	{//Struct that holds data to create satellite - allows satellites to be added in any order, but ensure they are created before particles
		size_t particleInd;
		float altitude;
		bool upwardFacing;
		string name;

		TempSat(size_t partInd, float alt, bool upward, string nameStr) :
			particleInd{ partInd }, altitude{ alt }, upwardFacing{ upward }, name{ nameStr } {}
	};

	struct SatandPart
	{//Satellite needs particle-specific data associated with it, so this struct holds a shared_ptr to the particle
		unique_ptr<Satellite> satellite;
		shared_ptr<Particles> particle;

		SatandPart(unique_ptr<Satellite> sat, shared_ptr<Particles> part) :
			satellite{ std::move(sat) }, particle{ std::move(part) } {}
	};

	//Simulation Characteristics
	float dt_m{ 0.0f };
	float simMin_m{ 0.0f };
	float simMax_m{ 0.0f };
	string saveRootDir_m{ "./" };
	float simTime_m{ 0.0f };

	//Flags
	bool initialized_m{ false };
	bool dataOnGPU_m{ false };
	bool saveReady_m{ false };
	bool previousSim_m{ false };

	//GPU Data
	int gpuCount_m{ 0 };
	vector<int>computeSplit_m;

	//Simulation-specific classes tracked by Simulation
	vector<unique_ptr<BModel>>     BFieldModel_m;
	unique_ptr<EField>             EFieldModel_m;
	unique_ptr<Log>                Log_m;
	vector<shared_ptr<Particles>>  particles_m;
	vector<unique_ptr<TempSat>>    tempSats_m; //holds data until the GPU data arrays are allocated, allows the user more flexibility of when to call createSatellitesAPI
	vector<unique_ptr<SatandPart>> satPartPairs_m;

	//Protected functions
	void createSatellite(TempSat* tmpsat, bool save = true);
	void incTime();
	void printSimAttributes(int numberOfIterations, int itersBtwCouts, string GPUName);
	void loadSimulation(string saveRootDir);
	void loadDataFromDisk();

	void setupGPU();

public:
	Simulation(float dt, float simMin, float simMax);
	Simulation(string saveRootDir); //for loading previous simulation data
	virtual ~Simulation();

	Simulation(const Simulation&) = delete;
	Simulation& operator=(const Simulation&) = delete;

	class Environment;

	//
	//======== Simulation Access Functions ========//
	//
	float simtime() const;
	float dt()      const;
	float simMin()  const;
	float simMax()  const;

	//Class data
	int    getNumberOfParticleTypes()         const;
	int    getNumberOfSatellites()            const;
	int    getNumberOfParticles(int partInd)  const;
	int    getNumberOfAttributes(int partInd) const;
	string getParticlesName(int partInd)      const;
	string getSatelliteName(int satInd)       const;
	int    getParticleIndexOfSat(int satInd)  const;

	//Class pointers
	Particles* particles(int partInd) const;
	Particles* particles(string name) const; //search for name, return particle
	Particles* particles(Satellite* satellite) const;
	Satellite* satellite(int satInd)  const;
	Satellite* satellite(string name) const; //search for name, return satellite
	BModel*    Bmodel()               const;
	EField*    Efield()               const;
	Log*	   getLog();

	//Simulation data
	const vector<vector<float>>& getParticleData(size_t partInd, bool originalData);
	const vector<vector<float>>& getSatelliteData(size_t satInd);

	//Fields data
	float getBFieldAtS(float s, float time) const;
	float getEFieldAtS(float s, float time) const;

	//
	//======== Class Creation Functions ========//
	//
	void createParticlesType(string name, float mass, float charge, size_t numParts, string loadFilesDir = "", bool save = true);
	void createTempSat(string partName, float altitude, bool upwardFacing, string name);
	void createTempSat(size_t partInd, float altitude, bool upwardFacing, string name);
	void setBFieldModel(string name, vector<float> args, bool save = true);
	void setBFieldModel(unique_ptr<BModel> BModelptr);
	void addEFieldModel(string name, vector<float> args, bool save = true);

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