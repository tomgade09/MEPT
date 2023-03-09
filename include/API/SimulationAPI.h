#ifndef SIMULATIONAPI_H
#define SIMULATIONAPI_H

#include "dlldefines.h"
#include "Simulation/Simulation.h"

typedef Simulation Sim;

//Simulation Management Functions
DLLEXP_EXTC Sim* createSimulationAPI(double dt, double simMin, double simMax, const char* rootdir);
DLLEXP_EXTC Sim* loadCompletedSimDataAPI(const char* fileDir);
DLLEXP_EXTC void initializeSimulationAPI(Sim* sim);
DLLEXP_EXTC void iterateSimCPUAPI(Sim* sim, int numberOfIterations, int itersBtwCouts);
DLLEXP_EXTC void iterateSimulationAPI(Sim* sim, int numberOfIterations, int itersBtwCouts);
DLLEXP_EXTC void freeGPUMemoryAPI(Sim* sim);
DLLEXP_EXTC void saveDataToDiskAPI(Sim* sim);
DLLEXP_EXTC void terminateSimulationAPI(Sim* sim);
DLLEXP_EXTC void setupExampleSimulationAPI(Sim* sim, int numParts, const char* loadFileDir);
DLLEXP_EXTC void setupSingleElectronAPI(Sim* sim, double vpara, double vperp, double s, double t_inc);
//build API function for resetSimulation


//Field Management Functions
DLLEXP_EXTC double getBFieldAtSAPI(Sim* sim, double s, double time);
DLLEXP_EXTC double getEFieldAtSAPI(Sim* sim, double s, double time);
DLLEXP_EXTC void   setBFieldModelAPI(Sim* sim, const char* modelName, const char* floatString); //switch to comma delimited string of variables
DLLEXP_EXTC void   addEFieldModelAPI(Sim* sim, const char* modelName, const char* floatString);


//Particles Management Functions
DLLEXP_EXTC void createParticlesTypeAPI(Sim* sim, const char* name, double mass, double charge, long numParts, const char* loadFileDir = "");


//Satellite Management Functions
DLLEXP_EXTC void          createSatelliteAPI(Sim* sim, double altitude, bool upwardFacing, const char* name, int particleCount);
DLLEXP_EXTC int           getNumberOfSatellitesAPI(Sim* sim);
DLLEXP_EXTC const double* getSatelliteDataPointersAPI(Sim* sim, int satelliteInd, int attributeInd);


//Access Functions
DLLEXP_EXTC double        getSimTimeAPI(Sim* sim);
DLLEXP_EXTC double        getDtAPI(Sim* sim);
DLLEXP_EXTC double        getSimMinAPI(Sim* sim);
DLLEXP_EXTC double        getSimMaxAPI(Sim* sim);
DLLEXP_EXTC int           getNumberOfParticleTypesAPI(Sim* sim);
DLLEXP_EXTC int           getNumberOfParticlesAPI(Sim* sim, int partInd);
DLLEXP_EXTC int           getNumberOfAttributesAPI(Sim* sim, int partInd);
DLLEXP_EXTC const char*   getParticlesNameAPI(Sim* sim, int partInd);
DLLEXP_EXTC const char*   getSatelliteNameAPI(Sim* sim, int satInd);
DLLEXP_EXTC const double* getPointerToParticlesAttributeArrayAPI(Sim* sim, int partIndex, int attrIndex, bool originalData);


//CSV Functions
DLLEXP_EXTC void writeCommonCSVAPI(Sim* sim);

#endif//end if for header guard