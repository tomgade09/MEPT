#include "API/SimulationAPI.h"

#include "utils/writeIOclasses.h"
#include "utils/strings.h"
#include "ErrorHandling/simExceptionMacros.h"

#include <cmath>

using utils::fileIO::CSV;
using utils::strings::strToFPVec;

vector<vector<vector<double>>> satdata_g;
vector<vector<vector<vector<double>>>> particledata_g;  //particle <orig/curr <attribute <particle number> > >

//Simulation Management Functions
DLLEXP_EXTC Sim* createSimulationAPI(double dt, double simMin, double simMax, const char* rootdir)
{
	SIM_API_EXCEP_CHECK(return new Sim(dt, simMin, simMax, rootdir));
	return nullptr; //if above fails
}

DLLEXP_EXTC Sim* loadCompletedSimDataAPI(const char* fileDir)
{
	SIM_API_EXCEP_CHECK(return new Sim(fileDir));
	return nullptr; //if above fails
}

DLLEXP_EXTC void initializeSimulationAPI(Sim* sim) {
	SIM_API_EXCEP_CHECK(sim->initializeSimulation()); }

DLLEXP_EXTC void iterateSimCPUAPI(Sim* sim, int numberOfIterations, int itersBtwCouts) {
	SIM_API_EXCEP_CHECK(sim->__iterateSimCPU(numberOfIterations, itersBtwCouts)); }

DLLEXP_EXTC void iterateSimulationAPI(Sim* sim, int numberOfIterations, int itersBtwCouts) {
	SIM_API_EXCEP_CHECK(sim->iterateSimulation(numberOfIterations, itersBtwCouts)); }

DLLEXP_EXTC void freeGPUMemoryAPI(Sim* sim) {
	SIM_API_EXCEP_CHECK(sim->freeGPUMemory()); }

DLLEXP_EXTC void saveDataToDiskAPI(Sim* sim) {
	SIM_API_EXCEP_CHECK(sim->saveDataToDisk()); }

DLLEXP_EXTC void terminateSimulationAPI(Sim* sim) {
	SIM_API_EXCEP_CHECK(delete sim); }

DLLEXP_EXTC void setupExampleSimulationAPI(Sim* sim, int numParts, const char* loadFileDir)
{
	sim->getLog()->createEntry("API: setupExampleSimulation");
	//SIM_API_EXCEP_CHECK(
	sim->setBFieldModel("DipoleBLUT", { 72.0, 637.12, 1000000 });
	//sim->setBFieldModel("DipoleB", { 72.0 });
	//sim->addEFieldModel("QSPS", { 3185500.0, 6185500.0, 0.02, 6556500.0, 9556500.0, 0.04 });
	
	sim->createParticlesType("elec", MASS_ELECTRON, -1 * CHARGE_ELEM, numParts);

	sim->createSatellite(sim->simMin(), true, "btmElec", numParts);
	sim->createSatellite(sim->simMax(), false, "topElec", numParts);
	sim->createSatellite(4071307.04106411, false, "4e6ElecUpg", numParts);
	sim->createSatellite(4071307.04106411, true, "4e6ElecDng", numParts);

	//sim->particles(0)->setParticlesSource_s(sim->simMin(), sim->simMax());
	//std::cout << "\n\nIn setupExampleSimulationAPI - Particles::generateDist(96, 0.5, 4.5, 36000, 180.0, 0.0, sim->simMin(), sim->simMax())\n\n\n";
	//sim->particles(0)->generateDist(96, 0.5, 4.5, 36000, 180.0, 0.0, sim->simMin(), sim->simMax());

	ParticleDistribution pd{ "./", sim->particles(0)->attributeNames(), sim->particles(0)->name(),
		sim->particles(0)->mass(), { 0.0, 0.0, 0.0, 0.0, -1.0 }, false };

	sim->getLog()->createEntry("Created Particle Distribution:");
	sim->getLog()->createEntry("     Energy: " + std::to_string(96) + " E bins, " + std::to_string(0.5) + " - " + std::to_string(4.5) + " logE");
	sim->getLog()->createEntry("     Pitch : " + std::to_string(numParts / (96 * 2)) + " E bins, " + std::to_string(180) + " - " + std::to_string(90) + " deg");
	sim->getLog()->createEntry("     Pitch2: " + std::to_string(numParts / (96 * 2)) + " E bins, " + std::to_string(16) + " - " + std::to_string(0) + " deg");
	
	pd.addEnergyRange(96, 0.5, 4.5);
	//pd.addPitchRange(numParts / (96 * 2), 180.0, 90.0);
	//pd.addPitchRange(numParts / (96 * 2), 16.0, 0.0);
	pd.addPitchRange(36000, 0.0f, 180.0f);

	sim->particles(0)->loadDistFromPD(pd, sim->simMin(), sim->simMax());
	//); /* SIM_API_EXCEP_CHECK() */
}

DLLEXP_EXTC void setupSingleElectronAPI(Sim* sim, double vpara, double vperp, double s, double t_inc)
{ //check that satellites and B/E fields have been created here??
	//SIM_API_EXCEP_CHECK(
		fp2Dvec attrs{ fp2Dvec({ { (flPt_t)vpara },{ (flPt_t)vperp },{ (flPt_t)s },{ (flPt_t)0.0 },{ (flPt_t)t_inc } }) };
		sim->particles("elec")->loadDataFromMem(attrs, true);
		sim->particles("elec")->loadDataFromMem(attrs, false);
	//); /* SIM_API_EXCEP_CHECK() */
}


//Field Management Functions
DLLEXP_EXTC void setBFieldModelAPI(Sim* sim, const char* modelName, const char* fpString) {
	SIM_API_EXCEP_CHECK(sim->setBFieldModel(modelName, strToFPVec(fpString))); }

DLLEXP_EXTC void addEFieldModelAPI(Sim* sim, const char* modelName, const char* fpString) {
	SIM_API_EXCEP_CHECK(sim->addEFieldModel(modelName, strToFPVec(fpString))); }

DLLEXP_EXTC double getBFieldAtSAPI(Sim* sim, double s, double time)
{
	SIM_API_EXCEP_CHECK(return (double)sim->getBFieldAtS(s, time));
	return 0.0; //if above fails
}

DLLEXP_EXTC double getEFieldAtSAPI(Sim* sim, double s, double time)
{
	SIM_API_EXCEP_CHECK(return (double)sim->getEFieldAtS(s, time));
	return 0.0; //if above fails
}


//Particles Management Functions
DLLEXP_EXTC void createParticlesTypeAPI(Sim* sim, const char* name, double mass, double charge, long numParts, const char* loadFileDir) {
	SIM_API_EXCEP_CHECK(sim->createParticlesType(name, mass, charge, numParts, loadFileDir)); }


//Satellite Management Functions
DLLEXP_EXTC void createSatelliteAPI(Sim* sim, double altitude, bool upwardFacing, const char* name, int particleCount) {
	SIM_API_EXCEP_CHECK(sim->createSatellite(altitude, upwardFacing, name, particleCount)); }

DLLEXP_EXTC int  getNumberOfSatellitesAPI(Sim* sim)
{
	SIM_API_EXCEP_CHECK(return (int)(sim->getNumberOfSatellites()));
	return -1; //if above fails
}

DLLEXP_EXTC const double* getSatelliteDataPointersAPI(Sim* sim, int satelliteInd, int attributeInd)
{
	if (sizeof(flPt_t) != sizeof(double))
	{   //not wild about this workaround to cast a double nested vector to a double type, but it works
		SIM_API_EXCEP_CHECK(
		const SatDataVecs& tmp{ sim->getSatelliteData(satelliteInd) };
		
		if (satdata_g.size() != (size_t)sim->getNumberOfSatellites())
			satdata_g.resize(sim->getNumberOfSatellites());

		if (satdata_g.at(satelliteInd).size() != (size_t)SatDataVecs::size)
			satdata_g.at(satelliteInd).resize(SatDataVecs::size);

		if (satdata_g.at(satelliteInd).at(attributeInd).size() != tmp.__at(attributeInd).size())
			satdata_g.at(satelliteInd).at(attributeInd).resize(tmp.__at(attributeInd).size());
		
		for (size_t i = 0; i < tmp.__at(attributeInd).size(); i++)
			satdata_g.at(satelliteInd).at(attributeInd).at(i) = (double)tmp.__at(attributeInd).at(i);
		
		return satdata_g.at(satelliteInd).at(attributeInd).data();
		);
	}
	else
	{
		SIM_API_EXCEP_CHECK(
		return reinterpret_cast<const double*>(sim->getSatelliteData(satelliteInd).__at(attributeInd).data())
		);
	}

	return nullptr; //if above fails
}


//Access Functions
DLLEXP_EXTC double getSimTimeAPI(Sim* sim)
{
	SIM_API_EXCEP_CHECK(return (double)sim->simtime());
	return -1.0; //if above fails
}

DLLEXP_EXTC double getDtAPI(Sim* sim)
{
	SIM_API_EXCEP_CHECK(return (double)sim->dt());
	return -1.0; //if above fails
}

DLLEXP_EXTC double getSimMinAPI(Sim* sim)
{
	SIM_API_EXCEP_CHECK(return (double)sim->simMin());
	return -1.0; //if above fails
}

DLLEXP_EXTC double getSimMaxAPI(Sim* sim)
{
	SIM_API_EXCEP_CHECK(return (double)sim->simMax());
	return -1.0; //if above fails
}

DLLEXP_EXTC int getNumberOfParticleTypesAPI(Sim* sim)
{
	SIM_API_EXCEP_CHECK(return (int)(sim->getNumberOfParticleTypes()));
	return -1; //if above fails
}

DLLEXP_EXTC int getNumberOfParticlesAPI(Sim* sim, int partInd)
{
	SIM_API_EXCEP_CHECK(return (int)(sim->getNumberOfParticles(partInd)));
	return -1; //if above fails
}

DLLEXP_EXTC int getNumberOfAttributesAPI(Sim* sim, int partInd)
{
	SIM_API_EXCEP_CHECK(return (int)(sim->getNumberOfAttributes(partInd)));
	return -1; //if above fails
}

DLLEXP_EXTC const char* getParticlesNameAPI(Sim* sim, int partInd)
{
	SIM_API_EXCEP_CHECK(return sim->getParticlesName(partInd).c_str());
	return nullptr; //if above fails
}

DLLEXP_EXTC const char* getSatelliteNameAPI(Sim* sim, int satInd)
{
	SIM_API_EXCEP_CHECK(return sim->getSatelliteName(satInd).c_str());
	return nullptr; //if above fails
}

DLLEXP_EXTC const double* getPointerToParticlesAttributeArrayAPI(Sim* sim, int partIndex, int attrIndex, bool originalData)
{
	if (sizeof(flPt_t) != sizeof(double))
	{   //not wild about this workaround to cast a double nested vector to a double type, but it works
		SIM_API_EXCEP_CHECK(
		const vector<vector<flPt_t>>& tmp{ sim->getParticleData(partIndex, originalData) };
		
		if (particledata_g.size() != (size_t)sim->getNumberOfParticleTypes())
			particledata_g.resize(sim->getNumberOfParticleTypes());

		if (particledata_g.at(partIndex).size() != 2)
			particledata_g.at(partIndex).resize(2);  //for curr / orig data

		if (particledata_g.at(partIndex).at(originalData).size() != tmp.size())
			particledata_g.at(partIndex).at(originalData).resize(tmp.size());  //number of attributes

		if (particledata_g.at(partIndex).at(originalData).at(attrIndex).size() != tmp.at(attrIndex).size())
			particledata_g.at(partIndex).at(originalData).at(attrIndex).resize(tmp.at(attrIndex).size());
		
		for (size_t i = 0; i < tmp.at(attrIndex).size(); i++)
			particledata_g.at(partIndex).at(originalData).at(attrIndex).at(i) = (double)tmp.at(attrIndex).at(i);
		
		return particledata_g.at(partIndex).at(originalData).at(attrIndex).data();
		);
	}
	else
	{
		SIM_API_EXCEP_CHECK(
		return reinterpret_cast<const double*>(sim->getParticleData(partIndex, originalData).at(attrIndex).data())
		);
	}

	return nullptr; //if above fails
}


//CSV functions
DLLEXP_EXTC void writeCommonCSVAPI(Sim* sim)
{
	SIM_API_EXCEP_CHECK(
		CSV csvtmp("./elecoutput.csv");
		fp2Dvec origData{ sim->getParticleData(0, true) };
		csvtmp.add({ origData.at(0), origData.at(1), origData.at(2) }, { "vpara orig", "vperp orig", "s orig" });
		csvtmp.addspace();
		
		const SatDataVecs& btmElecSat{ sim->getSatelliteData(0) };
		csvtmp.add({ btmElecSat.t_detect, btmElecSat.vpara, btmElecSat.mu, btmElecSat.s },
			{ "t_esc btm", "vpara btm", "vperp btm", "s btm" });
		csvtmp.addspace();

		const SatDataVecs& topElecSat{ sim->getSatelliteData(1) };
		csvtmp.add({ topElecSat.t_detect, topElecSat.vpara, topElecSat.mu, topElecSat.s },
			{ "t_esc top", "vpara top", "vperp top", "s top" });
		csvtmp.addspace();

		fp2Dvec energyPitch(2, std::vector<flPt_t>(origData.at(0).size()));
		for (size_t elem = 0; elem < energyPitch.at(0).size(); elem++)
		{
			energyPitch.at(0).at(elem) = 0.5 * MASS_ELECTRON * (pow(origData.at(0).at(elem), 2) + pow(origData.at(1).at(elem), 2)) / JOULE_PER_EV;
			energyPitch.at(1).at(elem) = atan2(abs(origData.at(1).at(elem)), -origData.at(0).at(elem)) / RADS_PER_DEG;
		}
		csvtmp.add(energyPitch, { "Energy (eV)", "Pitch Angle" });
	);
}

//#ifdef _DEBUG
//int main()
//{
//	/*SIM_API_EXCEP_CHECK(
//		auto sim{ std::make_unique<Simulation>(0.01, 101322.378940846, 19881647.2473464, "./out/") };
//		setupExampleSimulationAPI(sim.get(), 3456000, "./../_in/data");
//		//sim->addEFieldModel("QSPS", { 3185500.0, 6185500.0, 0.02, 6556500.0, 9556500.0, 0.04 });
//		sim->initializeSimulation();
//		sim->iterateSimulation(5000, 500);
//	);*/ /* SIM_API_EXCEP_CHECK() */
//
//	auto sim{ "../_dataout/200314_11.10.53/" };
//
//	return 0;
//}
//#endif
