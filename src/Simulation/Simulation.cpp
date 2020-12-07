#include "Simulation/Simulation.h"

#include <iomanip>
#include <filesystem>

#include "utils/loopmacros.h"
#include "utils/arrayUtilsGPU.h"
#include "utils/serializationHelpers.h"
#include "ErrorHandling/simExceptionMacros.h"

using std::cout;
using std::cerr;
using std::endl;
using std::move;
using std::put_time;
using std::to_string;
using std::make_unique;
using std::make_shared;
using std::logic_error;
using std::out_of_range;
using std::runtime_error;
using std::invalid_argument;
using std::filesystem::path;
using std::filesystem::exists;
using std::filesystem::absolute;
using std::chrono::system_clock;
using std::filesystem::create_directory;

using namespace utils::fileIO::serialize;

Simulation::Simulation(float dt, float simMin, float simMax) :
	dt_m{ dt }, simMin_m{ simMin }, simMax_m{ simMax }
{
	//generate a folder for output data in {ROOT}/_dataout
	stringstream dataout;
	auto in_time_t = system_clock::to_time_t(system_clock::now());
	dataout << "../_dataout/" << put_time(localtime(&in_time_t), "%C%m%d_%H.%M.%S/");
	cout << "===============================================================" << "\n";
	cout << "Data Folder:    " << dataout.str() << "\n";

	path datadir{ dataout.str() };
	if (!exists(datadir))
	{
		try
		{
			create_directory(datadir);

			create_directory(path(dataout.str() + "ADPIC"));
			create_directory(path(dataout.str() + "bins"));
			create_directory(path(dataout.str() + "bins/particles_final"));
			create_directory(path(dataout.str() + "bins/particles_init"));
			create_directory(path(dataout.str() + "bins/satellites"));
		}
		catch (std::exception& e)
		{
			cout << "Exception: " << e.what() << "\n";
			cout << "Ensure you are running from \"PTEM/bin\".  Absolute path: " << absolute(datadir) << "\n";
			exit(1);
		}
	}

	saveRootDir_m = dataout.str();
	Log_m = make_unique<Log>(saveRootDir_m + "simulation.log");

	// Setup GPU Information ( number of devices and compute capability of each device )
	setupGPU();
}

Simulation::Simulation(string saveRootDir) : saveRootDir_m{ saveRootDir + "/" },
	previousSim_m{ true }, Log_m{ make_unique<Log>(saveRootDir_m + "reload.log") }
{
	SIM_API_EXCEP_CHECK(loadSimulation(saveRootDir_m));
	loadDataFromDisk();
	setupGPU();
}

Simulation::~Simulation()
{
	SIM_API_EXCEP_CHECK(if (!previousSim_m) saveSimulation());
	if (!previousSim_m && saveReady_m) saveDataToDisk(); //save data if it hasn't been done
	else if (!previousSim_m) cerr << "Warning: Simulation::~Simulation: saveReady_m is false.";
	Log_m->createEntry("End simulation");
}


void Simulation::printSimAttributes(int numberOfIterations, int itersBtwCouts) //protected
{
	//Sim Header (folder) printed from Python - move here eventually
	cout << "GPU(s) In Use:  " << utils::GPU::getDeviceNames() << endl;
	cout << "Sim between:    " << simMin_m << "m - " << simMax_m << "m" << endl;
	cout << "dt:             " << dt_m << "s" << endl;
	cout << "BModel Model:   " << BFieldModel_m.at(0)->name() << endl;
	cout << "EField Elems:   " << ((Efield()->qspsCount() > 0) ? "QSPS: " + to_string(Efield()->qspsCount()) : "") << endl;
	cout << "Particles:      ";
	for (size_t iii = 0; iii < particles_m.size(); iii++)
	{
		cout << ((iii != 0) ? "                " : "") << particles_m.at(iii)->name() << ": #: " << particles_m.at(iii)->getNumberOfParticles() << ", loaded files?: " << (particles_m.at(iii)->getInitDataLoaded() ? "true" : "false") << std::endl;
	}
	cout << "Satellites:     ";
	for (int iii = 0; iii < getNumberOfSatellites(); iii++)
	{
		cout << ((iii != 0) ? "                " : "") << satellite(iii)->name() << ": alt: " << satellite(iii)->altitude() << " m, upward?: " << (satellite(iii)->upward() ? "true" : "false") << std::endl;
	}
	cout << "Iterations:     " << numberOfIterations << endl;
	cout << "Iters Btw Cout: " << itersBtwCouts << endl;
	cout << "Time to setup:  " << Log_m->timeElapsedTotal_s() << " s" << endl;
	cout << "===============================================================" << endl;
}

void Simulation::incTime()
{
	simTime_m += dt_m;
}


//
//======== Simulation Access Functions ========//
//
float Simulation::simtime() const
{
	return simTime_m;
}

float Simulation::dt() const
{
	return dt_m;
}

float Simulation::simMin() const
{
	return simMin_m;
}

float Simulation::simMax() const
{
	return simMax_m;
}


//Class data
int Simulation::getNumberOfParticleTypes() const
{
	return (int)particles_m.size();
}

int Simulation::getNumberOfSatellites() const
{
	if (gpuCount_m > 0)
	{
		// Size should be a perfect multiple so don't need to worry about rounding
		return (int)satPartPairs_m.size() / gpuCount_m;
	}
	else
	{
		return (int)satPartPairs_m.size();
	}
}

int Simulation::getNumberOfParticles(int partInd) const
{
	return (int)particles_m.at(partInd)->getNumberOfParticles();
}

int Simulation::getNumberOfAttributes(int partInd) const
{
	return (int)particles_m.at(partInd)->getNumberOfAttributes();
}

string Simulation::getParticlesName(int partInd) const
{
	return particles_m.at(partInd)->name();
}

string Simulation::getSatelliteName(int satInd) const
{
	return satPartPairs_m.at(satInd)->satellite->name();
}

int Simulation::getParticleIndexOfSat(int satInd) const
{
	return tempSats_m.at(satInd)->particleInd;
}


//Class pointers
Particles* Simulation::particles(int partInd) const
{
	return particles_m.at(partInd).get();
}

Particles* Simulation::particles(string name) const
{
	for (auto& part : particles_m)
		if (part->name() == name)
			return part.get();
	throw invalid_argument("Simulation::particle: no particle of name " + name);
}

Particles* Simulation::particles(Satellite* satellite) const
{
	for (auto& satPart : satPartPairs_m)
		if (satPart->satellite.get() == satellite)
			return satPart->particle.get();
	throw invalid_argument("Simulation::particle: no satellite " + satellite->name());
}

Satellite* Simulation::satellite(int satInd) const
{
	return satPartPairs_m.at(satInd)->satellite.get();
}

Satellite* Simulation::satellite(string name) const
{
	for (auto& satPart : satPartPairs_m)
		if (satPart->satellite->name() == name)
			return satPart->satellite.get();
	throw invalid_argument("Simulation::satellite: no satellite of name " + name);
}

BModel* Simulation::Bmodel(size_t dev) const
{
	return BFieldModel_m.at(dev).get();
}

EField* Simulation::Efield(size_t dev) const
{
	return EFieldModel_m.at(dev).get();
}

Log* Simulation::getLog()
{
	return Log_m.get();
}


//Simulation data
const vector<vector<float>>& Simulation::getParticleData(size_t partInd, bool originalData)
{
	if (partInd > (particles_m.size() - 1))
		throw out_of_range("Simulation::getParticleData: no particle at the specifed index " + to_string(partInd));
	return ((originalData) ? (particles_m.at(partInd)->data(true)) : (particles_m.at(partInd)->data(false)));
}

const vector<vector<float>>& Simulation::getSatelliteData(size_t satInd)
{
	if (satInd > (satPartPairs_m.size() - 1))
		throw out_of_range("Simulation::getSatelliteData: no satellite at the specifed index " + to_string(satInd));
	return satellite(satInd)->data();
}

//Fields data
float Simulation::getBFieldAtS(float s, float time) const
{
	return BFieldModel_m.at(0)->getBFieldAtS(s, time);
}

float Simulation::getEFieldAtS(float s, float time) const
{
	return EFieldModel_m.at(0)->getEFieldAtS(s, time);
}


//
//======== Class Creation Functions ========//
//
void Simulation::createParticlesType(string name, float mass, float charge, size_t numParts, string loadFilesDir, bool save)
{
	for (size_t part = 0; part < particles_m.size(); part++)
		if (name == particles_m.at(part).get()->name())
			throw invalid_argument("Simulation::createParticlesType: particle already exists with the name: " + name);

	Log_m->createEntry("Particles Type Created: " + name + ": Mass: " + to_string(mass) + ", Charge: " + to_string(charge) 
		+ ", Number of Parts: " + to_string(numParts) + ", Files Loaded?: " + ((loadFilesDir != "") ? "True" : "False"));
	
	vector<string> attrNames{ "vpara", "vperp", "s", "t_inc", "t_esc" };

	if (save)
	{
		vector<string> attrLabels;
		for (size_t atr = 0; atr < attrNames.size(); atr++)
			attrLabels.push_back("attrName");
		attrLabels.push_back("loadFilesDir");
		
		vector<string> namesCopy{ attrNames };
		namesCopy.push_back(loadFilesDir);
	}

	size_t devcnt{ utils::GPU::getDeviceCount() };
	vector<size_t> pcntPerGPU{ getSplitSize(numParts) };
	shared_ptr<Particles> newPart{ make_unique<Particles>(name, attrNames, mass, charge, numParts, devcnt, pcntPerGPU) };

	if (loadFilesDir != "")
		newPart->loadDataFromDisk(loadFilesDir);

	newPart->__data(true).at(4) = vector<float>(newPart->getNumberOfParticles(), -1.0f); //sets t_esc to -1.0 - i.e. particles haven't escaped yet
	particles_m.push_back(move(newPart));
}

void Simulation::createTempSat(string partName, float altitude, bool upwardFacing, string name)
{
	for (size_t partInd = 0; partInd < particles_m.size(); partInd++)
	{
		if (particles((int)partInd)->name() == partName)
		{
			createTempSat(partInd, altitude, upwardFacing, name);
			return;
		}
	}
	throw invalid_argument("Simulation::createTempSat: no particle of name " + name);
}

void Simulation::createTempSat(size_t partInd, float altitude, bool upwardFacing, string name)
{//"Temp Sats" are necessary to ensure particles are created before their accompanying satellites
	if (initialized_m)
		throw runtime_error("Simulation::createTempSat: initializeSimulation has already been called, no satellite will be created of name " + name);
	if (partInd >= particles_m.size())
		throw out_of_range("Simulation::createTempSat: no particle at the specifed index " + to_string(partInd));

	tempSats_m.push_back(make_unique<TempSat>(partInd, altitude, upwardFacing, name));
}

void Simulation::createSatellite(TempSat* tmpsat, bool save) //protected
{
	size_t partInd{ tmpsat->particleInd };
	float altitude{ tmpsat->altitude };
	bool upwardFacing{ tmpsat->upwardFacing };
	string name{ tmpsat->name };
	size_t dev = 0;
	// create satelite on all the devices
	// Satelite array indexed as Sat 1 dev 0-n, Sat 2 dev 0-n, ...
	do
	{
		utils::GPU::setDev(dev);

		if (particles_m.size() <= partInd)
			throw out_of_range("createSatellite: no particle at the specifed index " + to_string(partInd));
		if (particles_m.at(partInd)->getCurrDataGPUPtr(dev) == nullptr)
			throw runtime_error("createSatellite: pointer to GPU data is a nullptr of particle " + particles_m.at(partInd)->name() + " - that's just asking for trouble");

		Log_m->createEntry("Created Satellite: " + name + ", Particles tracked: " + particles_m.at(partInd)->name()
			+ ", Altitude: " + to_string(altitude) + ", " + ((upwardFacing) ? "Upward" : "Downward") + " Facing Detector");

		vector<string> attrNames{ "vpara", "vperp", "s", "time", "index" };
		shared_ptr<Particles> part{ particles_m.at(partInd) };
		unique_ptr<Satellite> sat{ make_unique<Satellite>(name, attrNames, altitude, upwardFacing, part->getNumberOfParticles(), part->getCurrDataGPUPtr(dev)) };
		satPartPairs_m.push_back(make_unique<SatandPart>(move(sat), move(part)));

		dev++;
	} while (dev < gpuCount_m);
}

void Simulation::setBFieldModel(string name, vector<float> args, bool save)
{//add log file messages
	if (BFieldModel_m.size() != 0)
		throw invalid_argument("Simulation::setBFieldModel: trying to assign B Field Model when one is already assigned - existing: " + BFieldModel_m.at(0)->name() + ", attempted: " + name);
	if (args.empty())
		throw invalid_argument("Simulation::setBFieldModel: no arguments passed in");
	vector<string> attrNames;
	
	// Multi GPU will run this for each device. If compiled for cpu this will still run a single time.
	size_t dev = 0;
	do /* while (dev < gpuCount_m) */
	{
		utils::GPU::setDev(dev);

		if (name == "DipoleB")
		{
			if (args.size() == 1)
			{ //for defaults in constructor of DipoleB
				BFieldModel_m.push_back( make_unique<DipoleB>(args.at(0)) );
				args.push_back(((DipoleB*)BFieldModel_m.at(dev).get())->getErrTol());
				args.push_back(((DipoleB*)BFieldModel_m.at(dev).get())->getds());
			}
			else if (args.size() == 3)
				BFieldModel_m.push_back( make_unique<DipoleB>(args.at(0), args.at(1), args.at(2)) );
			else
				throw invalid_argument("setBFieldModel: wrong number of arguments specified for DipoleB: " + to_string(args.size()));

			attrNames = { "ILAT", "ds", "errTol" };
		}
		else if (name == "DipoleBLUT")
		{
			if (args.size() == 3)
				BFieldModel_m.push_back( make_unique<DipoleBLUT>(args.at(0), simMin_m, simMax_m, args.at(1), (int)args.at(2)) );
			else
				throw invalid_argument("setBFieldModel: wrong number of arguments specified for DipoleBLUT: " + to_string(args.size()));

			attrNames = { "ILAT", "ds", "numMsmts" };
		}
		else
		{
			cout << "Not sure what model is being referenced.  Using DipoleB instead of " << name << endl;
			BFieldModel_m.push_back( make_unique<DipoleB>(args.at(0)) );
			args.resize(3);
			args.at(1) = ((DipoleB*)BFieldModel_m.at(dev).get())->getErrTol();
			args.at(1) = ((DipoleB*)BFieldModel_m.at(dev).get())->getds();
			attrNames = { "ILAT", "ds", "errTol" };
		}
		dev++;
	} while (dev < gpuCount_m);
}

void Simulation::setBFieldModel(unique_ptr<BModel> BModelptr)
{
	//Assume set for all devices
	BFieldModel_m.clear(); //Empty vector array if already allocated
	size_t dev = 0;
	do
	{
		utils::GPU::setDev(dev);
		BFieldModel_m.push_back(std::move(BModelptr));
		dev++;
	} while (dev < gpuCount_m);
}

void Simulation::addEFieldModel(string name, vector<float> args, bool save)
{
	if (EFieldModel_m.size() == 0)
	{
		size_t dev = 0;
		do
		{
			utils::GPU::setDev(dev);

			EFieldModel_m.push_back( make_unique<EField>() );
			dev++;
		} while (dev < gpuCount_m);
	}
	
	vector<string> attrNames;

	if (name == "QSPS") //need to check to make sure args is formed properly, as well as save to disk
	{
		if (args.size() != 3) throw invalid_argument("Simulation::addEFieldModel: QSPS: Argument vector is improperly formed.  Proper format is: { altMin, altMax, mag }");
		
		vector<meters> altMin;
		vector<meters> altMax;
		vector<Vperm>  mag;

		for (size_t entry = 0; entry < args.size() / 3; entry++)
		{
			altMin.push_back(args.at(3 * entry));
			altMax.push_back(args.at(3 * entry + 1));
			mag.push_back(args.at(3 * entry + 2));
			attrNames.push_back("altMin");
			attrNames.push_back("altMax");
			attrNames.push_back("magnitude");
		}
		cout << "QSPS temporary fix - instantiates with [vector].at(0)\n";
		size_t dev = 0;
		do
		{
			utils::GPU::setDev(dev);
			EFieldModel_m.at(dev)->add(make_unique<QSPS>(altMin.at(0), altMax.at(0), mag.at(0)));
			dev++;
		} while (dev < gpuCount_m);
		Log_m->createEntry("Added QSPS");
	}
	else if (name == "AlfvenLUT")
	{
		cout << "AlfvenLUT not implemented quite yet.  Returning." << endl;
		return;
	}
	/*else if (name == "AlfvenCompute")
	{
		cout << "AlfvenCompute not implemented quite yet.  Returning." << endl;
		return;
	}*/
}


//Other utilities
void Simulation::saveDataToDisk()
{
	if (!initialized_m)
		throw logic_error("Simulation::saveDataToDisk: simulation not initialized with initializeSimulation()");
	if (!saveReady_m)
		throw logic_error("Simulation::saveDataToDisk: simulation not iterated and/or copied to host with iterateSmiulation()");

	LOOP_OVER_1D_ARRAY(getNumberOfParticleTypes(), particles_m.at(iii)->saveDataToDisk(saveRootDir_m + "/bins/particles_init/", true));
	LOOP_OVER_1D_ARRAY(getNumberOfParticleTypes(), particles_m.at(iii)->saveDataToDisk(saveRootDir_m + "/bins/particles_final/", false));
	LOOP_OVER_1D_ARRAY(getNumberOfSatellites(), satellite(iii)->saveDataToDisk(saveRootDir_m + "/bins/satellites/"));

	saveReady_m = false;
}

void Simulation::loadDataFromDisk()
{
	if (initialized_m)
		cerr << "Simulation::loadDataFromDisk: simulation initialized.  Loading data at this point may result in unexpected behavior";

	LOOP_OVER_1D_ARRAY(getNumberOfParticleTypes(), particles_m.at(iii)->loadDataFromDisk(saveRootDir_m + "bins/particles_init/", true));
	LOOP_OVER_1D_ARRAY(getNumberOfParticleTypes(), particles_m.at(iii)->loadDataFromDisk(saveRootDir_m + "bins/particles_final/", false));
	LOOP_OVER_1D_ARRAY(getNumberOfSatellites(), satellite(iii)->loadDataFromDisk(saveRootDir_m + "bins/satellites/"));
}

void Simulation::resetSimulation(bool fields)
{
	while (satPartPairs_m.size() != 0)
		satPartPairs_m.pop_back();
	while (particles_m.size() != 0)
		particles_m.pop_back();

	if (fields)
	{
		size_t dev = 0;
		do
		{
			utils::GPU::setDev(dev);
			BFieldModel_m.at(dev).reset();
			EFieldModel_m.at(dev).reset();
			dev++;
		} while (dev < gpuCount_m);
	}
}

void Simulation::saveSimulation() const
{
	string filename{ saveRootDir_m + string("/simulation.ser") };

	if (std::filesystem::exists(filename))
	{
		cerr << __func__ << ": Warning: filename exists: " << filename << " Returning without saving.";
		return;
	}

	ofstream out(filename, std::ofstream::binary);
	if (!out) throw invalid_argument(__func__ + string(": unable to create file: ") + filename);

	auto writeStrBuf = [&](const stringbuf& sb)
	{
		out.write(sb.str().c_str(), sb.str().length());
	};

	out.write(reinterpret_cast<const char*>(&dt_m), sizeof(float));
	out.write(reinterpret_cast<const char*>(&simMin_m), sizeof(float));
	out.write(reinterpret_cast<const char*>(&simMax_m), sizeof(float));
	writeStrBuf(serializeString(saveRootDir_m));
	out.write(reinterpret_cast<const char*>(&simTime_m), sizeof(float));
	
	//Write BField
	Component comp{ Component::BField };
	out.write(reinterpret_cast<const char*>(&comp), sizeof(Component));
	BModel::Type type{ BFieldModel_m.at(0)->type() };
	out.write(reinterpret_cast<const char*>(&type), sizeof(BModel::Type));
	BFieldModel_m.at(0)->serialize(out);

	//Write EField
	comp = Component::EField;
	out.write(reinterpret_cast<const char*>(&comp), sizeof(Component));
	EFieldModel_m.at(0)->serialize(out);

	//Write Log
	//comp = Component::Log;
	//out.write(reinterpret_cast<const char*>(&comp), sizeof(Component));
	//Log_m->serialize(out); //Log::serialize is to be written

	//Write Particles
	comp = Component::Particles;
	for (const auto& part : particles_m)
	{
		out.write(reinterpret_cast<const char*>(&comp), sizeof(Component));
		part->serialize(out);
	}

	//Write Satellite
	comp = Component::Satellite;
	for (const auto& sat : satPartPairs_m)
	{
		out.write(reinterpret_cast<const char*>(&comp), sizeof(Component));
		for (size_t part = 0; part < particles_m.size(); part++) //write particle index
			if (sat->particle.get() == particles_m.at(part).get())
				out.write(reinterpret_cast<const char*>(&part), sizeof(size_t));
		sat->satellite->serialize(out);
	}

	out.close();
}

void Simulation::loadSimulation(string saveRootDir)
{
	string filename{ saveRootDir + string("/simulation.ser") };
	ifstream in(filename, std::ifstream::binary);
	if (!in) throw invalid_argument(__func__ + string(": unable to open file: ") + filename);
	
	in.read(reinterpret_cast<char*>(&dt_m), sizeof(float));
	in.read(reinterpret_cast<char*>(&simMin_m), sizeof(float));
	in.read(reinterpret_cast<char*>(&simMax_m), sizeof(float));
	/*saveRootDir_m = */deserializeString(in);
	in.read(reinterpret_cast<char*>(&simTime_m), sizeof(float));
	
	while (true)
	{
		Component comp{ 6 };
		in.read(reinterpret_cast<char*>(&comp), sizeof(Component));
		
		if (in.eof()) break;
		
		if (comp == Component::BField)
		{
			BModel::Type type{ BModel::Type::Other };
			in.read(reinterpret_cast<char*>(&type), sizeof(BModel::Type));
			size_t dev = 0;
			do
			{
				utils::GPU::setDev(dev);

				if (type == BModel::Type::DipoleB)
					BFieldModel_m.push_back( make_unique<DipoleB>(in) );
				else if (type == BModel::Type::DipoleBLUT)
					BFieldModel_m.push_back(make_unique<DipoleBLUT>(in));
				else throw std::runtime_error("Simulation::load: BModel type not recognized");
				dev++;
			} while (dev < gpuCount_m);
		}
		else if (comp == Component::EField)
		{
			size_t dev = 0;
			do
			{
				utils::GPU::setDev(dev);
				dev++;
				EFieldModel_m.push_back( make_unique<EField>(in));
			} while (dev < gpuCount_m);
		}
		//else if (comp == Component::Log)
		//{
			//Log::deserialize to be written
			//Log_m = make_unique<Log>(in);
		//}
		else if (comp == Component::Particles)
		{
			particles_m.push_back(make_unique<Particles>(in));
		}
		else if (comp == Component::Satellite)
		{
			size_t part{ readSizetLength(in) };
			shared_ptr<Particles> particles{ particles_m.at(part) };

			satPartPairs_m.push_back(
				make_unique<SatandPart>(
					make_unique<Satellite>(in, particles_m.at(part)->getCurrDataGPUPtr()),
					move(particles)));
		}
		else throw std::runtime_error("Simulation::load: Simulation Component not recognized");
	}
}