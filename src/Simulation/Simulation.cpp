#include "Simulation/Simulation.h"

#include <iomanip>
#include <filesystem>

#include "utils/loopmacros.h"
#include "utils/arrayUtilsGPU.h"
#include "utils/serializationHelpers.h"
#include "ErrorHandling/simExceptionMacros.h"

using std::cout;
using std::clog;
using std::cerr;
using std::endl;
using std::move;
using std::string;
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

Simulation::Simulation(seconds dt, meters simMin, meters simMax, string rootdir) :
	dt_m{ dt }, simMin_m{ simMin }, simMax_m{ simMax }
{
	//generate a folder for output data in {ROOT}/_dataout
	stringstream dataout;
	auto in_time_t = system_clock::to_time_t(system_clock::now());

	if (rootdir.back() != '/' || rootdir.back() != '\\')
		rootdir.push_back('/');

	dataout << rootdir << "_dataout/" << put_time(localtime(&in_time_t), "%y%m%d_%H.%M.%S/");

	path datadir{ dataout.str() };
	if (!exists(rootdir + "_dataout/"))
	{
		try
		{
			create_directory(rootdir + "_dataout/");
		}
		catch(std::exception& e)
		{
			cout << "Exception: " << e.what() << "\n";
			exit(1);
		}
	}
	
	if (!exists(datadir))
	{
		try
		{
			create_directory(datadir);

			create_directory(path(dataout.str() + "PADIC"));
			create_directory(path(dataout.str() + "bins"));
			create_directory(path(dataout.str() + "bins/particles_final"));
			create_directory(path(dataout.str() + "bins/particles_init"));
			create_directory(path(dataout.str() + "bins/satellites"));
		}
		catch (std::exception& e)
		{
			cout << "Exception: " << e.what() << "\n";
			exit(1);
		}
	}

	saveRootDir_m = dataout.str();
	Log_m = make_unique<Log>(saveRootDir_m + "simulation.log");
	Log_m->createEntry(string("Simulation FP precision: ") + ((sizeof(flPt_t) == 8) ? "double precision floating point" :
		                                                                              "single precision floating point"));
	
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
	Log_m->createEntry("End simulation");
}


void Simulation::printSimAttributes(size_t numberOfIterations, size_t itersBtwCouts) //protected
{
	stringstream out;

	//Sim Header (folder) printed from Python - move here eventually
	out << "===============================================================" << "\n";
	out << "Data Folder:    " << saveRootDir_m << "\n";
	out << "GPU(s) In Use:  " << utils::GPU::getDeviceNames() << endl;
	out << "Sim between:    " << simMin_m << "m - " << simMax_m << "m" << endl;
	out << "dt:             " << dt_m << "s" << endl;
	out << "FlPt Precision: " << ((sizeof(flPt_t) == 4) ? "single precision floating point (float)" :
	                                                      "double precision floating point (double)") << endl;
	out << "BModel Model:   " << BFieldModel_m->name() << endl;
	out << "EField Elems:   " << ((Efield()->qspsCount() > 0) ? "QSPS" : "") << endl;
	out << "Particles:      ";
	for (size_t iii = 0; iii < particles_m.size(); iii++)
	{
		out << ((iii != 0) ? "                " : "") << particles_m.at(iii)->name()
		    << ": #: " << particles_m.at(iii)->getNumberOfParticles() << ", loaded files?: "
			<< (particles_m.at(iii)->getInitDataLoaded() ? "true" : "false") << std::endl;
	}
	out << "Satellites:     ";
	for (int iii = 0; iii < getNumberOfSatellites(); iii++)
	{
		out << ((iii != 0) ? "                " : "") << satellites_m.at(iii)->name()
			<< ": alt: " << satellites_m.at(iii)->altitude() << " m, upward?: "
			<< (satellites_m.at(iii)->upward() ? "true" : "false") << std::endl;
	}
	out << "Iterations:     " << numberOfIterations << endl;
	out << "Iters Btw Cout: " << itersBtwCouts << endl;
	out << "Time to setup:  " << Log_m->timeElapsedTotal_s() << " s" << endl;
	out << "===============================================================" << endl;

	cout << out.str(); //print this output
	clog << out.str(); //log this output
}

void Simulation::incTime()
{
	simTime_m += dt_m;
}


//
//======== Simulation Access Functions ========//
//
seconds Simulation::simtime() const
{
	return simTime_m;
}

seconds Simulation::dt() const
{
	return dt_m;
}

meters Simulation::simMin() const
{
	return simMin_m;
}

meters Simulation::simMax() const
{
	return simMax_m;
}


//Class data
int Simulation::getNumberOfParticleTypes() const
{
	return static_cast<int>(particles_m.size());
}

int Simulation::getNumberOfSatellites() const
{
	return static_cast<int>(satellites_m.size());
}

int Simulation::getNumberOfParticles(int partInd) const
{
	return static_cast<int>(particles_m.at(partInd)->getNumberOfParticles());
}

int Simulation::getNumberOfAttributes(int partInd) const
{
	return static_cast<int>(particles_m.at(partInd)->getNumberOfAttributes());
}

string Simulation::getParticlesName(int partInd) const
{
	return particles_m.at(partInd)->name();
}

string Simulation::getSatelliteName(int satInd) const
{
	return satellites_m.at(satInd)->name();
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

Satellite* Simulation::satellite(int satInd) const
{
	return satellites_m.at(satInd).get();
}

Satellite* Simulation::satellite(string name) const
{
	for (auto& sat : satellites_m)
		if (sat->name() == name)
			return sat.get();
	throw invalid_argument("Simulation::satellite: no satellite of name " + name);
}

BModel* Simulation::Bmodel() const
{
	return BFieldModel_m.get();
}

EField* Simulation::Efield() const
{
	return EFieldModel_m.get();
}

Log* Simulation::getLog()
{
	return Log_m.get();
}


//Simulation data
const fp2Dvec& Simulation::getParticleData(size_t partInd, bool originalData)
{
	if (partInd > (particles_m.size() - 1))
		throw out_of_range("Simulation::getParticleData: no particle at the specifed index " + to_string(partInd));
	return ((originalData) ? (particles_m.at(partInd)->data(true)) : (particles_m.at(partInd)->data(false)));
}

const SatDataVecs& Simulation::getSatelliteData(size_t satInd)
{
	if (satInd > (satellites_m.size() - 1))
		throw out_of_range("Simulation::getSatelliteData: no satellite at the specifed index " + to_string(satInd));
	return satellite(satInd)->data();
}

//Fields data
tesla Simulation::getBFieldAtS(meters s, seconds time) const
{
	return BFieldModel_m->getBFieldAtS(s, time);
}

Vperm Simulation::getEFieldAtS(meters s, seconds time) const
{
	return EFieldModel_m->getEFieldAtS(s, time);
}


//
//======== Class Creation Functions ========//
//
void Simulation::createParticlesType(string name, kg mass, coulomb charge, size_t numParts, string loadFilesDir)
{
	if (particles_m.size() >= 1)
		throw logic_error("Simulation::createParticlesType: one particle type already exists.  " +
			              string("Capability to handle >1 particle type not implemented."));
	//here, the Satellite class needs to be modified to accommodate _each_ particle type created.  So a GPU mem region
	//would need to be created for each particle type for each satellite.  Also, Satellite ctor signature needs to be
	//modified to accommodate the caller passing in how many particle types total will be created.  There may also be
	//changes needed to the CUDA particle E/B Lorentz force particle pusher, although that might be ok.
	//The workaround for now is to run one simulation for one particle type, then another for another particle type, etc
	//and to consolidate the output data at the end.  Of course, this assumes non-interacting particles, which we have
	//assumed since the beginning anyway.

	for (size_t part = 0; part < particles_m.size(); part++)
		if (name == particles_m.at(part).get()->name())
			throw invalid_argument("Simulation::createParticlesType: particle already exists with the name: " + name);

	Log_m->createEntry("Particles Type Created: " + name + ": Mass: " + to_string(mass) + ", Charge: " + to_string(charge) 
		+ ", Number of Parts: " + to_string(numParts) + ", Files Loaded?: " + ((loadFilesDir != "") ? "True" : "False"));
	
	strvec attrNames{ "vpara", "vperp", "s", "t_inc", "t_esc" };

	particles_m.push_back( make_unique<Particles>( name, attrNames, mass, charge, numParts ) );

	particles_m.back()->__data(true).at(4) = fp1Dvec(numParts, -1.0f); //sets t_esc to -1.0 - i.e. particles haven't escaped yet
	if (loadFilesDir != "")
		particles_m.back()->loadDataFromDisk(loadFilesDir);
}

void Simulation::createSatellite(meters altitude, bool upwardFacing, string name, size_t totalParticleCount)
{
	if (altitude < simMin_m)
		throw invalid_argument("Simulation::createSatellite: satellite altitude is lower than simulation min altitude.  " +
		                       string("Satellite won't detect any particles."));
	if (altitude > simMax_m)
		throw invalid_argument("Simulation::createSatellite: satellite altitude is higher than simulation max altitude.  " +
			string("Satellite won't detect any particles."));

	Log_m->createEntry("Created Satellite: " + name + ", Altitude: " + to_string(altitude) + ", " +
		               ((upwardFacing) ? "Upward" : "Downward") + " Facing Detector");

	satellites_m.push_back(make_unique<Satellite>(name, altitude, upwardFacing, totalParticleCount));
}

void Simulation::setBFieldModel(string name, fp1Dvec args)
{//add log file messages
	if (BFieldModel_m != nullptr)
		throw invalid_argument("Simulation::setBFieldModel: trying to assign B Field Model when one is already assigned - existing: " + BFieldModel_m->name() + ", attempted: " + name);
	vector<string> attrNames;
	
	if (name == "DipoleB")
	{
		if (args.size() == 1)  //for defaults in constructor of DipoleB
			BFieldModel_m = make_unique<DipoleB>(args.at(0));
		else if (args.size() == 3)
			BFieldModel_m = make_unique<DipoleB>(args.at(0), args.at(1), args.at(2));
		else
			throw invalid_argument("setBFieldModel: wrong number of arguments specified for DipoleB: " + to_string(args.size()));
	}
	else if (name == "DipoleBLUT")
	{
		if (args.size() == 3)
			BFieldModel_m = make_unique<DipoleBLUT>(args.at(0), simMin_m, simMax_m, args.at(1), (int)args.at(2));
		else
			throw invalid_argument("setBFieldModel: wrong number of arguments specified for DipoleBLUT: " + to_string(args.size()));
	}
	else
	{
		cout << "Simulation::setBFieldModel: Invalid model name.  Using DipoleB instead of " << name << endl;
		BFieldModel_m = make_unique<DipoleB>(args.at(0));
	}
}

/*void Simulation::setBFieldModel(unique_ptr<BModel> BModelptr)
{
	//Assume set for all devices
	BFieldModel_m.clear(); //Empty vector array if already allocated
	size_t dev = 0;
	do
	{
		utils::GPU::setDev(dev);
		BFieldModel_m.push_back(std::move(BModelptr));  //doesn't work - std::move will only work on an object the first time
		dev++;
	} while (dev < gpuCount_m);
}*/

void Simulation::addEFieldModel(string name, fp1Dvec args)
{
	if (EFieldModel_m == nullptr)
		EFieldModel_m = make_unique<EField>();

	if (name == "QSPS") //need to check to make sure args is formed properly, as well as save to disk
	{
		if (args.size() != 3)
			throw invalid_argument("Simulation::addEFieldModel: QSPS: Argument vector is improperly formed.  Proper format is: { altMin, altMax, mag }");
		
		vector<meters> altMin;
		vector<meters> altMax;
		vector<Vperm>  mag;

		for (size_t entry = 0; entry < args.size() / 3; entry++)
		{
			altMin.push_back(args.at(3 * entry));
			altMax.push_back(args.at(3 * entry + 1));
			mag.push_back(args.at(3 * entry + 2));
		}
		std::clog << "Simulation::addEFieldModel: QSPS temporary fix - instantiates with [vector].at(0)\n";
		EFieldModel_m->add(make_unique<QSPS>(altMin.at(0), altMax.at(0), mag.at(0)));
		Log_m->createEntry("Added QSPS");
	}
	else if (name == "AlfvenLUT")
		throw invalid_argument("AlfvenLUT not implemented quite yet");
	else if (name == "AlfvenCompute")
		throw invalid_argument("AlfvenCompute not implemented quite yet");
	else
		throw invalid_argument("Simulation::addEFieldModel: invalid E Model name specified: " + name);
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
		cerr << "Simulation::loadDataFromDisk: simulation initialized.  Loading data at this point may result in unexpected behavior\n";

	LOOP_OVER_1D_ARRAY(getNumberOfParticleTypes(), particles_m.at(iii)->loadDataFromDisk(saveRootDir_m + "bins/particles_init/", true));
	LOOP_OVER_1D_ARRAY(getNumberOfParticleTypes(), particles_m.at(iii)->loadDataFromDisk(saveRootDir_m + "bins/particles_final/", false));
	LOOP_OVER_1D_ARRAY(getNumberOfSatellites(), satellite(iii)->loadDataFromDisk(saveRootDir_m + "bins/satellites/"));
}

void Simulation::resetSimulation(bool fields)
{
	while (satellites_m.size() != 0)
		satellites_m.pop_back();
	while (particles_m.size() != 0)
		particles_m.pop_back();

	if (fields)
	{
		BFieldModel_m.reset();
		EFieldModel_m.reset();
	}
}

void Simulation::saveSimulation() const
{
	string filename{ saveRootDir_m + string("/simulation.ser") };

	if (std::filesystem::exists(filename))
	{
		cerr << __func__ << ": Warning: filename exists: " << filename << " Returning without saving.\n";
		return;
	}

	ofstream out(filename, std::ofstream::binary);
	if (!out) throw invalid_argument(__func__ + string(": unable to create file: ") + filename);

	auto writeStrBuf = [&](const stringbuf& sb)
	{
		out.write(sb.str().c_str(), sb.str().length());
	};

	auto writeFPBuf = [&](const flPt_t fp)
	{   //casts to double so that SP MEPT doesn't break when loading a previously saved DP save file (and vice versa)
		double tmp{ static_cast<double>(fp) };  //cast to double precision FP
		out.write(reinterpret_cast<const char*>(&tmp), sizeof(double));
	};

	auto writeComponentBuf = [&](Component comp)
	{
		out.write(reinterpret_cast<const char*>(&comp), sizeof(Component));
	};

	writeFPBuf(dt_m);
	writeFPBuf(simMin_m);
	writeFPBuf(simMax_m);
	writeFPBuf(simTime_m);
	
	//Write BField
	writeComponentBuf(Component::BField);
	BModel::Type type{ BFieldModel_m->type() };
	out.write(reinterpret_cast<const char*>(&type), sizeof(BModel::Type));
	BFieldModel_m->serialize(out);

	//Write EField
	writeComponentBuf(Component::EField);
	EFieldModel_m->serialize(out);

	//Write Log
	//writeComponentBuf(Component::Log);
	//Log_m->serialize(out); //Log::serialize is to be written

	//Write Particles
	for (const auto& part : particles_m)
	{
		writeComponentBuf(Component::Particles);
		part->serialize(out);
	}

	//Write Satellite
	for (const auto& sat : satellites_m)
	{
		writeComponentBuf(Component::Satellite);
		sat->serialize(out);
	}

	out.close();
}

void Simulation::loadSimulation(string saveRootDir)
{
	string filename{ saveRootDir + string("/simulation.ser") };
	ifstream in(filename, std::ifstream::binary);
	if (!in) throw invalid_argument(__func__ + string(": unable to open file: ") + filename);
	
	auto readFPBuf = [&]()
	{   //casts to double so that SP MEPT doesn't break when loading a previously saved DP save file (and vice versa)
		double tmp{ 0.0 };  //read in double precision FP
		in.read(reinterpret_cast<char*>(&tmp), sizeof(double));
		return tmp;
	};

	dt_m = static_cast<flPt_t>(readFPBuf());
	simMin_m = static_cast<flPt_t>(readFPBuf());
	simMax_m = static_cast<flPt_t>(readFPBuf());
	simTime_m = static_cast<flPt_t>(readFPBuf());
	
	while (true)
	{
		Component comp{ 6 };
		in.read(reinterpret_cast<char*>(&comp), sizeof(Component));
		
		if (in.eof()) break;
		
		if (comp == Component::BField)
		{
			BModel::Type type{ BModel::Type::Other };
			in.read(reinterpret_cast<char*>(&type), sizeof(BModel::Type));

			if (type == BModel::Type::DipoleB)
				BFieldModel_m = make_unique<DipoleB>(in);
			else if (type == BModel::Type::DipoleBLUT)
				BFieldModel_m = make_unique<DipoleBLUT>(in);
			else
				throw std::runtime_error("Simulation::load: BModel type not recognized");
		}
		else if (comp == Component::EField)
			EFieldModel_m = make_unique<EField>(in);
		//else if (comp == Component::Log)
		//{
			//Log::deserialize to be written
			//Log_m = make_unique<Log>(in);
		//}
		else if (comp == Component::Particles)
			particles_m.push_back(make_unique<Particles>(in));
		else if (comp == Component::Satellite)
			satellites_m.push_back(make_unique<Satellite>(in));
		else
			throw std::runtime_error("Simulation::load: Simulation Component not recognized");
	}
}