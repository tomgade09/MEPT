#include <fstream>

//CUDA includes
#include "ErrorHandling/simExceptionMacros.h"

#include "Particles/Particles.h"
#include "utils/fileIO.h"
#include "utils/arrayUtilsGPU.h"
#include "utils/serializationHelpers.h"

using std::cout;
using std::cerr;
using std::endl;
using std::ofstream;
using std::ifstream;
using std::to_string;
using std::logic_error;
using std::runtime_error;
using std::invalid_argument;

using utils::fileIO::readDblBin;
using utils::fileIO::writeDblBin;
using namespace utils::fileIO::serialize;

Particles::Particles(string name, vector<string> attributeNames, double mass, double charge, size_t numParts) :
	name_m{ name }, attributeNames_m{ attributeNames }, mass_m{ mass }, charge_m{ charge }, numberOfParticles_m{ numParts }
{
	origData_m = vector<vector<double>>(attributeNames.size(), vector<double>(numParts));
	currData_m = vector<vector<double>>(attributeNames.size(), vector<double>(numParts));

	//multi-GPU: probably do like this: make this class aware of Environment, initialize on all GPUs where use_m = true with a loop or something
	initializeGPU();
}

Particles::Particles(ifstream& in)
{ //for loading serialzed class
	deserialize(in);

	origData_m = vector<vector<double>>(attributeNames_m.size(), vector<double>(numberOfParticles_m));
	currData_m = vector<vector<double>>(attributeNames_m.size(), vector<double>(numberOfParticles_m));
	initializeGPU();
}

Particles::~Particles()
{
	freeGPUMemory(); //don't forget to free on multiple GPUs - probably do same as initializing
}

// ================ Particles - protected ================ //
void Particles::initializeGPU()
{
	utils::GPU::setup2DArray(&currData1D_d, &currData2D_d, attributeNames_m.size() + 2, numberOfParticles_m);
	initializedGPU_m = true;
}

void Particles::copyDataToGPU(bool origToGPU)
{
	if (!initializedGPU_m)
		throw logic_error("Particles::copyDataToGPU: GPU memory has not been initialized yet for particle " + name_m);
	if (!initDataLoaded_m)
		throw logic_error("Particles::copyDataToGPU: data not loaded from disk with Particles::loadDataFromDisk or generated with Particles::generateRandomParticles " + name_m);

	if (origToGPU) utils::GPU::copy2DArray(origData_m, &currData1D_d, true); //sending to orig(host) -> data(GPU)
	else           utils::GPU::copy2DArray(currData_m, &currData1D_d, true); //sending to curr(host) -> data(GPU)
}

void Particles::copyDataToHost()
{
	if (!initializedGPU_m)
		throw logic_error("Particles::copyDataToHost: GPU memory has not been initialized yet for particle " + name_m);

	utils::GPU::copy2DArray(currData_m, &currData1D_d, false); //coming back data(GPU) -> curr(host)
}

void Particles::freeGPUMemory()
{
	if (!initializedGPU_m) return;

	utils::GPU::free2DArray(&currData1D_d, &currData2D_d);
	
	initializedGPU_m = false;
}

//need a custom solution for this...
//file read/write exception checking (probably should mostly wrap fileIO functions)
#define FILE_RDWR_EXCEP_CHECK(x) \
	try{ x; } \
	catch(const invalid_argument& a) { cerr << __FILE__ << ":" << __LINE__ << " : " << "Invalid argument error: " << a.what() << ": continuing without loading file" << endl; cout << "FileIO exception: check log file for details" << endl; } \
	catch(...)                            { throw; }


// ================ Particles - public ================ //

vector<vector<double>>& Particles::__data(bool orig)
{ //returns a non-const version of data
	return ((orig) ? origData_m : currData_m);
}

const vector<vector<double>>& Particles::data(bool orig) const
{ //returns a const version of data
	return ((orig) ? origData_m : currData_m);
}

const vector<string>& Particles::attributeNames() const
{
	return attributeNames_m;
}

string Particles::name() const
{
	return name_m;
}

double Particles::mass() const
{
	return mass_m;
}

double Particles::charge() const
{
	return charge_m;
}

size_t Particles::getNumberOfParticles() const
{
	return numberOfParticles_m;
}

size_t Particles::getNumberOfAttributes() const
{
	return attributeNames_m.size();
}

bool Particles::getInitDataLoaded() const
{
	return initDataLoaded_m;
}

double** Particles::getCurrDataGPUPtr() const
{
	return currData2D_d;
}

size_t Particles::getAttrIndByName(string searchName) const
{
	for (size_t name = 0; name < attributeNames_m.size(); name++)
		if (searchName == attributeNames_m.at(name))
			return name;

	throw invalid_argument("Particles::getDimensionIndByName: specified name is not present in name array: " + searchName);
}

string Particles::getAttrNameByInd(size_t searchIndx) const
{
	if (!(searchIndx <= (attributeNames_m.size() - 1) && (searchIndx >= 0)))
		throw invalid_argument("Particles::getDimensionNameByInd: specified index is invalid: " + to_string(searchIndx));

	return attributeNames_m.at(searchIndx);
}

void Particles::setParticlesSource_s(double s_ion, double s_mag)
{
	cout << "Particles::setParticlesSource_s: This function is being depreciated and will be removed in a future release.\n";
	//need to create a ParticleDistribution and load into Particles::origData_m before I get rid of this
	size_t s_ind{ getAttrIndByName("s") };
	size_t v_ind{ getAttrIndByName("vpara") };

	for (size_t ind = 0; ind < origData_m.at(s_ind).size(); ind++)
	{
		if (origData_m.at(v_ind).at(ind) > 0.0)
			origData_m.at(s_ind).at(ind) = s_ion;
		else if (origData_m.at(v_ind).at(ind) < 0.0)
			origData_m.at(s_ind).at(ind) = s_mag;
		else
			throw std::logic_error("Particles::setParticlesSource_s: vpara value is exactly 0.0 - load data to the origData_m array first.  Aborting.  index: " + std::to_string(ind));
	}

	copyDataToGPU();
}

void Particles::generateDist(size_t numEbins, eV E_min, eV E_max, size_t numPAbins, degrees PA_min, degrees PA_max, meters s_ion, meters s_mag)
{
	std::clog << "Particles::generateDist(numEbins: " << to_string(numEbins) << ", E_min: " << E_min << ", E_max: "
		<< E_max << ", numPAbins: " << numPAbins << ", PA_min: " << PA_min << ", PA_max: " << PA_max << ", s_ion: "
		<< s_ion << ", s_mag: " << s_mag << ")";

	ParticleDistribution dist{ "./", attributeNames_m, name_m, mass_m, { 0.0, 0.0, 0.0, 0.0, -1.0 }, false };
	dist.addEnergyRange(numEbins, E_min, E_max);   //defaults to log spaced bins, mid bin
	dist.addPitchRange(numPAbins, PA_min, PA_max); //defaults to linear spaced bins, mid bin
	origData_m = dist.generate(s_ion, s_mag);

	initDataLoaded_m = true;
	copyDataToGPU();
}

void Particles::loadDistFromPD(const ParticleDistribution& pd, meters s_ion, meters s_mag)
{
	//std::cout << "Particles::loadDistFromPD(ParticleDistribution, s_ion: " << s_ion << ", s_mag: " << s_mag << ")\n";

	origData_m = pd.generate(s_ion, s_mag);

	initDataLoaded_m = true;
	copyDataToGPU();
}

void Particles::loadDistFromPD(const ParticleDistribution& pd, vector<meters>& s)
{
	//std::cout << "Particles::loadDistFromPD(ParticleDistribution, s vector)\n";

	origData_m = pd.generate(s);

	initDataLoaded_m = true;
	copyDataToGPU();
}

void Particles::loadDistFromDisk(string folder, string distName, meters s_ion, meters s_mag)
{
	utils::fileIO::ParticleDistribution dist{ folder, distName };
	
	if (attributeNames_m != dist.attrNames())
		throw logic_error("Particles::loadDistFromDisk: Particles attribute \
			names do not match ParticleDistribution attribute names");
	
	origData_m = dist.generate(s_ion, s_mag);
	
	initDataLoaded_m = true;
	copyDataToGPU();
}

void Particles::loadDataFromMem(vector<vector<double>> data, bool orig) //orig defaults to true
{
	((orig) ? origData_m = data : currData_m = data);
	numberOfParticles_m = ((orig) ? (int)origData_m.at(0).size() : (int)currData_m.at(0).size());

	if (orig) initDataLoaded_m = true; //copyDataToGPU uses this flag to ensure data is present in origData_m
	if (orig) copyDataToGPU();
}


void Particles::loadDataFromDisk(string folder, bool orig) //orig defaults to true
{
	for (size_t attrs = 0; attrs < attributeNames_m.size(); attrs++)
		FILE_RDWR_EXCEP_CHECK(readDblBin((orig ? origData_m.at(attrs) : currData_m.at(attrs)), folder + "/" + name_m + "_" + attributeNames_m.at(attrs) + ".bin", numberOfParticles_m));

	if (orig) initDataLoaded_m = true; //copyDataToGPU uses this flag to ensure data is present in origData_m
	if (orig) copyDataToGPU();
}

void Particles::saveDataToDisk(string folder, bool orig) const
{
	for (size_t attrs = 0; attrs < attributeNames_m.size(); attrs++)
		FILE_RDWR_EXCEP_CHECK(writeDblBin((orig ? origData_m.at(attrs) : currData_m.at(attrs)), folder + "/" + name_m + "_" + attributeNames_m.at(attrs) + ".bin", numberOfParticles_m));
}

void Particles::serialize(ofstream& out) const
{ //saves necessary attributes about the particle to disk
	auto writeStrBuf = [&](const stringbuf& sb)
	{
		out.write(sb.str().c_str(), sb.str().length());
	};

	// ======== write data to file ======== //
	writeStrBuf(serializeString(name_m));
	writeStrBuf(serializeStringVector(attributeNames_m));
	
	out.write(reinterpret_cast<const char*>(&initDataLoaded_m), sizeof(bool));
	out.write(reinterpret_cast<const char*>(&initializedGPU_m), sizeof(bool));
	out.write(reinterpret_cast<const char*>(&numberOfParticles_m), sizeof(size_t));
	out.write(reinterpret_cast<const char*>(&mass_m), sizeof(double));
	out.write(reinterpret_cast<const char*>(&charge_m), sizeof(double));
}

void Particles::deserialize(ifstream& in) //protected function
{ //recreates a saved "serialization"
	name_m = deserializeString(in);
	attributeNames_m = deserializeStringVector(in);

	in.read(reinterpret_cast<char*>(&initDataLoaded_m), sizeof(bool));
	in.read(reinterpret_cast<char*>(&initializedGPU_m), sizeof(bool));
	in.read(reinterpret_cast<char*>(&numberOfParticles_m), sizeof(size_t));
	in.read(reinterpret_cast<char*>(&mass_m), sizeof(double));
	in.read(reinterpret_cast<char*>(&charge_m), sizeof(double));
}