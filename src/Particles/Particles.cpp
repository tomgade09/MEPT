#include <fstream>

//CUDA includes
#include "ErrorHandling/simExceptionMacros.h"

#include "Particles/Particles.h"
#include "utils/fileIO.h"
#include "utils/arrayUtilsGPU.h"
#include "utils/serializationHelpers.h"

using std::cout;
using std::clog;
using std::cerr;
using std::endl;
using std::ofstream;
using std::ifstream;
using std::to_string;
using std::logic_error;
using std::runtime_error;
using std::invalid_argument;

using utils::fileIO::readBin;
using utils::fileIO::writeBin;
using namespace utils::fileIO::serialize;


Particles::Particles(string name, const strvec& attributeNames, kg mass, coulomb charge, size_t numParts) :
	name_m{ name }, attributeNames_m{ attributeNames }, mass_m{ mass },	charge_m{ charge },	numberOfParticles_m{ numParts }
{
	origData_m = fp2Dvec(attributeNames.size(), fp1Dvec(numParts));
	currData_m = fp2Dvec(attributeNames.size(), fp1Dvec(numParts));
	initializeGPU();
}

Particles::Particles(ifstream& in)
{ //for loading serialzed class
	deserialize(in);

	loadDataFromDisk("./bins/particles_init", true);  //load data from disk in the standard folder
	loadDataFromDisk("./bins/particles_final", false);

	origData_m = fp2Dvec(attributeNames_m.size(), fp1Dvec(numberOfParticles_m));
	currData_m = fp2Dvec(attributeNames_m.size(), fp1Dvec(numberOfParticles_m));
	initializeGPU();
	copyDataToGPU(true);
}

Particles::~Particles()
{
	freeGPUMemory(); //don't forget to free on multiple GPUs - probably do same as initializing
}

// ================ Particles - protected ================ //
void Particles::initializeGPU()
{
	numGPUs_m = utils::GPU::getDeviceCount();
	particleCountPerGPU_m = utils::GPU::getSplitSize(numberOfParticles_m, BLOCKSIZE);

	for (size_t i = 0; i < numGPUs_m; i++)
	{
		currData1D_d.push_back(nullptr);
		currData2D_d.push_back(nullptr);

		utils::GPU::setup2DArray(&currData1D_d.at(i), &currData2D_d.at(i), attributeNames_m.size() + 2,
			particleCountPerGPU_m.at(i), i);

		clog << "Particles::initializeGPU() : setup2DArray GPU num " + to_string(i) +
			    ": num particles " + to_string(particleCountPerGPU_m.at(i));
	}
	initializedGPU_m = true;
}

void Particles::copyDataToGPU(bool currToGPU)
{
	if (!initializedGPU_m)
		throw logic_error("Particles::copyDataToGPU: GPU memory has not been initialized yet for particle " + name_m);
	if (!initDataLoaded_m)
		throw logic_error("Particles::copyDataToGPU: data not loaded from disk with Particles::loadDataFromDisk or generated with Particles::generateRandomParticles " + name_m);

	const fp2Dvec& orig{ origData_m };
	const fp2Dvec& curr{ currData_m };
	
	auto getsubvec = [](const fp2Dvec& vec, size_t offset, size_t end)
	{
		fp2Dvec subvec(vec.size());
		for (size_t attr = 0; attr < vec.size(); attr++)
			subvec.at(attr) =
				fp1Dvec(vec.at(attr).begin() + offset, vec.at(attr).begin() + end);
		return subvec;
	};
	
	size_t offset{ 0 };
	for (size_t dev = 0; dev < numGPUs_m; dev++)
	{
		size_t end{ offset + particleCountPerGPU_m.at(dev) };
		if (end > numberOfParticles_m) end = numberOfParticles_m;

		fp2Dvec subvec;
		
		subvec = getsubvec(orig, offset, end);
		utils::GPU::copy2DArray(subvec, &currData1D_d.at(dev), true, dev); //sending to orig(host) -> data(GPU)
		
		if (currToGPU)
		{
			subvec = getsubvec(curr, offset, end);
			utils::GPU::copy2DArray(subvec, &currData1D_d.at(dev), true, dev); //sending to curr(host) -> data(GPU)
		}

		clog << "Particles::copyDataToGPU() : copy2DArray GPU num " + to_string(dev) +
			": start " + to_string(offset) + ": end " + to_string(end) + ": length " + to_string(subvec.at(0).size());
		
		offset = end;
	}
}

void Particles::copyDataToHost()
{
	if (!initializedGPU_m)
		throw logic_error("Particles::copyDataToHost: GPU memory has not been initialized yet for particle " + name_m);

	size_t offset{ 0 };
	for (size_t dev = 0; dev < numGPUs_m; dev++)
	{
		size_t end{ offset + particleCountPerGPU_m.at(dev) };
		if (end > numberOfParticles_m) end = numberOfParticles_m;
		
		fp2Dvec subvec(currData_m.size(), fp1Dvec(end - offset));

		utils::GPU::copy2DArray(subvec, &currData1D_d.at(dev), false, dev); //coming back data(GPU) -> curr(host)
		for (size_t i = 0; i < subvec.size(); i++)
			std::copy(subvec.at(i).begin(), subvec.at(i).end(), currData_m.at(i).begin() + offset);  //add data back at the appropriate location

		clog << "Particles::copyDataToHost() : copy2DArray GPU num " + to_string(dev) +
			": start " + to_string(offset) + ": end " + to_string(end) + ": length " + to_string(subvec.at(0).size());
		
		offset = end;
	}
}

void Particles::freeGPUMemory()
{
	if (!initializedGPU_m) return;

	for (size_t dev = 0; dev < numGPUs_m; dev++)
		utils::GPU::free2DArray(&currData1D_d.at(dev), &currData2D_d.at(dev), dev);
	
	initializedGPU_m = false;
}

//need a custom solution for this...
//file read/write exception checking (probably should mostly wrap fileIO functions)
#define FILE_RDWR_EXCEP_CHECK(x) \
	try{ x; } \
	catch(const invalid_argument& a) { cerr << __FILE__ << ":" << __LINE__ << " : " << "Invalid argument error: " << a.what() << ": continuing without loading file" << endl; cout << "FileIO exception: check log file for details" << endl; } \
	catch(...)                            { throw; }


// ================ Particles - public ================ //

fp2Dvec& Particles::__data(bool orig)
{ //returns a non-const version of data
	return ((orig) ? origData_m : currData_m);
}

const fp2Dvec& Particles::data(bool orig) const
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

kg Particles::mass() const
{
	return mass_m;
}

coulomb Particles::charge() const
{
	return charge_m;
}

size_t Particles::getNumberOfParticles() const
{
	return numberOfParticles_m;
}

size_t Particles::getNumParticlesPerGPU(size_t GPUind) const
{
	return particleCountPerGPU_m.at(GPUind);
}

vector<size_t> Particles::getNumParticlesPerGPU( ) const
{
	return particleCountPerGPU_m;
}

size_t Particles::getNumberOfAttributes() const
{
	return attributeNames_m.size();
}

bool Particles::getInitDataLoaded() const
{
	return initDataLoaded_m;
}

flPt_t** Particles::getCurrDataGPUPtr(size_t GPUind) const
{
	return currData2D_d.at(GPUind);
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

void Particles::setParticlesSource_s(meters s_ion, meters s_mag)
{
	cout << "Particles::setParticlesSource_s: This function is being depreciated and will be removed in a future release.\n";
	//need to create a ParticleDistribution and load into Particles::origData_m before I get rid of this
	size_t s_ind{ getAttrIndByName("s") };
	size_t v_ind{ getAttrIndByName("vpara") };

	for (size_t ind = 0; ind < origData_m.at(s_ind).size(); ind++)
	{
		if (origData_m.at(v_ind).at(ind) > 0.0f)
			origData_m.at(s_ind).at(ind) = s_ion;
		else if (origData_m.at(v_ind).at(ind) < 0.0f)
			origData_m.at(s_ind).at(ind) = s_mag;
		else
			throw std::logic_error("Particles::setParticlesSource_s: vpara value is exactly 0.0f - load data to the origData_m array first.  Aborting.  index: " + std::to_string(ind));
	}

	copyDataToGPU();
}

void Particles::generateDist(size_t numEbins, eV E_min, eV E_max, size_t numPAbins, degrees PA_min, degrees PA_max, meters s_ion, meters s_mag)
{
	std::clog << "Particles::generateDist(numEbins: " << to_string(numEbins) << ", E_min: " << E_min << ", E_max: "
		<< E_max << ", numPAbins: " << numPAbins << ", PA_min: " << PA_min << ", PA_max: " << PA_max << ", s_ion: "
		<< s_ion << ", s_mag: " << s_mag << ")";

	ParticleDistribution dist{ "./", attributeNames_m, name_m, mass_m, { 0.0f, 0.0f, 0.0f, 0.0f, -1.0f }, false };
	dist.addEnergyRange(numEbins, E_min, E_max);   //defaults to log spaced bins, mid bin
	dist.addPitchRange(numPAbins, PA_min, PA_max); //defaults to linear spaced bins, mid bin
	origData_m = dist.generate(s_ion, s_mag);

	initDataLoaded_m = true;
	copyDataToGPU();
}

void Particles::loadDistFromPD(const ParticleDistribution& pd, meters s_ion, meters s_mag)
{
	origData_m = pd.generate(s_ion, s_mag);

	initDataLoaded_m = true;
	copyDataToGPU();
}

void Particles::loadDistFromPD(const ParticleDistribution& pd, vector<meters>& s)
{
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

void Particles::loadDataFromMem(const fp2Dvec& data, bool orig) //orig defaults to true
{
	((orig) ? origData_m = data : currData_m = data);
	numberOfParticles_m = ((orig) ? (int)origData_m.at(0).size() : (int)currData_m.at(0).size());

	if (orig) initDataLoaded_m = true; //copyDataToGPU uses this flag to ensure data is present in origData_m
	if (orig) copyDataToGPU();
}


void Particles::loadDataFromDisk(string folder, bool orig) //orig defaults to true
{
	for (size_t attrs = 0; attrs < attributeNames_m.size(); attrs++)
		FILE_RDWR_EXCEP_CHECK(readBin((orig ? origData_m.at(attrs) : currData_m.at(attrs)), folder + "/" + name_m + "_" + attributeNames_m.at(attrs) + ".bin", numberOfParticles_m));
	
	if (orig) initDataLoaded_m = true; //copyDataToGPU uses this flag to ensure data is present in origData_m
	if (orig) copyDataToGPU();
}

void Particles::saveDataToDisk(string folder, bool orig) const
{
	for (size_t attrs = 0; attrs < attributeNames_m.size(); attrs++)
		FILE_RDWR_EXCEP_CHECK(writeBin((orig ? origData_m.at(attrs) : currData_m.at(attrs)), folder + "/" + name_m + "_" + attributeNames_m.at(attrs) + ".bin", numberOfParticles_m));
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
	
	auto writeFPBuf = [&](const flPt_t fp)
	{   //casts to double so that SP MEPT doesn't break when loading a previously saved DP save file (and vice versa)
		double tmp{ static_cast<double>(fp) };  //cast to double precision FP
		out.write(reinterpret_cast<const char*>(&tmp), sizeof(double));
	};

	out.write(reinterpret_cast<const char*>(&numberOfParticles_m), sizeof(size_t));
	writeFPBuf(mass_m);
	writeFPBuf(charge_m);
}

void Particles::deserialize(ifstream& in) //protected function
{ //recreates a saved "serialization"
	name_m = deserializeString(in);
	attributeNames_m = deserializeStringVector(in);

	auto readFPBuf = [&]()
	{   //casts to double so that SP MEPT doesn't break when loading a previously saved DP save file (and vice versa)
		double tmp{ 0.0 };  //read in double precision FP
		in.read(reinterpret_cast<char*>(&tmp), sizeof(double));
		return tmp;
	};

	in.read(reinterpret_cast<char*>(&numberOfParticles_m), sizeof(size_t));
	mass_m = readFPBuf();
	charge_m = readFPBuf();
}
