//STL includes
#include <iterator> //for back_inserter
#include <algorithm>
#include <fstream>

//Project specific includes
#include "Satellite/Satellite.h"
#include "utils/fileIO.h"
#include "utils/arrayUtilsGPU.h"
#include "utils/serializationHelpers.h"
#include "ErrorHandling/simExceptionMacros.h"

using std::cerr;
using std::ofstream;
using std::streambuf;
using std::to_string;
using std::logic_error;
using std::runtime_error;
using std::invalid_argument;

using utils::fileIO::readFltBin;
using utils::fileIO::writeFltBin;
using namespace utils::fileIO::serialize;

Satellite::Satellite(string name, vector<string> attributeNames, meters altitude, bool upwardFacing, size_t numberOfParticles, const std::shared_ptr<Particles> &particle, size_t numGPUs, vector<size_t> particleCountPerGPU) :
	name_m{ name }, attributeNames_m{ attributeNames }, altitude_m{ altitude }, upwardFacing_m{ upwardFacing }, numberOfParticles_m{ numberOfParticles }, numGPUs_m{ numGPUs }, particleCountPerGPU_m{particleCountPerGPU}
{
	data_GPU_m.clear();
	for (int dev = 0; dev < numGPUs; ++dev)
	{
		particleData2D_d.push_back(particle->getCurrDataGPUPtr(dev));
		data_GPU_m.push_back( vector<vector<float>>(attributeNames_m.size(), vector<float>(particleCountPerGPU.at(dev))));
	}
	data_m = vector<vector<float>>(attributeNames_m.size(), vector<float>(numberOfParticles_m));
	initializeGPU();
}

Satellite::Satellite(ifstream& in, const std::shared_ptr<Particles> &particle, size_t numGPUs, vector<size_t> particleCountPerGPU) 
	: numGPUs_m {numGPUs}, particleCountPerGPU_m{particleCountPerGPU}
{
	data_GPU_m.clear();
	for (int dev = 0; dev < numGPUs; ++dev)
	{
		particleData2D_d.push_back(particle->getCurrDataGPUPtr(dev));
		data_GPU_m.push_back(vector<vector<float>>(attributeNames_m.size(), vector<float>(particleCountPerGPU.at(dev))));
	}
	deserialize(in);

	data_m = vector<vector<float>>(attributeNames_m.size(), vector<float>(numberOfParticles_m));
	initializeGPU();
}

Satellite::~Satellite()
{
	freeGPUMemory();
}

void Satellite::initializeGPU()
{
	for (int dev = 0; dev < numGPUs_m; ++dev)
	{
		satCaptrData1D_d.push_back(nullptr);
		satCaptrData2D_d.push_back(nullptr);
		utils::GPU::setup2DArray(&satCaptrData1D_d.at(dev), &satCaptrData2D_d.at(dev),
			attributeNames_m.size(), particleCountPerGPU_m.at(dev), dev);

		cerr << "::DEBUG(not error):: Satellite::initializeGPU() : setup2DArray GPU num " + to_string(dev) +
			    ": length " + to_string(particleCountPerGPU_m.at(dev));
	}
	initializedGPU_m = true;
}

size_t Satellite::getAttrIndByName(string name)
{
	for (size_t attr = 0; attr < attributeNames_m.size(); attr++)
	{
		if (name == attributeNames_m.at(attr))
			return attr;
	}
	
	throw invalid_argument("Satellite::getAttrIndByName: attribute name " + name + " doesn't exist.");
}

//void Satellite::iterateDetectorCPU is in Detector.cpp
//void Satellite::iterateDetector is in Detector.cu

void Satellite::copyDataToHost()
{// data_m array: [v_para, mu, s, time, partindex][particle number]
	//vector<vector<float>>  data; //[attribute][particle]
	//data_m.clear();

	for (int dev = 0; dev < numGPUs_m; ++dev)
	{
		// Copy data in data_m arrays
		utils::GPU::copy2DArray(data_GPU_m.at(dev), &satCaptrData1D_d.at(dev), false, dev);
	}
	data_m = data_GPU_m.at(0);
	for (int dev = 1; dev < numGPUs_m; ++dev)
	{
		for (int attr = 0; attr < attributeNames_m.size(); attr++)
		{
			// Copy data_m arrays into data_m array
			data_m.at(attr).reserve(data_m.at(attr).size() + data_GPU_m.at(dev).at(attr).size());
			data_m.at(attr).insert(data_m.at(attr).end(), data_GPU_m.at(dev).at(attr).begin(),
				data_GPU_m.at(dev).at(attr).end());
		}
	}

	for (size_t idx = 0; idx < data_m.at(4).size(); idx++)
		if (data_m.at(4).at(idx) > -0.5f)
			data_m.at(4).at(idx) = static_cast<float>(idx);
}

void Satellite::freeGPUMemory()
{
	if (!initializedGPU_m) { return; }

	for (int dev = 0; dev < numGPUs_m; ++dev)
	{
		utils::GPU::free2DArray(&satCaptrData1D_d.at(dev), &satCaptrData2D_d.at(dev), dev);
	}

	particleData2D_d.clear();
	initializedGPU_m = false;
}

vector<vector<float>> Satellite::removeZerosData()
{//GOING TO HAVE TO REMOVE TO SOMEWHERE - IMPLEMENTATION DEFINED, NOT GENERIC
	copyDataToHost();

	vector<vector<float>> dataCopy{ data_m }; //don't want to do this to the live data so create a copy
	vector<float> timeCopy{ dataCopy.at(getAttrIndByName("time")) }; //make a copy, because t_esc's zeroes are removed as well

	for (auto& attr : dataCopy)
	{//below searches time vector copy for -1.0 and removes the element if so (no negatives should exist except -1)
		auto checkIfNegOne = [&](float& x)
		{
			return (timeCopy.at(&x - &(*attr.begin())) < 0.0f);
		};

		attr.erase(remove_if(attr.begin(), attr.end(), checkIfNegOne), attr.end());
	}

	return dataCopy;
}

void Satellite::saveDataToDisk(string folder) //move B and mass to getConsolidatedData and have it convert back (or in gpu?)
{
	vector<vector<float>> results{ removeZerosData() };
	
	for (size_t attr = 0; attr < results.size(); attr++)
		writeFltBin(results.at(attr), folder + name_m + "_" + attributeNames_m.at(attr) + ".bin", results.at(attr).size());
}

void Satellite::loadDataFromDisk(string folder)
{
	data_m = vector<vector<float>>(attributeNames_m.size()); //this is done so readFltBin doesn't assume how many particles it's reading

	for (size_t attr = 0; attr < attributeNames_m.size(); attr++)
		readFltBin(data_m.at(attr), folder + name_m + "_" + attributeNames_m.at(attr) + ".bin");

	bool expand{ false };
	if (data_m.at(0).size() == static_cast<size_t>(numberOfParticles_m))
		expand = false;
	else if (data_m.at(0).size() < static_cast<size_t>(numberOfParticles_m))
		expand = true;
	else
		throw logic_error("Satellite::loadDataFromDisk: number of particles loaded from disk is greater than specified numberOfParticles_m.  "
			+ string("That means that the wrong data was loaded or wrong number of particles was specified.  Not loading data."));

	if (!expand)
	{
		return;
	}
	else //expand into a sparse array
	{
		size_t index{ getAttrIndByName("index") };
		size_t t_esc{ getAttrIndByName("time") };

		for (auto& attr : data_m) //add extra zeroes to make vectors the right size
			attr.resize(numberOfParticles_m);

		for (int part = numberOfParticles_m - 1; part >= 0; part--) //particles, iterating backwards
		{
			int originalParticleIndex{ (int)data_m.at(index).at(part) }; //original index of the particle
			if (originalParticleIndex == 0 && data_m.at(0).at(part) == 0.0 && data_m.at(1).at(part) == 0.0f)
			{
				data_m.at(t_esc).at(part) = -1;
				data_m.at(index).at(part) = -1;
			}
			else if ((originalParticleIndex != part) && (originalParticleIndex != -1))
			{
				if ((int)data_m.at(0).at(originalParticleIndex) != 0.0f)
					throw runtime_error("Satellite::loadDataFromDisk: data is being overwritten in reconstructed array - something is wrong " + to_string(originalParticleIndex));

				for (size_t attr = 0; attr < data_m.size(); attr++)
				{
					data_m.at(attr).at(originalParticleIndex) = data_m.at(attr).at(part); //move attr at the current location in iteration - part - to the index where it should be - ind

					if (attr == index || attr == t_esc) data_m.at(attr).at(part) = -1.0f;
					else data_m.at(attr).at(part) = 0.0f; //overwrite the old data with 0s and -1s
				}
			}

		}
	}
}

string Satellite::name() const
{
	return name_m;
}

meters Satellite::altitude() const
{
	return altitude_m;
}

bool Satellite::upward() const
{
	return upwardFacing_m;
}

vector<vector<float>>& Satellite::__data()
{
	return data_m;
}

const vector<vector<float>>& Satellite::data() const
{
	return data_m;
}

float** Satellite::get2DDataGPUPtr(int GPUind) const
{
	return satCaptrData2D_d.at(GPUind);
}

float* Satellite::get1DDataGPUPtr(int GPUind) const
{
	return satCaptrData1D_d.at(GPUind);
}

size_t Satellite::getNumberOfAttributes() const
{
	return attributeNames_m.size();
}

size_t Satellite::getNumberOfParticles() const
{
	return numberOfParticles_m;
}

void Satellite::serialize(ofstream& out) const
{
	auto writeStrBuf = [&](const stringbuf& sb)
	{
		out.write(sb.str().c_str(), sb.str().length());
	};

	writeStrBuf(serializeString(name_m));
	writeStrBuf(serializeStringVector(attributeNames_m));

	out.write(reinterpret_cast<const char*>(&altitude_m), sizeof(meters));
	out.write(reinterpret_cast<const char*>(&upwardFacing_m), sizeof(bool));
	out.write(reinterpret_cast<const char*>(&initializedGPU_m), sizeof(bool));
	out.write(reinterpret_cast<const char*>(&numberOfParticles_m), sizeof(size_t));
}

void Satellite::deserialize(ifstream& in)
{
	name_m = deserializeString(in);
	attributeNames_m = deserializeStringVector(in);

	in.read(reinterpret_cast<char*>(&altitude_m), sizeof(meters));
	in.read(reinterpret_cast<char*>(&upwardFacing_m), sizeof(bool));
	in.read(reinterpret_cast<char*>(&initializedGPU_m), sizeof(bool));
	in.read(reinterpret_cast<char*>(&numberOfParticles_m), sizeof(size_t));
}
