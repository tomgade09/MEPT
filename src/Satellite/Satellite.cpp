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

using std::clog;
using std::ofstream;
using std::streambuf;
using std::to_string;
using std::logic_error;
using std::out_of_range;
using std::runtime_error;
using std::invalid_argument;

using utils::fileIO::readBin;
using utils::fileIO::writeBin;
using namespace utils::fileIO::serialize;


template<typename T1>
SatData<T1>::SatData(const T1& vpara_i, const T1& mu_i, const T1& s_i, const T1& t_i) :
	vpara{ vpara_i }, mu{ mu_i }, s{ s_i }, t_detect{ t_i }
{

}

template<typename T1>
T1& SatData<T1>::at(size_t loc)
{
	if (loc == 0) return vpara;
	else if (loc == 1) return mu;
	else if (loc == 2) return s;
	else if (loc == 3) return t_detect;
	else throw out_of_range("SatData out of range: " + to_string(loc));
}

template<typename T1>
const T1& SatData<T1>::__at(size_t loc) const
{
	if (loc == 0) return vpara;
	else if (loc == 1) return mu;
	else if (loc == 2) return s;
	else if (loc == 3) return t_detect;
	else throw out_of_range("SatData out of range: " + to_string(loc));
}

template struct SatData<flPt_t*>;
template struct SatData<fp1Dvec>;


Satellite::Satellite(string& name, meters altitude, bool upwardFacing, size_t numberOfParticles) :
	name_m{ name }, altitude_m{ altitude }, upwardFacing_m{ upwardFacing }, numberOfParticles_m{ numberOfParticles }
{
	data_m = SatDataVecs{ fp1Dvec(numberOfParticles), fp1Dvec(numberOfParticles),
						  fp1Dvec(numberOfParticles), fp1Dvec(numberOfParticles) };
	initializeGPU();
}

Satellite::Satellite(ifstream& in)
{
	deserialize(in);

	data_m = SatDataVecs{ fp1Dvec(numberOfParticles_m), fp1Dvec(numberOfParticles_m),
						  fp1Dvec(numberOfParticles_m), fp1Dvec(numberOfParticles_m) };
	
	initializeGPU();
}

Satellite::~Satellite()
{
	freeGPUMemory();
}

void Satellite::initializeGPU()
{
	numGPUs_m = utils::GPU::getDeviceCount();
	particleCountPerGPU_m = utils::GPU::getSplitSize(numberOfParticles_m, BLOCKSIZE);

	for (size_t dev = 0; dev < numGPUs_m; ++dev)
	{
		gpuMemRegion_d.push_back(nullptr);

		vector<flPt_t*> data_d(SatDataPtrs::size, nullptr);
		
		utils::GPU::setup2DArray(&gpuMemRegion_d.at(dev), data_d, //takes care of setting device number
			data_d.size(), particleCountPerGPU_m.at(dev), dev);

		gpuDataPtrs_d.emplace_back(data_d.at(0), data_d.at(1),
			data_d.at(2), data_d.at(3)); //constructs a SatDataPtrs directly in memory
		
		clog << "Satellite::initializeGPU() : setup2DArray GPU num " + to_string(dev) +
			    ": length " + to_string(particleCountPerGPU_m.at(dev)) << "\n";
	}
	initializedGPU_m = true;
}

//void Satellite::iterateDetectorCPU is in Detector.cpp

void Satellite::copyDataToHost()
{
	if (!initializedGPU_m) return;
	
	vector<SatDataVecs> data_GPU;

	for (size_t dev = 0; dev < numGPUs_m; dev++)
	{
		size_t cnt{ particleCountPerGPU_m.at(dev) };

		fp2Dvec data(data_m.size, fp1Dvec(cnt));
		
		// Copy data in data_m arrays
		utils::GPU::copy2DArray(data, &gpuMemRegion_d.at(dev), false, dev);

		data_GPU.emplace_back(move(data.at(0)), move(data.at(1)), move(data.at(2)),
			move(data.at(3)));
	}

	data_m = data_GPU.at(0);
	for (size_t dev = 1; dev < numGPUs_m; ++dev)
	{
		for (size_t attr = 0; attr < data_m.size; attr++)
		{
			// Copy data_m arrays into data_m array
			data_m.at(attr).reserve(data_m.at(attr).size() + data_GPU.at(dev).at(attr).size());
			data_m.at(attr).insert(data_m.at(attr).end(), data_GPU.at(dev).at(attr).begin(),
				data_GPU.at(dev).at(attr).end());
		}
	}
}

void Satellite::freeGPUMemory()
{
	if (!initializedGPU_m) return;

	for (size_t dev = 0; dev < numGPUs_m; ++dev)
		utils::GPU::free2DArray(&gpuMemRegion_d.at(dev), nullptr, dev);

	gpuDataPtrs_d.clear();
	
	initializedGPU_m = false;
}

SatDataVecs Satellite::removeZerosData(vector<int>& indices)
{
	copyDataToHost();

	SatDataVecs dcopy{ data_m }; //don't want to do this to the live data so create a copy
	vector<bool> tdel( dcopy.at(0).size(), false ); //create bool vec of whether t's < 0 - to remove sparse data

	indices.resize(dcopy.vpara.size());
	for (int ind = 0; ind < (int)indices.size(); ind++) //set index value to index number
		indices.at(ind) = ind;
	
	for (size_t t = 0; t < tdel.size(); t++)
		tdel.at(t) = (dcopy.t_detect.at(t) < 0.0f);
	
	for (size_t attr = 0; attr < SatDataVecs::size; attr++)
	{//below searches time vector copy for -1.0 and removes the element if so (no negatives should exist except -1)
		fp1Dvec& atref{ dcopy.at(attr) }; //reference to attribute within dcopy
		
		auto checkIfNegOne = [&](flPt_t& x)->bool
		{
			return tdel.at(&x - &(*atref.begin()));
		};

		atref.erase(remove_if(atref.begin(), atref.end(), checkIfNegOne), atref.end());
	}

	auto checkIfNegOne = [&](int& x)->bool
	{
		return tdel.at(&x - &(*indices.begin()));
	};
	indices.erase(remove_if(indices.begin(), indices.end(), checkIfNegOne), indices.end());
	
	return dcopy;
}

void Satellite::saveDataToDisk(string folder) //move B and mass to getConsolidatedData and have it convert back (or in gpu?)
{
	vector<int> indices;
	SatDataVecs results{ removeZerosData(indices) };

	for (size_t attr = 0; attr < results.size; attr++)
		writeBin(results.at(attr), folder + name_m + "_" + names_m.at(attr) + ".bin", results.at(attr).size());

	writeBin(indices, folder + name_m + "_index.bin", indices.size());
}

void Satellite::loadDataFromDisk(string folder)
{
	for (size_t attr = 0; attr < SatDataVecs::size; attr++)
		readBin(data_m.at(attr), folder + name_m + "_" + names_m.at(attr) + ".bin");

	vector<int> indices;
	readBin(indices, folder + name_m + "_index.bin");

	bool expand{ false };
	if (data_m.at(0).size() == numberOfParticles_m)
		expand = false;
	else if (data_m.at(0).size() < numberOfParticles_m)
		expand = true;
	else
		throw logic_error("Satellite::loadDataFromDisk: number of particles loaded from disk is greater than specified numberOfParticles_m.  "
			+ string("That means that the wrong data was loaded or wrong number of particles was specified.  Not loading data."));

	if (expand)  //expand into sparse array
	{
		for (size_t attr = 0; attr < data_m.size; attr++) //add extra zeroes to make vectors the right size
			data_m.at(attr).resize(numberOfParticles_m); //1-4 do this

		indices.resize(numberOfParticles_m, -1); //5 do this

		for (size_t aloc = numberOfParticles_m - 1; aloc >= 0; aloc--) //array location, iterating backwards
		{
			int pind{ indices.at(aloc) };   //the value in the index array for this particle

			if (pind != -1 && static_cast<size_t>(pind) != aloc) //if not in resized portion, and index doesn't match array location
			{
				if (data_m.vpara.at(aloc) != 0.0f || data_m.mu.at(aloc) != 0.0f || data_m.s.at(aloc) != 0.0f)
					throw runtime_error("Satellite::loadDataFromDisk: data is being overwritten in reconstructed array - something is wrong " + to_string(pind));
				if (pind < aloc)
					throw runtime_error("Satellite::loadDataFromDisk: data is being moved to a lower index.  This shouldn't happen due to the compression algorithm.");
				
				data_m.vpara.at(pind) = data_m.vpara.at(aloc); //move current array location to its actual index
				data_m.mu.at(pind) = data_m.mu.at(aloc);
				data_m.s.at(pind) = data_m.s.at(aloc);
				data_m.t_detect.at(pind) = data_m.t_detect.at(aloc);
				indices.at(pind) = indices.at(aloc);

				data_m.vpara.at(aloc) = 0.0f; //erase value at old location
				data_m.mu.at(aloc) = 0.0f;
				data_m.s.at(aloc) = 0.0f;
				data_m.t_detect.at(aloc) = -1.0f;
				indices.at(aloc) = (int)aloc; //set index value to current location in array
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

SatDataVecs& Satellite::__data()
{
	return data_m;
}

const SatDataVecs& Satellite::data() const
{
	return data_m;
}

size_t Satellite::getNumberOfAttributes() const
{
	return data_m.size;
}

size_t Satellite::getNumberOfParticles() const
{
	return numberOfParticles_m;
}

size_t Satellite::getNumParticlesPerGPU(int GPUind) const
{
	return particleCountPerGPU_m.at(GPUind);
}

Sat_d Satellite::getSat_d(int GPUind) const
{
	return Sat_d{ gpuDataPtrs_d.at(GPUind), altitude_m, upwardFacing_m };
}

SatDataPtrs Satellite::getPtrs_d(int GPUind) const
{
	return gpuDataPtrs_d.at(GPUind);
}


void Satellite::serialize(ofstream& out) const
{
	auto writeStrBuf = [&](const stringbuf& sb)
	{
		out.write(sb.str().c_str(), sb.str().length());
	};

	auto writeFPBuf = [&](const flPt_t fp)
	{   //casts to double so that SP MEPT doesn't break when loading a previously saved DP save file (and vice versa)
		double tmp{ static_cast<double>(fp) };  //cast to double precision FP
		out.write(reinterpret_cast<const char*>(&tmp), sizeof(double));
	};

	writeStrBuf(serializeString(name_m));
	writeStrBuf(serializeStringVector(names_m));

	writeFPBuf(altitude_m);
	out.write(reinterpret_cast<const char*>(&upwardFacing_m), sizeof(bool));
	out.write(reinterpret_cast<const char*>(&numberOfParticles_m), sizeof(size_t));
}

void Satellite::deserialize(ifstream& in)
{
	name_m = deserializeString(in);
	//data_m.names = deserializeStringVector(in); //this is implementation defined, eventually remove saving to disk
	vector<string> dump = deserializeStringVector(in); //vector to dump into, not used
	for (size_t name = 0; name < dump.size(); name++)
	{ //provides a way to check that implementations are the same
		try
		{
			if (dump.at(name) != names_m.at(name))
				throw logic_error("Satellite::deserialize: implementation of satellite data struct has changed");
		}
		catch (std::exception& e)
		{
			std::cerr << e.what() << "\n";
			throw;
		}
	}

	auto readFPBuf = [&]()
	{   //casts to double so that SP MEPT doesn't break when loading a previously saved DP save file (and vice versa)
		double tmp{ 0.0 };  //read in double precision FP
		in.read(reinterpret_cast<char*>(&tmp), sizeof(double));
		return tmp;
	};
	
	altitude_m = readFPBuf();
	in.read(reinterpret_cast<char*>(&upwardFacing_m), sizeof(bool));
	in.read(reinterpret_cast<char*>(&numberOfParticles_m), sizeof(size_t));
}
