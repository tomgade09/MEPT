#include "BField/DipoleB.h"
#include "BField/DipoleBLUT.h"

#include <iostream>
#include "utils/serializationHelpers.h"

using std::cerr;
using std::string;
using std::invalid_argument;
using namespace utils::fileIO::serialize;


fp1Dvec DipoleBLUT::getAllAttributes() const
{
	fp1Dvec ret{ ILAT_m, ds_msmt_m, ds_gradB_m, simMin_m, simMax_m, static_cast<flPt_t>(numMsmts_m) };
	
	for (size_t iii = 0; iii < altitude_m.size(); iii++)
		ret.push_back(altitude_m.at(iii));
	for (size_t iii = 0; iii < magnitude_m.size(); iii++)
		ret.push_back(magnitude_m.at(iii));

	return ret;
}

void DipoleBLUT::serialize(ofstream& out) const
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

	auto writeFPVecBuf = [&](const vector<flPt_t>& fpv)
	{
		vector<double> tmp;
		for (const auto& elem : fpv)
			tmp.push_back((double)elem);
		writeStrBuf(serializeDoubleVector(tmp));
	};

	// ======== write data to file ======== //
	writeFPBuf(ILAT_m);
	writeFPBuf(ds_msmt_m);
	writeFPBuf(ds_gradB_m);
	writeFPBuf(simMin_m);
	writeFPBuf(simMax_m);
	writeFPVecBuf(altitude_m);
	writeFPVecBuf(magnitude_m);
	out.write(reinterpret_cast<const char*>(&numMsmts_m), sizeof(int));
	out.write(reinterpret_cast<const char*>(&useGPU_m), sizeof(bool));
}

void DipoleBLUT::deserialize(ifstream& in)
{
	auto readFPBuf = [&]()
	{   //casts to double so that SP MEPT doesn't break when loading a previously saved DP save file (and vice versa)
		double tmp{ 0.0 };  //read in double precision FP
		in.read(reinterpret_cast<char*>(&tmp), sizeof(double));
		return tmp;
	};

	auto readFPVecBuf = [&]()
	{
		vector<double> tmp{ deserializeDoubleVector(in) };
		vector<flPt_t> ret;
		for (const auto& elem : tmp)
			ret.push_back((flPt_t)elem);
		return ret;
	};

	ILAT_m = readFPBuf();
	ds_msmt_m = readFPBuf();
	ds_gradB_m = readFPBuf();
	simMin_m = readFPBuf();
	simMax_m = readFPBuf();
	altitude_m = readFPVecBuf();
	magnitude_m = readFPVecBuf();
	in.read(reinterpret_cast<char*>(&numMsmts_m), sizeof(int));
	in.read(reinterpret_cast<char*>(&useGPU_m), sizeof(bool));

}