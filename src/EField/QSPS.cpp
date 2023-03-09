#include "EField/QSPS.h"

#include <iostream>
//#include <filesystem>

using std::cerr;
using std::to_string;
using std::invalid_argument;
using namespace utils::fileIO::serialize;

fp1Dvec QSPS::getAllAttributes() const
{
	fp1Dvec ret;
	for (size_t iii = 0; iii < altMin_m.size(); iii++)
	{//vectors are guaranteed to be the same size
		ret.push_back(altMin_m.at(iii));
		ret.push_back(altMax_m.at(iii));
		ret.push_back(magnitude_m.at(iii));
	}

	return ret;
}

void QSPS::serialize(ofstream& out) const
{
	auto writeStrBuf = [&](const stringbuf& sb)
	{
		out.write(sb.str().c_str(), sb.str().length());
	};

	auto writeFPVecBuf = [&](const vector<flPt_t>& fpv)
	{   //casts to double so that SP MEPT doesn't break when loading a previously saved DP save file (and vice versa)
		vector<double> tmp;
		for (const auto& elem : fpv)
			tmp.push_back((double)elem);
		writeStrBuf(serializeDoubleVector(tmp));
	};

	// ======== write data to file ======== //
	writeFPVecBuf(altMin_m);
	writeFPVecBuf(altMax_m);
	writeFPVecBuf(magnitude_m);
	out.write(reinterpret_cast<const char*>(&useGPU_m), sizeof(bool));
}

void QSPS::deserialize(ifstream& in)
{
	auto readFPVecBuf = [&]()
	{   //casts to double so that SP MEPT doesn't break when loading a previously saved DP save file (and vice versa)
		vector<double> tmp{ deserializeDoubleVector(in) };
		vector<flPt_t> ret;

		for (const auto& elem : tmp)
			ret.push_back((flPt_t)elem);

		return ret;
	};

	altMin_m = readFPVecBuf();
	altMax_m = readFPVecBuf();
	magnitude_m = readFPVecBuf();
	in.read(reinterpret_cast<char*>(&useGPU_m), sizeof(bool));
}