#include "EField/QSPS.h"

#include <iostream>
#include <filesystem>

using std::cerr;
using std::to_string;
using std::invalid_argument;
using namespace utils::fileIO::serialize;

vector<double> QSPS::getAllAttributes() const
{
	vector<double> ret;
	for (int iii = 0; iii < altMin_m.size(); iii++)
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

	// ======== write data to file ======== //
	writeStrBuf(serializeDoubleVector(altMin_m));
	writeStrBuf(serializeDoubleVector(altMax_m));
	writeStrBuf(serializeDoubleVector(magnitude_m));
	out.write(reinterpret_cast<const char*>(&useGPU_m), sizeof(bool));
}

void QSPS::deserialize(ifstream& in)
{
	altMin_m = deserializeDoubleVector(in);
	altMax_m = deserializeDoubleVector(in);
	magnitude_m = deserializeDoubleVector(in);
	in.read(reinterpret_cast<char*>(&useGPU_m), sizeof(bool));
}