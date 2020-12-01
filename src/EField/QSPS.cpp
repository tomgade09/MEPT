#include "EField/QSPS.h"

#include <iostream>
//#include <filesystem>

using std::cerr;
using std::to_string;
using std::invalid_argument;
using namespace utils::fileIO::serialize;

vector<float> QSPS::getAllAttributes() const
{
	vector<float> ret;
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

	// ======== write data to file ======== //
	writeStrBuf(serializeFloatVector(altMin_m));
	writeStrBuf(serializeFloatVector(altMax_m));
	writeStrBuf(serializeFloatVector(magnitude_m));
	out.write(reinterpret_cast<const char*>(&useGPU_m), sizeof(bool));
}

void QSPS::deserialize(ifstream& in)
{
	altMin_m = deserializeFloatVector(in);
	altMax_m = deserializeFloatVector(in);
	magnitude_m = deserializeFloatVector(in);
	in.read(reinterpret_cast<char*>(&useGPU_m), sizeof(bool));
}