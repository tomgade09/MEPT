#include "BField/DipoleB.h"
#include "BField/DipoleBLUT.h"

#include <iostream>
#include "utils/serializationHelpers.h"

using std::cerr;
using std::string;
using std::invalid_argument;
using namespace utils::fileIO::serialize;


vector<double> DipoleBLUT::getAllAttributes() const
{
	vector<double> ret{ ILAT_m, ds_msmt_m, ds_gradB_m, simMin_m, simMax_m, static_cast<double>(numMsmts_m) };
	
	for (int iii = 0; iii < altitude_m.size(); iii++)
		ret.push_back(altitude_m.at(iii));
	for (int iii = 0; iii < magnitude_m.size(); iii++)
		ret.push_back(magnitude_m.at(iii));

	return ret;
}

void DipoleBLUT::serialize(ofstream& out) const
{
	auto writeStrBuf = [&](const stringbuf& sb)
	{
		out.write(sb.str().c_str(), sb.str().length());
	};

	// ======== write data to file ======== //
	writeStrBuf(serializeDoubleVector(getAllAttributes()));
	writeStrBuf(serializeDoubleVector(altitude_m));
	writeStrBuf(serializeDoubleVector(magnitude_m));
	out.write(reinterpret_cast<const char*>(&numMsmts_m), sizeof(int));
	out.write(reinterpret_cast<const char*>(&useGPU_m), sizeof(bool));
}

void DipoleBLUT::deserialize(ifstream& in)
{
	vector<double> attrs{ deserializeDoubleVector(in) };
	ILAT_m = attrs.at(0);
	ds_msmt_m = attrs.at(1);
	ds_gradB_m = attrs.at(2);
	simMin_m = attrs.at(3);
	simMax_m = attrs.at(4);
	altitude_m = deserializeDoubleVector(in);
	magnitude_m = deserializeDoubleVector(in);
	in.read(reinterpret_cast<char*>(&numMsmts_m), sizeof(int));
	in.read(reinterpret_cast<char*>(&useGPU_m), sizeof(bool));

}