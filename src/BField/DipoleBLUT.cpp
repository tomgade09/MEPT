#include "BField/DipoleB.h"
#include "BField/DipoleBLUT.h"

#include <iostream>
#include "utils/serializationHelpers.h"

using std::cerr;
using std::string;
using std::invalid_argument;
using namespace utils::fileIO::serialize;


vector<float> DipoleBLUT::getAllAttributes() const
{
	vector<float> ret{ ILAT_m, ds_msmt_m, ds_gradB_m, simMin_m, simMax_m, static_cast<float>(numMsmts_m) };
	
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

	// ======== write data to file ======== //
	writeStrBuf(serializeFloatVector(getAllAttributes()));
	writeStrBuf(serializeFloatVector(altitude_m));
	writeStrBuf(serializeFloatVector(magnitude_m));
	out.write(reinterpret_cast<const char*>(&numMsmts_m), sizeof(int));
	out.write(reinterpret_cast<const char*>(&useGPU_m), sizeof(bool));
}

void DipoleBLUT::deserialize(ifstream& in)
{
	vector<float> attrs{ deserializeFloatVector(in) };
	ILAT_m = attrs.at(0);
	ds_msmt_m = attrs.at(1);
	ds_gradB_m = attrs.at(2);
	simMin_m = attrs.at(3);
	simMax_m = attrs.at(4);
	altitude_m = deserializeFloatVector(in);
	magnitude_m = deserializeFloatVector(in);
	in.read(reinterpret_cast<char*>(&numMsmts_m), sizeof(int));
	in.read(reinterpret_cast<char*>(&useGPU_m), sizeof(bool));

}