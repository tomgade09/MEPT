#include "BField/DipoleB.h"

#include <iostream>
#include "utils/serializationHelpers.h"

using std::cerr;
using std::string;
using namespace utils::fileIO::serialize;

vector<double> DipoleB::getAllAttributes() const
{
	return { L_m, L_norm_m, s_max_m, ILAT_m, ds_m, lambdaErrorTolerance_m };
}

void DipoleB::serialize(ofstream& out) const
{
	auto writeStrBuf = [&](const stringbuf& sb)
	{
		out.write(sb.str().c_str(), sb.str().length());
	};

	// ======== write data to file ======== //
	writeStrBuf(serializeDoubleVector(getAllAttributes()));
	out.write(reinterpret_cast<const char*>(&useGPU_m), sizeof(bool));
}

void DipoleB::deserialize(ifstream& in)
{
	vector<double> attrs{ deserializeDoubleVector(in) };
	L_m = attrs.at(0);
	L_norm_m = attrs.at(1);
	s_max_m = attrs.at(2);
	ILAT_m = attrs.at(3);
	ds_m = attrs.at(4);
	lambdaErrorTolerance_m = attrs.at(5);
	in.read(reinterpret_cast<char*>(&useGPU_m), sizeof(bool));
}