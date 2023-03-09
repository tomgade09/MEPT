#include "BField/DipoleB.h"

#include <iostream>
#include "utils/serializationHelpers.h"

using std::cerr;
using std::string;
using namespace utils::fileIO::serialize;

fp1Dvec DipoleB::getAllAttributes() const
{
	return { L_m, L_norm_m, s_max_m, ILAT_m, ds_m, lambdaErrorTolerance_m };
}

void DipoleB::serialize(ofstream& out) const
{
	auto writeFPBuf = [&](const flPt_t fp)
	{   //casts to double so that SP MEPT doesn't break when loading a previously saved DP save file (and vice versa)
		double tmp{ static_cast<double>(fp) };  //cast to double precision FP
		out.write(reinterpret_cast<const char*>(&tmp), sizeof(double));
	};

	// ======== write data to file ======== //
	writeFPBuf(L_m);
	writeFPBuf(L_norm_m);
	writeFPBuf(s_max_m);
	writeFPBuf(ILAT_m);
	writeFPBuf(ds_m);
	writeFPBuf(lambdaErrorTolerance_m);
	out.write(reinterpret_cast<const char*>(&useGPU_m), sizeof(bool));
}

void DipoleB::deserialize(ifstream& in)
{
	auto readFPBuf = [&]()
	{   //casts to double so that SP MEPT doesn't break when loading a previously saved DP save file (and vice versa)
		double tmp{ 0.0 };  //read in double precision FP
		in.read(reinterpret_cast<char*>(&tmp), sizeof(double));
		return tmp;
	};

	L_m = readFPBuf();
	L_norm_m = readFPBuf();
	s_max_m = readFPBuf();
	ILAT_m = readFPBuf();
	ds_m = readFPBuf();
	lambdaErrorTolerance_m = readFPBuf();
	in.read(reinterpret_cast<char*>(&useGPU_m), sizeof(bool));
}