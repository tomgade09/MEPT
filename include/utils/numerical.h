#ifndef UTILS_NUMERICAL_H
#define UTILS_NUMERICAL_H

#include <string>
#include <vector>
#include <iostream>

#include "dlldefines.h"
#include "physicalconstants.h"
#include "utils/unitsTypedefs.h"

using std::string;
using std::vector;

namespace utils
{
	namespace numerical
	{
		DLLEXP void v2DtoEPitch(const vector<float>& vpara, const vector<float>& vperp, float mass, vector<eV>& energies, vector<degrees>& pitches);
		DLLEXP void EPitchTov2D(const vector<eV>& energies, const vector<degrees>& pitches, float mass, vector<float>& vpara, vector<float>& vperp);
		DLLEXP vector<float> generateSpacedValues(float start, float end, int number, bool logSpaced, bool endInclusive);
		DLLEXP void normalize(vector<float>& normalizeMe, float normFactor, bool inverse = false);
		DLLEXP float calcMean(const vector<float>& calcMyMean, bool absValue = false);
		DLLEXP float calcStdDev(const vector<float>& calcMyStdDev);
		DLLEXP void coutMinMaxErr(const vector<float>& basevals, const vector<float>& testvals, string label="", bool skipzeroes = true);
		DLLEXP void coutNumAboveErrEps(const vector<float>& basevals, const vector<float>& testvals, float errEps, string label="", bool skipzeroes = true);
	}
}

#endif /* !UTILS_NUMERICAL_H */