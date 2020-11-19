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
		DLLEXP void v2DtoEPitch(const vector<double>& vpara, const vector<double>& vperp, double mass, vector<eV>& energies, vector<degrees>& pitches);
		DLLEXP void EPitchTov2D(const vector<eV>& energies, const vector<degrees>& pitches, double mass, vector<double>& vpara, vector<double>& vperp);
		DLLEXP vector<double> generateSpacedValues(double start, double end, int number, bool logSpaced, bool endInclusive);
		DLLEXP void normalize(vector<double>& normalizeMe, double normFactor, bool inverse = false);
		DLLEXP double calcMean(const vector<double>& calcMyMean, bool absValue = false);
		DLLEXP double calcStdDev(const vector<double>& calcMyStdDev);
		DLLEXP void coutMinMaxErr(const vector<double>& basevals, const vector<double>& testvals, string label="", bool skipzeroes = true);
		DLLEXP void coutNumAboveErrEps(const vector<double>& basevals, const vector<double>& testvals, double errEps, string label="", bool skipzeroes = true);
	}
}

#endif /* !UTILS_NUMERICAL_H */