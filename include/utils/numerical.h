#ifndef UTILS_NUMERICAL_H
#define UTILS_NUMERICAL_H

#include <string>
#include <vector>
#include <iostream>

#include "dlldefines.h"
#include "physicalconstants.h"
#include "utils/unitsTypedefs.h"

namespace utils
{
	namespace numerical
	{
		DLLEXP void    v2DtoEPitch(const vector<mpers>& vpara, const vector<mpers>& vperp, kg mass, vector<eV>& energies, vector<degrees>& pitches);
		DLLEXP void    EPitchTov2D(const vector<eV>& energies, const vector<degrees>& pitches, kg mass, vector<mpers>& vpara, vector<mpers>& vperp);
		DLLEXP fp1Dvec generateSpacedValues(flPt_t start, flPt_t end, int number, bool logSpaced, bool endInclusive);
		DLLEXP void    normalize(fp1Dvec& normalizeMe, flPt_t normFactor, bool inverse = false);
		DLLEXP flPt_t  calcMean(const fp1Dvec& calcMyMean, bool absValue = false);
		DLLEXP flPt_t  calcStdDev(const fp1Dvec& calcMyStdDev);
		DLLEXP void    coutMinMaxErr(const fp1Dvec& basevals, const fp1Dvec& testvals, string label="", bool skipzeroes = true);
		DLLEXP void    coutNumAboveErrEps(const fp1Dvec& basevals, const fp1Dvec& testvals, flPt_t errEps, string label="", bool skipzeroes = true);
	}
}

#endif /* !UTILS_NUMERICAL_H */