#ifndef UTILS_STRING_H
#define UTILS_STRING_H

#include "dlldefines.h"
#include "utils/unitsTypedefs.h"

namespace utils
{
	namespace strings
	{
		DLLEXP size_t  findAttrInd(string attr, vector<string> allAttrs);
		DLLEXP strvec  strToStrVec(string str, const char delim = ',');
		DLLEXP string  strVecToStr(vector<string> strVec, const char delim = ',');
		DLLEXP fp1Dvec strToFPVec(string str, const char delim = ',');
		DLLEXP void    stringPadder(string& in, size_t totalStrLen, int indEraseFrom = 0);
		DLLEXP string  getCurrentTimeString(string put_time_format);
	}
}

#endif /* !UTILS_STRING_H */