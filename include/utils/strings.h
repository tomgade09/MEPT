#ifndef UTILS_STRING_H
#define UTILS_STRING_H

#include <string>
#include <vector>
#include "dlldefines.h"

using std::vector;
using std::string;

namespace utils
{
	namespace strings
	{
		DLLEXP size_t findAttrInd(string attr, vector<string> allAttrs);
		DLLEXP vector<string> strToStrVec(string str, const char delim = ',');
		DLLEXP string strVecToStr(vector<string> strVec, const char delim = ',');
		DLLEXP vector<double> strToDblVec(string str, const char delim = ',');
		DLLEXP void stringPadder(string& in, size_t totalStrLen, int indEraseFrom = 0);
		DLLEXP string getCurrentTimeString(string put_time_format);
	}
}

#endif /* !UTILS_STRING_H */