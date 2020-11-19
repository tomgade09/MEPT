#include "utils/strings.h"
#include <stdexcept>
#include <ctime>
#include <chrono>
#include <iomanip>
#include <sstream>

using std::stringstream;
using std::invalid_argument;
using namespace std::chrono;

namespace utils
{
	namespace strings
	{
		DLLEXP size_t findAttrInd(string attr, vector<string> allAttrs)
		{
			for (size_t ind = 0; ind < allAttrs.size(); ind++)
			{
				if (allAttrs.at(ind) == attr)
					return ind;
			}

			string allAttrsStr;
			for (size_t ind = 0; ind < allAttrs.size(); ind++)
				allAttrsStr += allAttrs.at(ind);

			throw invalid_argument("utils::string::findAttrInd: cannot find attribute " + attr + " in string " + allAttrsStr);
		}

		DLLEXP vector<string> strToStrVec(string str, const char delim) //delim defaults to ','
		{
			vector<string> strVec;

			if (str == "")
				return strVec;

			size_t loc{ 0 };
			while (loc != string::npos)
			{
				loc = str.find(delim);
				strVec.push_back(str.substr(0, loc));
				str.erase(0, loc + 1);
				while (str.at(0) == ' ')
					str.erase(0, 1);
			}

			return strVec;
		}

		DLLEXP string strVecToStr(vector<string> strVec, const char delim) //delim defaults to ','
		{
			string ret;
			for (auto& str : strVec)
				ret += str + ((str != strVec.back()) ? string{ delim } : "");

			return ret;
		}

		DLLEXP vector<double> strToDblVec(string str, const char delim) //delim defaults to ','
		{
			vector<string> strVec{ strToStrVec(str, delim) };
			vector<double> ret;

			if (strVec.size() == 0)
				return ret;

			for (size_t ind = 0; ind < strVec.size(); ind++)
				ret.push_back(atof(strVec.at(ind).c_str()));

			return ret;
		}

		DLLEXP void stringPadder(string& in, size_t totalStrLen, int indEraseFrom) //indEraseFrom defaults to 0
		{
			if (totalStrLen <= 0 || indEraseFrom < 0)
				return;

			size_t txtlen = in.length();

			if ((totalStrLen - txtlen) > 0)
			{
				for (size_t iii = 0; iii < (totalStrLen - txtlen); iii++)
					in += ' ';
			}
			else
				in.erase(indEraseFrom, txtlen - totalStrLen);
		}

		DLLEXP string getCurrentTimeString(string put_time_format)
		{
			std::time_t cdftime{ system_clock::to_time_t(system_clock::now()) };
			std::tm     tm{ *std::localtime(&cdftime) };

			stringstream filename;
			filename << std::put_time(&tm, put_time_format.c_str());

			return filename.str();
		}
	}
}
