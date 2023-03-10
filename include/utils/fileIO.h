#ifndef UTILS_FILEIO_H
#define UTILS_FILEIO_H

#include <vector>
#include <string>

#include "dlldefines.h"
#include "utils/readIOclasses.h"
#include "utils/writeIOclasses.h"

using std::string;
using std::vector;

namespace utils
{
	namespace fileIO
	{
		template <typename T1>
		DLLEXP void readBin(vector<T1>& arrayToReadInto, string filename, size_t numOfNumsToRead=0);
		template <typename T1>
		DLLEXP void writeBin(const vector<T1>& dataarray, string filename, size_t numelements, bool overwrite = true);
		
		template <typename T1>
		DLLEXP void read2DCSV(vector<vector<T1>>& array2DToReadInto, string filename, size_t numofentries, size_t numofcols, const char delim = ',');
		template <typename T1>
		DLLEXP void write2DCSV(const vector<vector<T1>>& dataarray, string filename, size_t numofentries, size_t numofcols, const char delim = ',', bool overwrite = true, int precision = 20);
		
		DLLEXP void readTxtFile(string& readInto, string filename);
		DLLEXP void writeTxtFile(string textToWrite, string filename, bool overwrite = false);

		
	}
}
#endif /* !UTILS_FILEIO_H */