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
		DLLEXP void readFltBin(vector<float>& arrayToReadInto, string filename);
		DLLEXP void readFltBin(vector<float>& arrayToReadInto, string filename, size_t numOfFltsToRead);
		DLLEXP void read2DCSV(vector<vector<float>>& array2DToReadInto, string filename, size_t numofentries, size_t numofcols, const char delim);
		DLLEXP void readTxtFile(string& readInto, string filename);
		DLLEXP void writeFltBin(const vector<float>& dataarray, string filename, size_t numelements, bool overwrite = true);
		DLLEXP void write2DCSV(const vector<vector<float>>& dataarray, string filename, size_t numofentries, size_t numofcols, const char delim = ',', bool overwrite = true, int precision = 20);
		DLLEXP void writeTxtFile(string textToWrite, string filename, bool overwrite = false);
	}
}
#endif /* !UTILS_FILEIO_H */