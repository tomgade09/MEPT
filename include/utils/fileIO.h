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
		DLLEXP void readDblBin(vector<double>& arrayToReadInto, string filename);
		DLLEXP void readDblBin(vector<double>& arrayToReadInto, string filename, size_t numOfDblsToRead);
		DLLEXP void read2DCSV(vector<vector<double>>& array2DToReadInto, string filename, size_t numofentries, size_t numofcols, const char delim);
		DLLEXP void readTxtFile(string& readInto, string filename);
		DLLEXP void writeDblBin(const vector<double>& dataarray, string filename, size_t numelements, bool overwrite = true);
		DLLEXP void write2DCSV(const vector<vector<double>>& dataarray, string filename, size_t numofentries, size_t numofcols, const char delim = ',', bool overwrite = true, int precision = 20);
		DLLEXP void writeTxtFile(string textToWrite, string filename, bool overwrite = false);
	}
}
#endif /* !UTILS_FILEIO_H */