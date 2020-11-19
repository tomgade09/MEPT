#ifndef UTILS_FILEIO_SERIALIZATION_H
#define UTILS_FILEIO_SERIALIZATION_H

#include <string>
#include <vector>
#include <fstream>
#include <sstream>

using std::string;
using std::vector;
using std::ostream;
using std::ofstream;
using std::ifstream;
using std::stringbuf;

#define doublevec vector<double>
#define stringvec vector<string>

namespace utils
{
	namespace fileIO
	{
		namespace serialize
		{
			size_t readSizetLength(ifstream& in);
			void   writeSizetLength(ofstream& out, size_t size);

			stringbuf serializeString(const string& str);
			stringbuf serializeDoubleVector(const vector<double>& vec);
			stringbuf serializeStringVector(const vector<string>& vec);

			string    deserializeString(ifstream& istr);
			doublevec deserializeDoubleVector(ifstream& istr);
			stringvec deserializeStringVector(ifstream& istr);
		}
	}
}

#endif /* !UTILS_FILEIO_SERIALIZATION_H */