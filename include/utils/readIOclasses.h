#ifndef UTILS_IOCLASSES_READ_H
#define UTILS_IOCLASSES_READ_H

#include <vector>
#include <string>

#include "dlldefines.h"

using std::vector;
using std::string;

namespace utils
{
	namespace fileIO
	{
		class DistributionFromDisk
		{
		private:
			string name_m;
			vector<string> attrNames_m;
			vector<vector<double>> data_m;
			double mass_m;


		public:
			DistributionFromDisk(string name, string folder, string partName, vector<string> attrNames, double mass);
			~DistributionFromDisk() {}

			const vector<vector<double>>& data() const { return data_m; }
			const string& name() const { return name_m; }
			void print(int at) const;
			void printdiff(DistributionFromDisk& other, int at) const;
			void zeroes() const;
			void zeroes(vector<int>& zeroes, bool print = true) const;
			void compare(const DistributionFromDisk& other) const;
		};
	}
}

#endif /* UTILS_IOCLASSES_READ_H */