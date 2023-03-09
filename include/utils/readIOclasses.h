#ifndef UTILS_IOCLASSES_READ_H
#define UTILS_IOCLASSES_READ_H

#include "dlldefines.h"
#include "unitsTypedefs.h"

namespace utils
{
	namespace fileIO
	{
		class DistributionFromDisk
		{
		private:
			string  name_m;
			strvec  attrNames_m;
			fp2Dvec data_m;
			kg      mass_m;


		public:
			DistributionFromDisk(string name, string folder, string partName, strvec attrNames, kg mass);
			~DistributionFromDisk() {}

			const fp2Dvec& data() const { return data_m; }
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