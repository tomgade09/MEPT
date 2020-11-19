#include "utils/random.h"

#include <random>
#include <omp.h>

namespace utils
{
	namespace random
	{
		void generateNormallyDistributedValues(double mean, double sigma, std::vector<double>& arrayOut)
		{
			std::random_device randDev;
			std::mt19937 mtgen(randDev());

			std::normal_distribution<> data_nd(mean, sigma);

			#pragma omp parallel for
			for (int iii = 0; iii < (int)arrayOut.size(); iii++)
				arrayOut.at(iii) = data_nd(mtgen);
		}
	}
}