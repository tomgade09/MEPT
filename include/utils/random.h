#ifndef UTILS_RANDOM_H
#define UTILS_RANDOM_H

#include <vector>

namespace utils
{
	namespace random
	{
		void generateNormallyDistributedValues(double mean, double sigma, std::vector<double>& arrayOut);
	}
}

#endif /* !UTILS_RANDOM_H */