#ifndef UTILS_RANDOM_H
#define UTILS_RANDOM_H

#include <vector>

namespace utils
{
	namespace random
	{
		void generateNormallyDistributedValues(float mean, float sigma, std::vector<float>& arrayOut);
	}
}

#endif /* !UTILS_RANDOM_H */