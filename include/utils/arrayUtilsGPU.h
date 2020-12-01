#ifndef UTILS_ARRAY_H
#define UTILS_ARRAY_H

#include <vector>

using std::vector;

namespace utils
{
	namespace GPU
	{
		void setup2DArray(float** data1D_d, float*** data2D_d, size_t outerDim, size_t innerDim);
		void copy2DArray(vector<vector<float>>& data, float** data1D_d, bool hostToDev);
		void free2DArray(float** data1D_d, float*** data2D_d);
		void getGPUMemInfo(size_t* free, size_t* total, int GPUidx);
		void getCurrGPUMemInfo(size_t* free, size_t* total);
	}
}

#endif
