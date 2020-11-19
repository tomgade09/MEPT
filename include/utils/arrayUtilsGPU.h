#ifndef UTILS_ARRAY_H
#define UTILS_ARRAY_H

#include <vector>

using std::vector;

namespace utils
{
	namespace GPU
	{
		void setup2DArray(double** data1D_d, double*** data2D_d, size_t outerDim, size_t innerDim);
		void copy2DArray(vector<vector<double>>& data, double** data1D_d, bool hostToDev);
		void free2DArray(double** data1D_d, double*** data2D_d);
		void getGPUMemInfo(size_t* free, size_t* total, int GPUidx);
		void getCurrGPUMemInfo(size_t* free, size_t* total);
	}
}

#endif
