#ifndef UTILS_ARRAY_H
#define UTILS_ARRAY_H

#include <string>
#include <vector>

using std::vector;

namespace utils
{
	namespace GPU
	{
		size_t getDeviceCount();
		void setDev(size_t devNum);
		std::string getDeviceNames();
		void setup2DArray(float** data1D_d, float*** data2D_d, size_t outerDim, size_t innerDim, size_t devNum);
		void copy2DArray(vector<vector<float>>& data, float** data1D_d, bool hostToDev, size_t devNum);
		void free2DArray(float** data1D_d, float*** data2D_d, size_t devNum);
		void getGPUMemInfo(size_t* free, size_t* total, size_t devNum);
		void getCurrGPUMemInfo(size_t* free, size_t* total);
	}
}

#endif
