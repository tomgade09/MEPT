#ifndef UTILS_ARRAY_H
#define UTILS_ARRAY_H

#include "utils/unitsTypedefs.h"

namespace utils
{
	namespace GPU
	{
		size_t getDeviceCount();
		void setDev(size_t devNum);
		std::string getDeviceNames();
		void setup2DArray(flPt_t** data1D_d, vector<flPt_t*>& data2D_d, size_t outerDim, size_t innerDim, size_t devNum);
		void setup2DArray(flPt_t** data1D_d, flPt_t*** data2D_d, size_t outerDim, size_t innerDim, size_t devNum);
		void copy2DArray(fp2Dvec& data, flPt_t** data1D_d, bool hostToDev, size_t devNum);
		void free2DArray(flPt_t** data1D_d, flPt_t*** data2d_d, size_t devNum);
		void getGPUMemInfo(size_t* free, size_t* total, size_t devNum);
		void getCurrGPUMemInfo(size_t* free, size_t* total);
		vector<size_t> getSplitSize(size_t numOfParticles, size_t blocksize);
	}
}

#endif
