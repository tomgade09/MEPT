#ifndef CUDAERRORCHECK_H
#define CUDAERRORCHECK_H

/*
	Code originally posted at: https://codeyarns.com/2011/03/02/how-to-do-error-checking-in-cuda/
	Modified slightly
*/

#include <iostream>

//CUDA includes
#include "cuda_runtime.h"

#define CUDA_API_ERRCHK( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CUDA_KERNEL_ERRCHK() __cudaCheckError( __FILE__, __LINE__ )
#define CUDA_KERNEL_ERRCHK_WSYNC() __cudaCheckError( __FILE__, __LINE__, true )
#define CUDA_KERNEL_ERRCHK_WABORT() __cudaCheckError( __FILE__, __LINE__, false, true )
#define CUDA_KERNEL_ERRCHK_WSYNC_WABORT() __cudaCheckError( __FILE__, __LINE__, true, true )

inline bool __cudaSafeCall(cudaError err, const char* file, const int line)
{
	if (cudaSuccess != err)
	{
		std::cerr << file << ":" << line << " : " << "CUDA API error: " << cudaGetErrorString(err) << std::endl;
		std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
		return true;
	}

	return false;
}

inline bool __cudaCheckError(const char* file, const int line, bool sync=false, bool abort=false)
{
	if (sync)
	{
		cudaError err = cudaDeviceSynchronize();
		if (cudaSuccess != err)
		{
			std::cerr << file << ":" << line << " : " << "CUDA Kernel error: " << cudaGetErrorString(err) << std::endl;
			std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
			if (abort)
				exit(1);
			return true;
		}
	}
	else
	{
		cudaError err = cudaGetLastError();
		if (cudaSuccess != err)
		{
			std::cerr << file << ":" << line << " : " << "CUDA Kernel error: " << cudaGetErrorString(err) << std::endl;
			std::cout << "CUDA error: " << cudaGetErrorString(err) << std::endl;
			return true;
		}
	}

	return false;
}

inline void __cudaDeviceErrorCheck()
{

	cudaError_t err = cudaDeviceSynchronize();
	std::cout << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << "\n";
}

#endif /* CUDAERRORCHECK_H */