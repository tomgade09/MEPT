// CUDAGetDevices.cu
// Gets number of NVIDIA GPU devices and lists relevant properties
// Created for EE 5351 Applied Parallel Programming, not for credit
// Tom Gade, 16 Sep 20

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

using std::cout;

inline void checkForCUDAError(cudaError_t ce, int lineNum)
{
    if (ce != cudaSuccess)
    {
        cout << "CUDA Error on line " << lineNum << ": " << cudaGetErrorName(ce) << ": " << cudaGetErrorString(ce) << "\nExiting.\n";
        exit(1);
    }
}

int main()
{
    int deviceCount{ -1 };
    
    checkForCUDAError(cudaGetDeviceCount(&deviceCount), __LINE__);

    cout << "Number of Devices: " << deviceCount << "\n\n\n";
	
    if (deviceCount < 1)
    {
        cout << "Error: no GPUs detected.  Exiting.\n";
        exit(1);
    }
	
	for (int i = 0; i < deviceCount; i++)
	{
        cudaDeviceProp prop;
        checkForCUDAError(cudaGetDeviceProperties(&prop, i), __LINE__);

        cout << "================================================================\n";
        cout << "GPU Name:        " << prop.name << "\n";
        cout << "Tot Global Mem:  " << prop.totalGlobalMem/1024/1024 << " MB\n";
        cout << "Shared per Blk:  " << prop.sharedMemPerBlock/1024 << " KB\n";
        cout << "Regs per Block:  " << prop.regsPerBlock << "\n";
        cout << "Warp size:       " << prop.warpSize << " threads\n";
        cout << "Memory Pitch:    " << prop.memPitch/1024/1024 << " MB\n";
        cout << "Max Thd per Blk: " << prop.maxThreadsPerBlock << " threads\n";
        cout << "Clock Rate:      " << prop.clockRate/1024 << " MHz\n";
        cout << "Tot Const Mem:   " << prop.totalConstMem/1024 << " KB\n";
        cout << "Compute Capab:   " << prop.major << "." << prop.minor << "\n";
        cout << "Multiproc Count: " << prop.multiProcessorCount << "\n";
        cout << "Integrated GPU?: " << ((prop.integrated == true) ? ("True") : ("False")) << "\n\n\n";
	}
	
    return 0;
}