#include "EField/EModel.h"
#include "utils/arrayUtilsGPU.h"
#include <stdexcept>

using std::runtime_error;

__host__ __device__ EModel::EModel(Type type) : type_m{ type }
{
    #ifndef __CUDA_ARCH__
    this_d.resize(utils::GPU::getDeviceCount());
    #endif
}

__host__ __device__ EModel::~EModel()
{

}

__host__ EModel** EModel::this_dev(int GPUind) const
{
    #ifndef __CUDA_ARCH__
	return this_d.at(GPUind);
    #else
    return this_d;
    #endif
}

__host__ string EModel::name() const
{
    if (type_m == Type::QSPS) return "QSPS";
    else if (type_m == Type::AlfvenLUT) return "AlfvenBLUT";
    else if (type_m == Type::Other) return "Unknown";
    else throw runtime_error("EModel::name: unknown type");
}