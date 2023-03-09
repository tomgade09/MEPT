#include "BField/BModel.h"
#include "utils/arrayUtilsGPU.h"
#include <stdexcept>

using std::runtime_error;

__host__ __device__ BModel::BModel(Type type) : type_m{ type }
{
    #ifndef __CUDA_ARCH__
    this_d.resize(utils::GPU::getDeviceCount());
    #endif
}

__host__ __device__ BModel::~BModel()
{

}

__host__ BModel** BModel::this_dev(int GPUind) const
{
    #ifndef __CUDA_ARCH__
    return this_d.at(GPUind);
    #else
    return this_d;
    #endif
}

__host__ string BModel::name() const
{
    if (type_m == Type::DipoleB) return "DipoleB";
    else if (type_m == Type::DipoleBLUT) return "DipoleBLUT";
    else if (type_m == Type::Other) return "Unknown";
    else throw runtime_error("BModel::name: unknown type");
}

__host__ BModel::Type BModel::type() const
{
    return type_m;
}