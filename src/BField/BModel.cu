#include "BField/BModel.h"
#include <stdexcept>

using std::runtime_error;

__host__ __device__ BModel::BModel(Type type) : type_m{ type }
{

}
#include <iostream>
__host__ __device__ BModel::~BModel()
{

}

__host__ BModel** BModel::this_dev() const
{
    return this_d;
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