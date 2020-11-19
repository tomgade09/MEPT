#include "EField/EModel.h"
#include <stdexcept>

using std::runtime_error;

__host__ __device__ EModel::EModel(Type type) : type_m{ type }
{

}

__host__ __device__ EModel::~EModel()
{

}

__host__ EModel** EModel::this_dev() const
{
	return this_d;
}

__host__ string EModel::name() const
{
    if (type_m == Type::QSPS) return "QSPS";
    else if (type_m == Type::AlfvenLUT) return "AlfvenBLUT";
    else if (type_m == Type::Other) return "Unknown";
    else throw runtime_error("EModel::name: unknown type");
}