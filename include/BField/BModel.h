#ifndef BFIELD_BMODEL_H
#define BFIELD_BMODEL_H

#include <string>

//CUDA includes
#include "cuda_runtime.h"

#include "utils/unitsTypedefs.h"

using std::string;
using std::ifstream;
using std::ofstream;

class BModel
{
public:
	enum class Type
	{
		DipoleB,
		DipoleBLUT,
		Other
	};

protected:
	#ifndef __CUDA_ARCH__
	vector<BModel**> this_d; //pointer to device-side instance
	#else
	BModel** this_d{ nullptr };
	#endif

	Type type_m{ Type::Other };

	__host__            virtual void setupEnvironment() = 0; //define this function in derived classes to assign a pointer to that function's B Field code to the location indicated by BModelFcnPtr_d and gradBFcnPtr_d
	__host__            virtual void deleteEnvironment() = 0;
	__host__            virtual void deserialize(ifstream& in) = 0;

	__host__ __device__ BModel(Type type);

public:
	__host__ __device__ ~BModel();
	__host__ __device__ BModel(const BModel&) = delete;

	__host__ __device__ virtual tesla  getBFieldAtS(const meters s, const seconds t) const = 0;
	__host__ __device__ virtual flPt_t getGradBAtS (const meters s, const seconds t) const = 0;
	__host__ __device__ virtual meters getSAtAlt(const meters alt_fromRe) const = 0;

	__host__            virtual meters ILAT() const = 0;

	__host__            BModel** this_dev(int GPUind) const; //once returned, have to cast it to the appropriate type
	__host__            string name() const;
	__host__            Type   type() const;

	__host__            virtual vector<flPt_t> getAllAttributes() const = 0;
	__host__            virtual void serialize(ofstream& out) const = 0;
};

#endif