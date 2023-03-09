#ifndef EMODEL_EFIELD_H
#define EMODEL_EFIELD_H

#include <memory>
#include <vector>
#include <string>

//CUDA includes
#include "cuda_runtime.h"

//Project includes
#include "dlldefines.h"
#include "utils/unitsTypedefs.h"
#include "utils/serializationHelpers.h"

using std::string;
using std::vector;
using std::unique_ptr;

class EModel //inherit from this class
{
public:
	enum class Type
	{
		QSPS,
		AlfvenLUT,
		Other
	};

	Type type_m{ Type::Other };

protected:
	#ifndef __CUDA_ARCH__
	vector<EModel**> this_d; //not really used on device
	#else
	EModel** this_d{ nullptr };
	#endif

	__host__            virtual void setupEnvironment() = 0; //define this function in derived classes to assign a pointer to that function's B Field code to the location indicated by BModelFcnPtr_d and gradBFcnPtr_d
	__host__            virtual void deleteEnvironment() = 0;
	__host__            virtual void deserialize(ifstream& in) = 0;

	__host__ __device__ EModel(Type type);

	friend class EField;

public:
	__host__ __device__ ~EModel();
	__host__ __device__ EModel(const EModel&) = delete;

	__host__ __device__ virtual Vperm getEFieldAtS(const meters s, const seconds t) const = 0;

	__host__            EModel** this_dev(int GPUind) const;
	__host__            string name() const;

	__host__            virtual vector<flPt_t> getAllAttributes() const = 0;
	__host__            virtual void serialize(ofstream& out) const = 0;
};

#endif /* !EFIELD_EMODEL_H */
