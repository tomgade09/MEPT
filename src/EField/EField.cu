#include "EField/EField.h"
#include "EField/QSPS.h"
//#include "EField/AlfvenLUT.h"

#include <sstream>

//CUDA includes
#include "device_launch_parameters.h"
#include "ErrorHandling/cudaErrorCheck.h"
#include "ErrorHandling/cudaDeviceMacros.h"

using std::string;
using std::bad_cast;
using std::to_string;
using std::stringstream;
using std::runtime_error;

//device global kernels
namespace EField_d
{
	__global__ void setupEnvironment_d(EField* efield, EModel** emodelPtrArray)
	{
		ZEROTH_THREAD_ONLY(
			(*efield) = EField();
			efield->emodelArray_d(emodelPtrArray);
		);
	}

	__global__ void deleteEnvironment_d(EField* efield)
	{
		ZEROTH_THREAD_ONLY(delete efield);
	}

	__global__ void add_d(EField* efield, EModel** emodel) //move these two functions into EField class as device functions
	{
		ZEROTH_THREAD_ONLY(efield->add(*emodel));
	}

	__global__ void increaseCapacity_d(EField* efield, EModel** newArray, int capacity)
	{
		ZEROTH_THREAD_ONLY(
			EModel** oldArray{ efield->emodelArray_d() };

			for (int elem = 0; elem < efield->size(); elem++)
				newArray[elem] = oldArray[elem];

			efield->capacity(capacity);
			efield->emodelArray_d(newArray); //still retaining the pointer to this memory on host, so no big deal if it's lost here
		);
	}
}


//EField ctor, dtor
__host__ __device__ EField::EField()/*bool useGPU) : useGPU_m{ useGPU }*/
{
	#ifndef __CUDA_ARCH__ //host code
	if(useGPU_m) setupEnvironment();
	#endif /* !__CUDA_ARCH__ */
}

__host__ EField::EField(ifstream& in)
{
	deserialize(in);
	if(useGPU_m) setupEnvironment();
}

__host__ __device__ EField::~EField()
{
	#ifndef __CUDA_ARCH__ //host code
	if(useGPU_m) deleteEnvironment();

	for (auto& model : emodels_m)
		model.release();
	#endif /* !__CUDA_ARCH__ */
}

//EField functions
/*__host__ string EField::getElementNames() const
{
	stringstream out;
	
	#ifndef __CUDA_ARCH__ //host code
	for (const auto& emodel : emodels_m)
	{
		if (emodel->type_m == EModel::Type::QSPS) out << "QSPS";
		//else if (checkIfDerived<EModel, AlfvenLUT>(element(elem))) out << "AlfvenLUT";
		else out << "Unknown";
		out << ", ";
	}
	#endif *//* !__CUDA_ARCH__ *//*
	
	return out.str();
}*/

__host__ int EField::qspsCount() const
{
	return qsps_m;
}

__host__ int EField::alutCount() const
{
	return alut_m;
}

__host__ bool EField::timeDependent() const
{
	return alut_m > 0;
}

__host__ void EField::setupEnvironment()
{
	CUDA_API_ERRCHK(cudaMalloc((void**)&this_d, sizeof(EField)));                  //allocate memory for EField
	CUDA_API_ERRCHK(cudaMalloc((void**)&emodels_d, sizeof(EModel*) * capacity_d)); //allocate memory for EModel* array
	CUDA_API_ERRCHK(cudaMemset(emodels_d, 0, sizeof(EModel*) * capacity_d));       //clear memory
	
	EField_d::setupEnvironment_d <<< 1, 1 >>> (this_d, emodels_d);
	CUDA_KERNEL_ERRCHK_WSYNC();
}

__host__ void EField::deleteEnvironment()
{
	CUDA_API_ERRCHK(cudaFree(this_d));
	CUDA_API_ERRCHK(cudaFree(emodels_d));
}

#ifndef __CUDA_ARCH__ //host code
__host__ EModel* EField::emodel(int ind) const
{
	return emodels_m.at(ind).get();
}

__host__ void EField::add(unique_ptr<EModel> emodel)
{
	if (capacity_d == size_d && useGPU_m)
	{
		EModel** oldArray{ emodels_d }; //retain so we can cudaFree at the end
		capacity_d += 5;
		
		CUDA_API_ERRCHK(cudaMalloc((void**)&emodels_d, sizeof(EModel*) * capacity_d)); //create new array that is 5 larger in capacity than the previous
		CUDA_API_ERRCHK(cudaMemset(emodels_d, 0, sizeof(EModel*) * capacity_d));

		EField_d::increaseCapacity_d <<< 1, 1 >>> (this_d, emodels_d, capacity_d);
		CUDA_KERNEL_ERRCHK();

		CUDA_API_ERRCHK(cudaFree(oldArray));
	}
	
	//add elem to dev
	if (useGPU_m)
	{
		EField_d::add_d <<< 1, 1 >>> (this_d, emodel->this_dev());
		CUDA_KERNEL_ERRCHK_WSYNC();
		size_d++;
	}

	auto checkIfDerived = [](const EModel* model)
	{
		try
		{//uses the fact that dynamic_cast<derived>(base) throws if base is not of type derived
			const QSPS* der = dynamic_cast<const QSPS*>(model);
			if (der == nullptr) return false;
			return true;
		}
		catch (bad_cast & e)
		{
			return false;
		}
	};

	//add elem to host
	if (checkIfDerived(emodel.get())) qsps_m++;
	//need to add alut_m++ condition, maybe make lambda a template?
	emodels_m.push_back(move(emodel));
}
#endif /* !__CUDA_ARCH__ */

__device__ void EField::add(EModel* emodel)
{
	emodels_d[size_d] = emodel;
	size_d++;
}

__host__ __device__ int EField::capacity() const
{
	return capacity_d;
}

__host__ __device__ int EField::size() const
{
	return size_d;
}

__device__ void EField::capacity(int cap)
{
	capacity_d = cap;
}

__device__ EModel** EField::emodelArray_d() const
{
	return emodels_d;
}

__device__ void EField::emodelArray_d(EModel** emodels)
{
	emodels_d = emodels;
}
	
__host__ EField* EField::this_dev() const
{
	return this_d;
}

__host__ __device__ Vperm EField::getEFieldAtS(const meters s, const seconds t) const
{
	tesla ret{ 0.0 };

	#ifndef __CUDA_ARCH__ //host code
	for (auto& elem : emodels_m) //vector of unique_ptr<EModel>'s
		ret += elem->getEFieldAtS(s, t);
	#else //device code
	for (int elem = 0; elem < size_d; elem++) //c-style array of EModel*'s
		ret += (emodels_d[elem])->getEFieldAtS(s, t);
	#endif /* !__CUDA_ARCH__ */

	return ret;
}