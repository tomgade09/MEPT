#ifndef EFIELD_H
#define EFIELD_H

#include "EField/EModel.h"

class EField final //not meant to be inherited from
{
private:
	EField* this_d{ nullptr }; //pointer to EField instance on GPU

	//the below vector references need these dev exclusion blocks or there
	//is a cudaIllegalMemoryAccess error that's really tricky to track down
#ifndef __CUDA_ARCH__ //host code
	vector<unique_ptr<EModel>> emodels_m; //EModels pointers on host
#endif /* !__CUDA_ARCH__ */

//GPU container of EModels variables
	EModel** emodels_d{ nullptr }; //host: holds ptr to on GPU array, used to increase size, device: holds ptr to on GPU array, used to access elements
	int capacity_d{ 5 };           //denotes size and capacity of E element array on device
	int size_d{ 0 };
	//End container variables

	int qsps_m{ 0 };
	int alut_m{ 0 };

	bool useGPU_m{ true };

	__host__ void setupEnvironment();
	__host__ void deleteEnvironment();

	__host__ void deserialize(ifstream& in);

public:
	__host__ __device__ EField();
	__host__            EField(ifstream& in);
	__host__ __device__ ~EField();

	__host__            void add(unique_ptr<EModel> emodel);
	__device__          void add(EModel* emodel);

	__host__ __device__ int capacity() const;
	__host__ __device__ int size() const;
	__device__          void capacity(int cap);
	__device__          EModel** emodelArray_d() const;
	__device__          void emodelArray_d(EModel** emodelPtrArray);
	
	__host__ __device__ Vperm   getEFieldAtS(const meters s, const seconds t) const;
	__host__            EField* this_dev() const;
	
	__host__            EModel* emodel(int ind) const;
	//__host__            string getElementNames() const;
	__host__            int qspsCount() const;
	__host__            int alutCount() const;
	__host__            bool timeDependent() const;

	__host__            void serialize(ofstream& out) const;
};

#endif /* EFIELD_H */
