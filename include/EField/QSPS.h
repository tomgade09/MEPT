#ifndef QSPS_EFIELD_H
#define QSPS_EFIELD_H

#include "EField/EModel.h"

using std::vector;

class QSPS : public EModel
{
protected:
	#ifndef __CUDA_ARCH__ //host code
	vector<meters> altMin_m;
	vector<meters> altMax_m;
	vector<Vperm> magnitude_m;
	#endif /* !__CUDA_ARCH__ */
	
	int numRegions_m{ 0 };

	meters* altMin_d; //on host this stores the pointer to the data on GPU, on GPU ditto
	meters* altMax_d;
	Vperm*  magnitude_d;

	bool useGPU_m{ true };

	__host__ void setupEnvironment()  override;
	__host__ void deleteEnvironment() override;
	__host__ void deserialize(ifstream& in) override;

public:
	__host__ QSPS(meters altMin, meters altMax, Vperm magnitude, int stepUpRegions = 0);
	__host__ QSPS(ifstream& in);
	__device__ QSPS(meters* altMin, meters* altMax, Vperm* magnitude, int numRegions);
	__host__ __device__ ~QSPS();

	__host__ __device__ Vperm getEFieldAtS(const meters s, const seconds t) const override;

	__host__ const vector<meters>& altMin() const;
	__host__ const vector<meters>& altMax() const;
	__host__ const vector<Vperm>&  magnitude() const;
	
	__host__ vector<double> getAllAttributes() const override;
	__host__ void serialize(ofstream& out) const override;
};

#endif /* !QSPS_EFIELD_H */