#ifndef DIPOLEBLUT_BFIELD_H
#define DIPOLEBLUT_BFIELD_H

#include <vector>
#include "BField/BModel.h"
#include "physicalconstants.h"
#include "utils/unitsTypedefs.h"

class DipoleBLUT : public BModel
{
protected:
	//specified variables
	degrees ILAT_m{ 0.0 };
	meters  ds_msmt_m{ 0.0 };
	meters  ds_gradB_m{ 0.0 };

	#ifndef __CUDA_ARCH__ //host code
	std::vector<meters> altitude_m;
	std::vector<tesla>  magnitude_m;
	#endif /* !__CUDA_ARCH__ */

	//on device variables
	double* altitude_d{ nullptr }; //serves as device-side C-style vector equivalent, and on host, pointer to this on dev
	double* magnitude_d{ nullptr };
	
	meters  simMin_m{ 0.0 };
	meters  simMax_m{ 0.0 };
	int     numMsmts_m{ 0 }; //used instead of size() because vectors aren't implemented on device

	bool useGPU_m{ true };

	//protected functions
	__host__ void setupEnvironment() override;
	__host__ void deleteEnvironment() override;

	__host__ void deserialize(ifstream& in) override;

public:
	__host__ __device__ DipoleBLUT(degrees ILAT, meters simMin, meters simMax, meters ds_gradB, int numberOfMeasurements, bool useGPU = true);
	__host__            DipoleBLUT(ifstream& in);
	__host__ __device__ ~DipoleBLUT();
	__host__ __device__ DipoleBLUT(const DipoleBLUT&) = delete;

	__host__            degrees ILAT() const;

	__host__ __device__ tesla   getBFieldAtS(const meters s, const seconds t) const override;
	__host__ __device__ double  getGradBAtS(const meters s, const seconds t) const override;
	__host__ __device__ meters  getSAtAlt(const meters alt_fromRe) const override;

    __device__          void    setAltArray(double* altArray);
	__device__          void    setMagArray(double* magArray);

	__host__            ratio   getErrTol() const;
	__host__            meters  getds()     const;

	__host__            vector<double> getAllAttributes() const override;
	__host__            void           serialize(ofstream& out) const override;
};

#endif /* !DIPOLEBLUT_BMODEL_H */
