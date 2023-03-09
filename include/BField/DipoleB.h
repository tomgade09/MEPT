#ifndef DIPOLEB_BFIELD_H
#define DIPOLEB_BFIELD_H

#include "BField/BModel.h"
#include "physicalconstants.h"
#include "utils/unitsTypedefs.h"

class DipoleB : public BModel
{
protected:
	//Field simulation constants
	meters  L_m{ 0.0f };
	meters  L_norm_m{ 0.0f };
	meters  s_max_m{ 0.0f };

	//specified variables
	degrees ILAT_m{ 0.0f };
	meters  ds_m{ 0.0f };
	ratio   lambdaErrorTolerance_m{ 0.0f };

	bool    useGPU_m{ true };

	//protected functions
	__host__            void    setupEnvironment() override;
	__host__            void    deleteEnvironment() override;
	__host__            void    deserialize(ifstream& in) override;

	__host__ __device__ meters  getSAtLambda(const degrees lambda) const;
	__host__ __device__ degrees getLambdaAtS(const meters s) const;

public:
	__host__ __device__ DipoleB(degrees ILAT, ratio lambdaErrorTolerance = 1.0e-4, meters ds = RADIUS_EARTH / 1000.0, bool useGPU = true);
	__host__            DipoleB(ifstream& in);
	__host__ __device__ ~DipoleB();
	__host__ __device__ DipoleB(const DipoleB&) = delete;

	__host__            degrees ILAT()  const;

	__host__ __device__ tesla  getBFieldAtS(const meters s, const seconds t) const override;
	__host__ __device__ flPt_t getGradBAtS (const meters s, const seconds t) const override;
	__host__ __device__ meters getSAtAlt   (const meters alt_fromRe) const override;

	__host__            ratio  getErrTol() const;
	__host__            meters getds()     const;

	__host__            vector<flPt_t> getAllAttributes() const override;
	__host__            void serialize(ofstream& out) const override;
};

#endif