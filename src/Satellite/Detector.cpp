#include <omp.h>
#include <thread>

#include "Satellite/Satellite.h"
#include "ErrorHandling/simExceptionMacros.h"

using std::runtime_error;

void Satellite::iterateDetectorCPU(const vector<vector<float>>& particleData, seconds simtime, seconds dt)
{
	if (data_m.at(0).size() != particleData.at(0).size())
		throw runtime_error("Satellite::iterateDetectorCPU: data_m is improperly formed for Satellite named " + name_m);

	#pragma omp parallel for
	for (int partind = 0; partind < (int)particleData.at(0).size(); partind++)
	{
		if (simtime == 0.0f)
		{
			data_m.at(3).at(partind) = -1.0f; //t_esc
			data_m.at(4).at(partind) = -1.0f; //index
		}

		if (data_m.at(3).at(partind) > -0.1f)
			continue;

		float s_minus_vdt{ particleData.at(2).at(partind) - particleData.at(0).at(partind) * dt };

		if (
			//(detected.at(3).at(partind) < -0.1f) &&
			((!upwardFacing_m) && (particleData.at(2).at(partind) >= altitude_m) && (s_minus_vdt < altitude_m))
			||
			((upwardFacing_m)  && (particleData.at(2).at(partind) <= altitude_m) && (s_minus_vdt > altitude_m))
			)
		{
			data_m.at(0).at(partind) = particleData.at(0).at(partind); //vpara
			data_m.at(1).at(partind) = particleData.at(1).at(partind); //mu
			data_m.at(2).at(partind) = particleData.at(2).at(partind); //s
			data_m.at(3).at(partind) = simtime;                        //t_esc
			data_m.at(4).at(partind) = (float)partind;                //index
		}
	}
}