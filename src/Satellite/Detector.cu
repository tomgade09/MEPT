//CUDA includes
#include "cuda_runtime.h"
#include "Satellite/Satellite.h"


//satelliteDetector not associated with an instance of a "Satellite" object - it's a standalone kernel that operates on any Satellite's data that is passed in
__device__ void satelliteDetector(Sat_d sat, const mpers v, const mpers v0, const meters s, const meters s0, const flPt_t mu,
	int ind, seconds simtime, seconds dt)
{
	const SatDataPtrs& data{ sat.capture_d };
	if (simtime == 0.0f) //not sure I fully like this, but it works
		data.t_detect[ind] = -1.0f;

	//guard to prevent unnecessary variable creation, if condition checks
	if (data.t_detect[ind] > -0.1f) return; //if the slot in detected_t[thdInd] is filled (gt or equal to 0), return

	if (//no detected particle is in the data array at the thread's index already AND
		((!sat.upward) && (s >= sat.altitude) && (s0 < sat.altitude)) //detector is facing down and particle crosses altitude in dt
		|| //OR
		(( sat.upward) && (s <= sat.altitude) && (s0 > sat.altitude))  //detector is facing up and particle crosses altitude in dt
		)
	{		
		const ratio   s_ratio{ (sat.altitude - s0) / (s - s0) };
		const mpers   interpol_v_d{ v0 + (v - v0) * s_ratio }; //interpolates v back to where the particle was detected
		const seconds interpol_t_d{ simtime + dt * s_ratio };
		
		data.vpara[ind] = interpol_v_d;
		data.mu[ind] = mu;
		data.s[ind] = sat.altitude;
		data.t_detect[ind] = interpol_t_d;
	}//particle not removed from sim
}