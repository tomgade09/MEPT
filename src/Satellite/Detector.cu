//CUDA includes
#include "cuda_runtime.h"
#include "Satellite/Satellite.h"

__device__ void satelliteDetector(Sat_d sat, float** data_d, const float v, const float v0, const float s, const float s0,
	int ind, float simtime, float dt)
{
	float* detected_t_d{ sat.capture_d[3] };    //do this first before creating a bunch of pointers

	if (simtime == 0.0f) //not sure I fully like this, but it works
	{
		float* detected_ind_d{ sat.capture_d[4] };
		detected_t_d[ind] = -1.0f;
		detected_ind_d[ind] = -1.0f;
	}

	//guard to prevent unnecessary variable creation, if condition checks
	if (detected_t_d[ind] > -0.1f) return; //if the slot in detected_t[thdInd] is filled (gt or equal to 0), return

	if (//no detected particle is in the data array at the thread's index already AND
		((!sat.upward) && (s >= sat.altitude) && (s0 < sat.altitude)) //detector is facing down and particle crosses altitude in dt
		|| //OR
		(( sat.upward) && (s <= sat.altitude) && (s0 > sat.altitude))  //detector is facing up and particle crosses altitude in dt
		)
	{
		const float* mu_d{ data_d[1] };

		float* detected_v_d{ sat.capture_d[0] }; float* detected_mu_d{ sat.capture_d[1] };
		float* detected_s_d{ sat.capture_d[2] }; float* detected_ind_d{ sat.capture_d[4] };
		
		const float s_ratio{ (sat.altitude - s0) / (s - s0) };
		const float interpol_v_d{ v0 + (v - v0) * s_ratio }; //interpolates v back to where the particle was detected
		const float interpol_t_d{ simtime + dt * s_ratio };
		
		detected_v_d[ind] = interpol_v_d;
		detected_mu_d[ind] = mu_d[ind];
		detected_s_d[ind] = sat.altitude;
		detected_t_d[ind] = interpol_t_d;
		detected_ind_d[ind] = static_cast<float>(ind);
	}//particle not removed from sim
}