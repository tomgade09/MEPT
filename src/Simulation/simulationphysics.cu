//Standard Library includes
#include <cmath>
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>

//CUDA includes
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

//Project specific includes
#include "physicalconstants.h"
#include "utils/loopmacros.h"
#include "Simulation/Simulation.h"
#include "ErrorHandling/cudaErrorCheck.h"

using std::cout;
using std::cerr;
using std::endl;
using std::setw;
using std::fixed;
using std::to_string;
using std::make_unique;
using std::stringstream;

//CUDA Variables
constexpr int  BLOCKSIZE{ 256 }; //Number of threads per block

//Commonly used values
extern const int SIMCHARSIZE{ 3 * sizeof(float) };

namespace physics
{
	__global__ void vperpMuConvert_d(float** dataToConvert, BModel** BModel, float mass, bool vperpToMu, int timeInd = 4)
	{//dataToConvert[0] = vpara, [1] = vperp, [2] = s, [3] = t_incident, [4] = t_escape
		unsigned int thdInd{ blockIdx.x * blockDim.x + threadIdx.x };
		
		if (dataToConvert[1][thdInd] != 0.0f)
		{
			float B_s{ (*BModel)->getBFieldAtS(dataToConvert[2][thdInd], dataToConvert[timeInd][thdInd]) };
			if (vperpToMu)
				dataToConvert[1][thdInd] = 0.5f * mass * dataToConvert[1][thdInd] * dataToConvert[1][thdInd] / B_s;
			else
				dataToConvert[1][thdInd] = sqrt(2 * dataToConvert[1][thdInd] * B_s / mass);
		}
	}

	__host__ void vperpMuConvert(const float vpara, float* vperpOrMu, const float s, const float t_convert, BModel* BModel, const float mass, const bool vperpToMu)
	{//dataToConvert[0] = vpara, [1] = vperp, [2] = s, [3] = t_incident, [4] = t_escape
		if (*vperpOrMu != 0.0f)
		{
			float B_s{ BModel->getBFieldAtS(s, t_convert) };
			if (vperpToMu)
				*vperpOrMu = 0.5f * mass * (*vperpOrMu) * (*vperpOrMu) / B_s;
			else
				*vperpOrMu = sqrt(2 * (*vperpOrMu) * B_s / mass);
		}
	}

	__device__ __host__ float accel1dCUDA(const float vs_RK, const float t_RK, const float* args, BModel** bmodel, EField* efield) //made to pass into 1D Fourth Order Runge Kutta code
	{//args array: [s_0, mu, q, m, simtime]
		float F_lor, F_mir, stmp;
		stmp = args[0] + vs_RK * t_RK; //ps_0 + vs_RK * t_RK

		//Mirror force
		F_mir = -args[1] * (*bmodel)->getGradBAtS(stmp, t_RK + args[4]); //-mu * gradB(pos, runge-kutta time + simtime)

		//Lorentz force - simply qE - v x B is taken care of by mu - results in kg.m/s^2 - to convert to Re equivalent - divide by Re
		F_lor = args[2] * efield->getEFieldAtS(stmp, t_RK + args[4]); //q * EFieldatS

		return (F_lor + F_mir) / args[3];
	}//returns an acceleration in the parallel direction to the B Field

	__device__ __host__ float foRungeKuttaCUDA(const float y_0, const float h, const float* funcArg, BModel** bmodel, EField* efield)
	{
		// dy / dt = f(t, y), y(t_0) = y_0
		// funcArgs are whatever you need to pass to the equation
		// args array: [s_0, mu, q, m, simtime]
		float k1, k2, k3, k4; float y{ y_0 }; float t_RK{ 0.0f };
		
		k1 = accel1dCUDA(y, t_RK, funcArg, bmodel, efield); //k1 = f(t_n, y_n), returns units of dy / dt

		t_RK = h / 2;
		y = y_0 + k1 * t_RK;
		k2 = accel1dCUDA(y, t_RK, funcArg, bmodel, efield); //k2 = f(t_n + h/2, y_n + h/2 * k1)

		y = y_0 + k2 * t_RK;
		k3 = accel1dCUDA(y, t_RK, funcArg, bmodel, efield); //k3 = f(t_n + h/2, y_n + h/2 * k2)

		t_RK = h;
		y = y_0 + k3 * t_RK;
		k4 = accel1dCUDA(y, t_RK, funcArg, bmodel, efield); //k4 = f(t_n + h, y_n + h k3)

		return (k1 + 2 * k2 + 2 * k3 + k4) * h / 6; //returns delta y, not dy / dt, not total y
	}

	__global__ void simActiveCheck(float** currData_d, bool* simDone)
	{
		//Answers the question: Are there no particles left in the simulation?
		//stores the value to simDone, which is defaulted to true, and flipped to false
		//only if t_escape is less than zero for at least one particle
		//(in that case, the sim is not completely done iterating)
		if (*simDone)
		{
			const float* t_escape_d{ currData_d[4] }; //const float* t_incident_d{ currData_d[3] }; //to be implemented

			unsigned int thdInd{ blockIdx.x * blockDim.x + threadIdx.x };

			if (t_escape_d[thdInd] >= 0.0f) //particle has escaped the sim
				return;
			else
				(*simDone) = false;
		}
	}

	__global__ void iterateParticle(float** currData_d, BModel** bmodel, EField* efield,
		const float simtime, const float dt, const float mass, const float charge, const float simmin, const float simmax, int len)
	{
		unsigned int thdInd{ blockIdx.x * blockDim.x + threadIdx.x };

		if (thdInd >= len) return;
		
		float* s_d{ currData_d[2] };
		const float* t_incident_d{ currData_d[3] };
		float* t_escape_d{ currData_d[4] };
		float* s0_d{ currData_d[5] };
		float* v0_d{ currData_d[6] };

		if (t_escape_d[thdInd] >= 0.0f) //particle has escaped, t_escape is >= 0 iff it has both entered previously and is currently outside the sim boundaries
			return;
		else if (t_incident_d[thdInd] > simtime) //particle hasn't "entered the sim" yet
			return;
		else if (s_d[thdInd] < simmin) //particle is out of sim to the bottom and t_escape not set yet
		{
			t_escape_d[thdInd] = simtime;
			return;
		}
		else if (s_d[thdInd] > simmax) //particle is out of sim to the top and t_escape not set yet
		{//maybe eventaully create new particle with initial characteristics on escape
			t_escape_d[thdInd] = simtime;
			return;
		}

		float* v_d{ currData_d[0] }; //these aren't needed if any of the above conditions is true
		const float* mu_d{ currData_d[1] };

		//args array: [ps_0, mu, q, m, simtime]
		const float args[]{ s_d[thdInd], mu_d[thdInd], charge, mass, simtime };

		//foRK (plus v0 in this case) gives v at the next time step (indicated vf in this note):
		//for downgoing (as an example), due to the mirror force, vf will be lower than v0 as the mirror force is acting in the opposite direction as v
		//along the path of the particle, ds, and so s will end up higher if we use ds = (vf * dt) than where it would realistically
		//if we use the ds = (v0 * dt), s will be lower down than where it would end up really (due to the fact that the mirror force acting along ds
		//will slow v down as the particle travels along ds), so I take the average of the two and it seems close enough s = (v0 + (v0 + dv)) / 2 * dt = v0 + dv/2 * dt
		//hence the /2 factor below - FYI, this was checked by the particle's energy (steady state, no E Field) remaining the same throughout the simulation
		s0_d[thdInd] = s_d[thdInd];
		v0_d[thdInd] = v_d[thdInd];
		v_d[thdInd] += foRungeKuttaCUDA(v_d[thdInd], dt, args, bmodel, efield);
		s_d[thdInd] += (v_d[thdInd] + v0_d[thdInd]) / 2 * dt;
	}

	__host__ void iterateParticle(float* vpara, float* mu, float* s, float* t_incident, float* t_escape, BModel* bmodel, EField* efield,
		const float simtime, const float dt, const float mass, const float charge, const float simmin, const float simmax)
	{
		if (simtime == 0.0f) { *t_escape = -1.0f; }
		if (*t_escape >= 0.0f) //see above function for description of conditions
			return;
		else if (*t_incident > simtime)
			return;
		else if (*s < simmin)
		{
			*t_escape = simtime;
			return;
		}
		else if (*s > simmax)
		{
			*t_escape = simtime;
			return;
		}

		const float args[]{ *s, *mu, charge, mass, simtime };

		float v_orig{ *vpara };
		*vpara += foRungeKuttaCUDA(*vpara, dt, args, &bmodel, efield);
		*s += (*vpara + v_orig) / 2 * dt;
	}
}

//Simulation member functions
void Simulation::initializeSimulation()
{
	if (BFieldModel_m == nullptr)
		throw logic_error("Simulation::initializeSimulation: no Earth Magnetic Field model specified");
	if (particles_m.size() == 0)
		throw logic_error("Simulation::initializeSimulation: no particles in simulation, sim cannot be initialized without particles");
	
	if (EFieldModel_m == nullptr) //make sure an EField (even if empty) exists
		EFieldModel_m = make_unique<EField>();
	
	if (tempSats_m.size() > 0)
	{//create satellites 
		LOOP_OVER_1D_ARRAY(tempSats_m.size(), createSatellite(tempSats_m.at(iii).get()));
	}
	else
		cerr << "Simulation::initializeSimulation: warning: no satellites created" << endl;

	initialized_m = true;
}

void Simulation::iterateSimulation(size_t numberOfIterations, size_t checkDoneEvery)
{//conducts iterative calculations of data previously copied to GPU - runs the data through the computeKernel
	using namespace physics;
	
	if (!initialized_m)
		throw logic_error("Simulation::iterateSimulation: sim not initialized with initializeSimulation()");
	
	//
	// Quick fix to show active GPU name.
	// This code is not checked for returned CUDA errors.  Eventually replace by reading for devices set to use by Environment. 
	// Maybe build this into printSimAttributes.
	//
	cudaDeviceProp gpuprop;
	int dev{ -1 };
	CUDA_API_ERRCHK(cudaGetDevice(&dev));
	CUDA_API_ERRCHK(cudaGetDeviceProperties(&gpuprop, dev));
	//
	//
	//
	//
	//

	printSimAttributes(numberOfIterations, checkDoneEvery, gpuprop.name);
	
	Log_m->createEntry("Start Iteration of Sim:  " + to_string(numberOfIterations));
	
	//convert particle vperp data to mu
	for (auto& part : particles_m)
		vperpMuConvert_d <<< part->getNumberOfParticles() / BLOCKSIZE, BLOCKSIZE >>> (part->getCurrDataGPUPtr(), BFieldModel_m->this_dev(), part->mass(), true);
	CUDA_KERNEL_ERRCHK_WSYNC();

	//Setup on GPU variable that checks to see if any threads still have a particle in sim and if not, end iterations early
	bool* simDone_d{ nullptr };
	CUDA_API_ERRCHK(cudaMalloc((void**)&simDone_d, sizeof(bool)));

	//Loop code
	size_t initEntry{ Log_m->createEntry("Iteration 1", false) };
	for (size_t cudaloopind = 0; cudaloopind < numberOfIterations; cudaloopind++)
	{	
		if (cudaloopind % checkDoneEvery == 0) { CUDA_API_ERRCHK(cudaMemset(simDone_d, true, sizeof(bool))); } //if it's going to be checked in this iter (every checkDoneEvery iterations), set to true
		
		for (auto part = particles_m.begin(); part < particles_m.end(); part++)
		{
			iterateParticle <<< (*part)->getNumberOfParticles() / BLOCKSIZE, BLOCKSIZE >>> ((*part)->getCurrDataGPUPtr(), BFieldModel_m->this_dev(), EFieldModel_m->this_dev(),
				simTime_m, dt_m, (*part)->mass(), (*part)->charge(), simMin_m, simMax_m, (*part)->getNumberOfParticles());
			
			//kernel will set boolean to false if at least one particle is still in sim
			if (cudaloopind % checkDoneEvery == 0)
				simActiveCheck <<< (*part)->getNumberOfParticles() / BLOCKSIZE, BLOCKSIZE >>> ((*part)->getCurrDataGPUPtr(), simDone_d);
		}

		CUDA_KERNEL_ERRCHK_WSYNC_WABORT(); //side effect: cudaDeviceSynchronize() needed for computeKernel to function properly, which this macro provides

		for (auto sat = satPartPairs_m.begin(); sat < satPartPairs_m.end(); sat++)
			(*sat)->satellite->iterateDetector(simTime_m, dt_m, BLOCKSIZE);
		
		if (cudaloopind % checkDoneEvery == 0)
		{
			{
				//if (cudaloopind == 0)
				//{
					//size_t free, total;
					//cudaMemGetInfo(&free, &total);
					//cout << "cuda mem: free: " << static_cast<float>(free)/1024.0f/1024.0f/1024.0f << "GB, total: " << static_cast<float>(total)/1024.0/1024.0/1024.0 << "GB\n";
				//}

				string loopStatus;
				stringstream out;

				out << setw(to_string(numberOfIterations).size()) << cudaloopind;
				loopStatus = out.str() + " / " + to_string(numberOfIterations) + "  |  Sim Time (s): ";
				out.str("");
				out.clear();

				out << setw(to_string((int)(numberOfIterations)*dt_m).size()) << fixed << simTime_m;
				loopStatus += out.str() + "  |  Real Time Elapsed (s): " + to_string(Log_m->timeElapsedSinceEntry_s(initEntry));

				Log_m->createEntry(loopStatus);
				cout << loopStatus << "\n";

				
			}

			bool done{ false };
			CUDA_API_ERRCHK(cudaMemcpy(&done, simDone_d, sizeof(bool), cudaMemcpyDeviceToHost));
			if (done)
			{
				cout << "All particles finished early.  Breaking loop.\n";
				break;
			}
		}

		incTime();
	}

	CUDA_API_ERRCHK(cudaFree(simDone_d));

	//Convert particle, satellite mu data to vperp
	for (auto part = particles_m.begin(); part < particles_m.end(); part++)
		vperpMuConvert_d <<< (*part)->getNumberOfParticles() / BLOCKSIZE, BLOCKSIZE >>> ((*part)->getCurrDataGPUPtr(), BFieldModel_m->this_dev(), (*part)->mass(), false); //nullptr will need to be changed if B ever becomes time dependent, would require loop to record when it stops tracking the particle

	for (auto sat = satPartPairs_m.begin(); sat < satPartPairs_m.end(); sat++)
		vperpMuConvert_d <<< (*sat)->particle->getNumberOfParticles() / BLOCKSIZE, BLOCKSIZE >>>  ((*sat)->satellite->get2DDataGPUPtr(), BFieldModel_m->this_dev(), (*sat)->particle->mass(), false, 3);

	//Copy data back to host
	LOOP_OVER_1D_ARRAY(getNumberOfParticleTypes(), particles_m.at(iii)->copyDataToHost());
	LOOP_OVER_1D_ARRAY(getNumberOfSatellites(), satellite(iii)->copyDataToHost());

	saveReady_m = true;
	saveDataToDisk();
	simTime_m = 0.0f;

	cout << "Total sim time: " << Log_m->timeElapsedTotal_s() << " s" << endl;

	Log_m->createEntry("Simulation::iterateSimulation: End Iteration of Sim:  " + to_string(numberOfIterations));
}

void Simulation::freeGPUMemory()
{//used to free the memory on the GPU that's no longer needed
	if (!initialized_m)
		throw logic_error("Simulation::freeGPUMemory: simulation not initialized with initializeSimulation()");

	Log_m->createEntry("Free GPU Memory");

	if (!dataOnGPU_m) { return; }

	CUDA_API_ERRCHK(cudaProfilerStop()); //For profiling with the CUDA bundle
}

void Simulation::setupGPU()
{
	cudaDeviceProp devProp;
	int gpuCount	 = 0;
	int computeTotal = 0;

	// Get total number of NVIDIA GPUs
	if (CUDA_API_ERRCHK(cudaGetDeviceCount(&gpuCount)))
	{
		cout << "Error: cannot get GPU count.  Only default GPU used.  Assuming at least one CUDA capable device.\n";
		computeSplit_m.push_back(1);
		gpuCount_m = 1;
		return;
	}

	// Store the count of the total number of GPUs
	gpuCount_m = gpuCount;

	// Iterate over each GPU and determine how much data it can handle
	for (int gpu = 0; gpu < gpuCount; gpu++)
	{
		// Get the GPU Speed
		CUDA_API_ERRCHK(cudaGetDeviceProperties(&devProp, gpu));

		// For now speed number of multiprocessors * clock speed
		// In future: either optimize for specific hardware create a more precise equation
		int compute = devProp.clockRate * devProp.multiProcessorCount;
		computeTotal += compute;
		computeSplit_m.push_back(compute);
	}
	// Iterate through computeSplit and get percent ratio work each device will get
	for (int i = 0; i < computeSplit_m.size(); ++i)
	{
		computeSplit_m.at(i) /= computeTotal;
	}
}
