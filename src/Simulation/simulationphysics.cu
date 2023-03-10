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
#include "utils/arrayUtilsGPU.h"
#include "utils/loopmacros.h"
#include "Simulation/Simulation.h"
#include "ErrorHandling/cudaErrorCheck.h"
#include "ErrorHandling/cudaDeviceMacros.h"

using std::cout;
using std::cerr;
using std::clog;
using std::endl;
using std::setw;
using std::fixed;
using std::to_string;
using std::make_unique;
using std::stringstream;

//CUDA Variables
__device__ Sat_d sats_d[32];       //allow for 32 satellites in the simulation - memory will probably run out long before this many are used, but will also check from host side to prevent overflows

//forward declaration for kernel in another file - requires rdc flag to be set
__device__ void satelliteDetector(Sat_d sat, const mpers v, const mpers v0, const meters s, const meters s0, const flPt_t mu, int ind, seconds simtime, seconds dt);

namespace physics
{
	__global__ void vperpMuConvert_d(flPt_t** dataToConvert, BModel** BModel, kg mass, bool vperpToMu, int len, int timeInd = 4)
	{//dataToConvert[0] = vpara, [1] = vperp, [2] = s, [3] = t_incident, [4] = t_escape
		unsigned int thdInd{ blockIdx.x * blockDim.x + threadIdx.x };

		if (thdInd >= len) return;
		
		if (dataToConvert[1][thdInd] != 0.0f)
		{
			tesla B_s{ (*BModel)->getBFieldAtS(dataToConvert[2][thdInd], dataToConvert[timeInd][thdInd]) };
			if (vperpToMu)
				dataToConvert[1][thdInd] = 0.5f * mass * dataToConvert[1][thdInd] * dataToConvert[1][thdInd] / B_s;
			else
				dataToConvert[1][thdInd] = sqrt(2.0f * dataToConvert[1][thdInd] * B_s / mass);
		}
	}

	__global__ void vperpMuConvert_d(Sat_d dataToConvert, BModel** BModel, kg mass, bool vperpToMu, int len)
	{//dataToConvert[0] = vpara, [1] = vperp, [2] = s, [3] = t_incident, [4] = t_escape
		unsigned int thdInd{ blockIdx.x * blockDim.x + threadIdx.x };

		if (thdInd >= len) return;

		if (dataToConvert.capture_d.mu[thdInd] != 0.0f)
		{
			tesla B_s{ (*BModel)->getBFieldAtS(dataToConvert.capture_d.s[thdInd], dataToConvert.capture_d.t_detect[thdInd]) };
			if (vperpToMu)
				dataToConvert.capture_d.mu[thdInd] = 0.5f * mass * dataToConvert.capture_d.mu[thdInd] * dataToConvert.capture_d.mu[thdInd] / B_s;
			else
				dataToConvert.capture_d.mu[thdInd] = sqrt(2.0f * dataToConvert.capture_d.mu[thdInd] * B_s / mass);
		}
	}

	__host__ void vperpMuConvert(const mpers vpara, flPt_t* vperpOrMu, const meters s, const seconds t_convert, BModel* BModel, const kg mass, const bool vperpToMu)
	{//dataToConvert[0] = vpara, [1] = vperp, [2] = s, [3] = t_incident, [4] = t_escape
		if (*vperpOrMu != 0.0f)
		{
			tesla B_s{ BModel->getBFieldAtS(s, t_convert) };
			if (vperpToMu)
				*vperpOrMu = 0.5f * mass * (*vperpOrMu) * (*vperpOrMu) / B_s;
			else
				*vperpOrMu = sqrt(2.0f * (*vperpOrMu) * B_s / mass);
		}
	}

	__device__ __host__ mpers2 accelEOM(const meters s_step, const seconds t_step, const flPt_t* args, BModel* bmodel, EField* efield) //made to pass into 1D Fourth Order Runge Kutta code
	{   //s_step, t_step, args are based on the overall position, and total time from the start of the simulation
		//args array: [s_0, mu, q, m, simtime]
		flPt_t F_lor, F_mir;

		//Mirror force
		F_mir = -args[1] * bmodel->getGradBAtS(s_step, t_step); //-mu * gradB(pos, runge-kutta time + simtime)

		//Lorentz force - simply qE - v x B is taken care of by mu - results in kg.m/s^2 - to convert to Re equivalent - divide by Re
		F_lor = args[2] * efield->getEFieldAtS(s_step, t_step); //q * EFieldatS

		return (F_lor + F_mir) / args[3];
	}//returns an acceleration in the parallel direction to the B Field

	__device__ __host__ void RK4_2ndOrd_ODE(const flPt_t s0, const flPt_t v0, const flPt_t t0, const flPt_t dt, const flPt_t* funcArgs, BModel* bmodel, EField* efield, flPt_t& sout, flPt_t& vout)
	{   // Referenced: https://www.compadre.org/PICUP/resources/Numerical-Integration/
		// Solves coupled system of ODEs:
		// ds / dt = v
		// dv / dt = F(s,v,t)/m, where F(s,v,t)/m is given by "accelEOM"
		// funcArgs are whatever you need to pass to the equation
		// args array: [s_0, mu, q, m, simtime]
		flPt_t k1v, k2v, k3v, k4v;  //k update values are in units of the value (so kNv is in units of v, etc)
		flPt_t k1s, k2s, k3s, k4s;

		auto eom = [&](flPt_t s_step, flPt_t t_step)
		{
			return accelEOM(s_step, t_step, funcArgs, bmodel, efield);
		};
		
		k1v = eom(s0, t0) * dt;                        //k1 = f(t, s), returns units of s
		k1s = v0 * dt;

		k2v = eom(s0 + k1s/2.0, t0 + dt/2.0) * dt;     //k2 = f(t_n + h/2, s_n + h/2 * k1)
		k2s = (v0 + k1v/2.0) * dt;

		k3v = eom(s0 + k2s/2.0, t0 + dt/2.0) * dt;     //k3 = f(t_n + h/2, s_n + h/2 * k2)
		k3s = (v0 + k2v/2.0) * dt;

		k4v = eom(s0 + k3s, t0 + dt) * dt;             //k4 = f(t_n + h, s_n + h k3)
		k4s = (v0 + k3v) * dt;

		sout = s0 + 1.0/6.0 * (k1s + 2.0*k2s + 2.0*k3s + k4s);
		vout = v0 + 1.0/6.0 * (k1v + 2.0*k2v + 2.0*k3v + k4v);
	}

	__global__ void simActiveCheck(flPt_t** currData_d, bool* simDone, int len)
	{
		//Answers the question: Are there no particles left in the simulation?
		//stores the value to simDone, which is defaulted to true, and flipped to false
		//only if t_escape is less than zero for at least one particle
		//(in that case, the sim is not completely done iterating)
		if (*simDone)
		{
			unsigned int thdInd{ blockIdx.x * blockDim.x + threadIdx.x };
			if (thdInd >= len) return;

			const seconds* t_escape_d{ currData_d[4] }; //const flPt_t* t_incident_d{ currData_d[3] }; //to be implemented
			
			if (t_escape_d[thdInd] >= 0.0f) //particle has escaped the sim
				return;
			else
				(*simDone) = false;
		}
	}

	__global__ void setSat(int ind, Sat_d sat_d)
	{
		//ZEROTH_THREAD_ONLY(
			if (ind >= 32) return;   //invalid memory location
			sats_d[ind] = sat_d;
		//);
	}

	__global__ void iterateParticle(flPt_t** currData_d, BModel** bmodel, EField* efield, const int numsats,
		const seconds simtime, const seconds dt, const kg mass, const coulomb charge, const meters simmin, const meters simmax, int len)
	{
		unsigned int idx{ blockIdx.x * blockDim.x + threadIdx.x };

		if (idx >= len) return;
		
		seconds t_inc{ currData_d[3][idx] };
		seconds t_esc{ currData_d[4][idx] };
		
		if (t_esc >= 0.0f) //particle has escaped, t_escape is >= 0 iff it has both entered previously and is currently outside the sim boundaries
			return;
		else if (t_inc > simtime) //particle hasn't "entered the sim" yet
			return;
		
		mpers  v0{ currData_d[0][idx] };
		meters s0{ currData_d[2][idx] };

		//args array: [ps_0, mu, q, m, simtime]
		const flPt_t args[]{ s0, currData_d[1][idx], charge, mass, simtime };

		//update particle position and velocity with RK4 solving coupled system of ODEs
		//RK4_2ndOrd_ODE(const flPt_t s0, const flPt_t v0, const flPt_t t0, const flPt_t dt, const flPt_t* funcArgs, BModel* bmodel, EField* efield, flPt_t& sout, flPt_t& vout)
		
		mpers v;
		meters s;
		RK4_2ndOrd_ODE(s0, v0, simtime, dt, args, *bmodel, efield, s, v);
		currData_d[0][idx] = v;
		currData_d[2][idx] = s;

		for (int sat = 0; sat < numsats; sat++)  //no divergence here - number of satellites is set before sim start
			satelliteDetector(sats_d[sat], v, v0, s, s0, args[1], idx, simtime, dt);
		
		if (s < simmin || s > simmax) //particle is out of sim to the bottom or the top so set t_escape
			currData_d[4][idx] = simtime;
	}

	__host__ void iterateParticle(mpers* vpara, flPt_t* mu, meters* s, seconds* t_incident, seconds* t_escape, BModel* bmodel, EField* efield,
		const seconds simtime, const seconds dt, const kg mass, const coulomb charge, const meters simmin, const meters simmax)
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

		const flPt_t args[]{ *s, *mu, charge, mass, simtime };

		RK4_2ndOrd_ODE(args[0], *vpara, simtime, dt, args, bmodel, efield, *s, *vpara);
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
	{
		EFieldModel_m = make_unique<EField>();
	}
	
	if (satellites_m.size() > 32)
		clog << "Simulation::initializeSimulation: Warning: more than 32 satellites specified.  Only 32 will be used\n";

	if (satellites_m.size() > 0)
	{//create satellites
		for (int i = 0; i < (int)satellites_m.size(); i++)
		{
			if (i < 32)
			{
				Satellite* s{ satellites_m.at(i).get() };
				for (int dev = 0; dev < gpuCount_m; dev++)
				{
					utils::GPU::setDev(dev);
					physics::setSat <<< 1, 1 >>> (i, s->getSat_d(dev));
				}
			}
		}
	}
	else
		cerr << "Simulation::initializeSimulation: Warning: no satellites created" << endl;
	
	initialized_m = true;
}

void Simulation::iterateSimulation(size_t numberOfIterations, size_t checkDoneEvery)
{//conducts iterative calculations of data previously copied to GPU - runs the data through the computeKernel
	using namespace physics;
	
	if (!initialized_m)
		throw logic_error("Simulation::iterateSimulation: sim not initialized with initializeSimulation()");

	printSimAttributes(numberOfIterations, checkDoneEvery);
	
	Log_m->createEntry("Start Iteration of Sim:  " + to_string(numberOfIterations) + " iterations");
	Log_m->createEntry("Blocksize:  " + to_string(BLOCKSIZE));
	
	//convert particle vperp data to mu
	vector<vector<size_t>> grid(gpuCount_m); //save grid size so it doesn't have to be recomputed every time
	for (int dev = 0; dev < gpuCount_m; dev++)
	{
		cudaSetDevice(dev);
		for (auto& part : particles_m)
		{ 
			grid.at(dev).push_back(part->getNumParticlesPerGPU(dev) / BLOCKSIZE);

			Log_m->createEntry("GPU " + to_string(dev) + ": Number of blocks: " + to_string(grid.at(dev).back()));

			vperpMuConvert_d <<< grid.at(dev).back(), BLOCKSIZE >>> (part->getCurrDataGPUPtr(dev),
				BFieldModel_m->this_dev(dev), part->mass(), true, part->getNumParticlesPerGPU(dev));
		}
		CUDA_KERNEL_ERRCHK_WSYNC();
	}

	//Setup on GPU variable that checks to see if any threads still have a particle in sim and if not, end iterations early
	vector<bool*> simDone_d;  //setup is done in loop below

	//Create pointers to device bools
	for (int dev = 0; dev < gpuCount_m; dev++)
	{
		CUDA_API_ERRCHK(cudaSetDevice(dev));
		simDone_d.push_back(nullptr);
		CUDA_API_ERRCHK(cudaMalloc((void**)&(simDone_d.at(dev)), sizeof(bool)));
	}

	//Loop code
	size_t initEntry{ Log_m->createEntry("Iteration 1", false) };
	
	for (size_t cudaloopind = 0; cudaloopind < numberOfIterations; cudaloopind++)
	{
		for (int dev = 0; dev < gpuCount_m; dev++) //iterate over devices - everything is non-blocking / async in loop
		{
			CUDA_API_ERRCHK(cudaSetDevice((int)dev));

			for (size_t part = 0; part < particles_m.size(); part++)
			{
				Particles* p{ particles_m.at(part).get() };
				iterateParticle <<< grid.at(dev).at(part), BLOCKSIZE >>> (p->getCurrDataGPUPtr(dev), BFieldModel_m->this_dev(dev), EFieldModel_m->this_dev(dev), 
					(int)satellites_m.size(), simTime_m, dt_m, p->mass(), p->charge(), simMin_m, simMax_m, p->getNumParticlesPerGPU(dev));
			}
		}

		if (cudaloopind % checkDoneEvery == 0)
		{
			for (int dev = 0; dev < gpuCount_m; dev++)
			{   //synchronize all devices
				CUDA_API_ERRCHK(cudaSetDevice((int)dev));
				CUDA_KERNEL_ERRCHK_WSYNC_WABORT();   //side effect: cudaDeviceSynchronize() needed for computeKernel to function properly, which this macro provides
				CUDA_API_ERRCHK(cudaMemset(simDone_d.at(dev), true, sizeof(bool))); //if it's going to be checked in this iter (every checkDoneEvery iterations), set to true
				for (size_t part = 0; part < particles_m.size(); part++)
				{   //check if every Particles instance has completely escaped the simulation
					Particles* p{ particles_m.at(0).get() };
					simActiveCheck <<< grid.at(dev).at(part), BLOCKSIZE >>> (p->getCurrDataGPUPtr(dev), simDone_d.at(dev), p->getNumParticlesPerGPU(dev));
				}
				CUDA_KERNEL_ERRCHK_WSYNC_WABORT();
			}

			string loopStatus;
			stringstream out;

			out << setw(to_string(numberOfIterations).size()) << cudaloopind;
			loopStatus = out.str() + " / " + to_string(numberOfIterations) + "  |  Sim Time (s): ";
			out.str("");
			out.clear();

			out << setw(to_string((int)(numberOfIterations)*dt_m).size()) << fixed << simTime_m;
			loopStatus += out.str() + "  |  Real Time Elapsed (s): " + to_string(Log_m->timeElapsedSinceEntry_s(initEntry));

			clog << loopStatus << "\n";
			cout << loopStatus << "\n";
			
			bool done{ true };
			for (int dev = 0; dev < gpuCount_m; dev++)
			{
				bool tmp{ false };
				CUDA_API_ERRCHK(cudaMemcpy(&tmp, simDone_d.at(dev), sizeof(bool), cudaMemcpyDeviceToHost));
				if (!tmp) done = false;
			}
			if (done)
			{
				cout << "All particles finished early.  Breaking loop.\n";
				break;
			}
		}

		incTime();
	}

	for (int dev = 0; dev < gpuCount_m; dev++)
		CUDA_API_ERRCHK(cudaFree(simDone_d.at(dev)));

	//Convert particle, satellite mu data to vperp
	for (int dev = 0; dev < gpuCount_m; dev++)
	{
		CUDA_API_ERRCHK(cudaSetDevice(dev));
		for (int p = 0; p < particles_m.size(); p++)
		{
			Particles* part{ particles_m.at(p).get() };
			vperpMuConvert_d <<< grid.at(dev).at(p), BLOCKSIZE >>> (part->getCurrDataGPUPtr(dev), BFieldModel_m->this_dev(dev),
				part->mass(), false, part->getNumParticlesPerGPU(dev));
		}
	}

	for (int dev = 0; dev < gpuCount_m; dev++)
	{
		CUDA_API_ERRCHK(cudaSetDevice(dev));
		for (int s = 0; s < satellites_m.size(); s++)
		{
			Satellite* sat{ satellites_m.at(s).get() };
			vperpMuConvert_d <<< sat->getNumParticlesPerGPU(dev) / BLOCKSIZE, BLOCKSIZE >>> (sat->getSat_d(dev), 
				BFieldModel_m->this_dev(dev), MASS_ELECTRON, false, sat->getNumParticlesPerGPU(dev));
			//If non-electrons are used, this ^^^^^ needs to be updated with the particle mass
		}
	}

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
	int gpuCount = 0;

	// Get total number of NVIDIA GPUs
	if (CUDA_API_ERRCHK(cudaGetDeviceCount(&gpuCount)))
	{
		cerr << "Cannot get GPU count.  Only default GPU used.  Assuming at least one CUDA capable device.\n";
		gpuCount_m = 1;
		return;
	}

	gpuCount_m = gpuCount;

	Log_m->createEntry("Simulation::setupGPU: Found " + to_string(gpuCount) + " GPU(s).");
}
