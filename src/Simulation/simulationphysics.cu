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

using std::cout;
using std::cerr;
using std::endl;
using std::setw;
using std::fixed;
using std::to_string;
using std::make_unique;
using std::stringstream;

//CUDA Variables
constexpr size_t BLOCKSIZE{ 256 }; //Number of threads per block

namespace physics
{
	__global__ void vperpMuConvert_d(float** dataToConvert, BModel** BModel, float mass, bool vperpToMu, int len, int timeInd = 4)
	{//dataToConvert[0] = vpara, [1] = vperp, [2] = s, [3] = t_incident, [4] = t_escape
		int thdInd{ blockIdx.x * blockDim.x + threadIdx.x };

		if (thdInd >= len) return;
		
		if (dataToConvert[1][thdInd] != 0.0f)
		{
			float B_s{ (*BModel)->getBFieldAtS(dataToConvert[2][thdInd], dataToConvert[timeInd][thdInd]) };
			if (vperpToMu)
				dataToConvert[1][thdInd] = 0.5f * mass * dataToConvert[1][thdInd] * dataToConvert[1][thdInd] / B_s;
			else
				dataToConvert[1][thdInd] = sqrt(2.0f * dataToConvert[1][thdInd] * B_s / mass);
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
				*vperpOrMu = sqrt(2.0f * (*vperpOrMu) * B_s / mass);
		}
	}

	__device__ __host__ float accel1dCUDA(const float vs_RK, const float t_RK, const float* args, BModel* bmodel, EField* efield) //made to pass into 1D Fourth Order Runge Kutta code
	{//args array: [s_0, mu, q, m, simtime]
		float F_lor, F_mir, stmp;
		stmp = args[0] + vs_RK * t_RK; //ps_0 + vs_RK * t_RK

		//Mirror force
		F_mir = -args[1] * bmodel->getGradBAtS(stmp, t_RK + args[4]); //-mu * gradB(pos, runge-kutta time + simtime)

		//Lorentz force - simply qE - v x B is taken care of by mu - results in kg.m/s^2 - to convert to Re equivalent - divide by Re
		F_lor = args[2] * efield->getEFieldAtS(stmp, t_RK + args[4]); //q * EFieldatS

		return (F_lor + F_mir) / args[3];
	}//returns an acceleration in the parallel direction to the B Field

	__device__ __host__ float RungeKutta4CUDA(const float y_0, const float h, const float* funcArg, BModel* bmodel, EField* efield)
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

	__global__ void simActiveCheck(float** currData_d, bool* simDone, int len)
	{
		//Answers the question: Are there no particles left in the simulation?
		//stores the value to simDone, which is defaulted to true, and flipped to false
		//only if t_escape is less than zero for at least one particle
		//(in that case, the sim is not completely done iterating)
		if (*simDone)
		{
			unsigned int thdInd{ blockIdx.x * blockDim.x + threadIdx.x };
			if (thdInd >= len) return;

			const float* t_escape_d{ currData_d[4] }; //const float* t_incident_d{ currData_d[3] }; //to be implemented
			
			if (t_escape_d[thdInd] >= 0.0f) //particle has escaped the sim
				return;
			else
				(*simDone) = false;
		}
	}

	__global__ void iterateParticle(float** currData_d, BModel** bmodel, EField* efield,
		const float simtime, const float dt, const float mass, const float charge, const float simmin, const float simmax, int len)
	{
		int thdInd{ blockIdx.x * blockDim.x + threadIdx.x };

		if (thdInd >= len) return;
		
		float* s_d{ currData_d[2] };
		float* t_incident_d{ currData_d[3] };
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

		//RK4 (plus v0 in this case) gives v at the next time step (indicated vf in this note):
		//for downgoing (as an example), due to the mirror force, vf will be lower than v0 as the mirror force is acting in the opposite direction as v
		//along the path of the particle, ds, and so s will end up higher if we use ds = (vf * dt) than where it would realistically
		//if we use the ds = (v0 * dt), s will be lower down than where it would end up really (due to the fact that the mirror force acting along ds
		//will slow v down as the particle travels along ds), so I take the average of the two and it seems close enough s = (v0 + (v0 + dv)) / 2 * dt = v0 + dv/2 * dt
		//hence the /2 factor below - FYI, this was checked by the particle's energy (steady state, no E Field) remaining the same throughout the simulation

		float v{ v_d[thdInd] };
		float v_pr{ v + RungeKutta4CUDA(v, dt, args, *bmodel, efield) };
		
		s0_d[thdInd] = args[0];
		v0_d[thdInd] = v;
		v_d[thdInd] = v_pr;
		s_d[thdInd] = args[0] + (v + v_pr) / 2 * dt;
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
		*vpara += RungeKutta4CUDA(*vpara, dt, args, bmodel, efield);
		*s += (*vpara + v_orig) / 2 * dt;
	}
}

//Simulation member functions
void Simulation::initializeSimulation()
{
	if (BFieldModel_m.size() == 0)
		throw logic_error("Simulation::initializeSimulation: no Earth Magnetic Field model specified");
	if (particles_m.size() == 0)
		throw logic_error("Simulation::initializeSimulation: no particles in simulation, sim cannot be initialized without particles");
	
	if (EFieldModel_m.size() == 0) //make sure an EField (even if empty) exists
	{
		size_t dev = 0;
		do
		{
			utils::GPU::setDev(dev);

			EFieldModel_m.push_back( make_unique<EField>() );
			dev++;
		} while (dev < gpuCount_m);

	}
	
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
				BFieldModel_m.at(dev)->this_dev(), part->mass(), true, part->getNumParticlesPerGPU(dev));
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
			if (cudaloopind % checkDoneEvery == 0) { CUDA_API_ERRCHK(cudaMemset(simDone_d.at(dev), true, sizeof(bool))); } //if it's going to be checked in this iter (every checkDoneEvery iterations), set to true

			for (size_t part = 0; part < particles_m.size(); part++)
			{
				Particles* p{ particles_m.at(part).get() };
				iterateParticle <<< grid.at(dev).at(part), BLOCKSIZE >>> (p->getCurrDataGPUPtr(dev), BFieldModel_m.at(dev)->this_dev(), EFieldModel_m.at(dev)->this_dev(),
					simTime_m, dt_m, p->mass(), p->charge(), simMin_m, simMax_m, p->getNumParticlesPerGPU(dev));

				//kernel will set boolean to false if at least one particle is still in sim
				if (cudaloopind % checkDoneEvery == 0)
					simActiveCheck <<< grid.at(dev).at(part), BLOCKSIZE >>> (p->getCurrDataGPUPtr(dev), simDone_d.at(dev), p->getNumParticlesPerGPU(dev));
			}
		}

		for (int dev = 0; dev < gpuCount_m; dev++)
		{   //synchronize all devices
			CUDA_API_ERRCHK(cudaSetDevice((int)dev));
			CUDA_KERNEL_ERRCHK_WSYNC_WABORT();   //side effect: cudaDeviceSynchronize() needed for computeKernel to function properly, which this macro provides
		}

		for (int dev = 0; dev < gpuCount_m; dev++)
			for (auto sat = satPartPairs_m.begin(); sat < satPartPairs_m.end(); sat++)
				(*sat)->satellite->iterateDetector(simTime_m, dt_m, BLOCKSIZE, dev);

		for (int dev = 0; dev < gpuCount_m; dev++)
		{   //synchronize all devices
			CUDA_API_ERRCHK(cudaSetDevice((int)dev));
			CUDA_KERNEL_ERRCHK_WSYNC_WABORT();   //side effect: cudaDeviceSynchronize() needed for computeKernel to function properly, which this macro provides
		}
		
		if (cudaloopind % checkDoneEvery == 0)
		{       //only executes once
			{
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
			vperpMuConvert_d <<< grid.at(dev).at(p), BLOCKSIZE >>> (part->getCurrDataGPUPtr(dev), BFieldModel_m.at(dev)->this_dev(),
				part->mass(), false, part->getNumParticlesPerGPU(dev));
		}
	}

	for (int dev = 0; dev < gpuCount_m; dev++)
	{
		CUDA_API_ERRCHK(cudaSetDevice(dev));
		for (int s = 0; s < satPartPairs_m.size(); s++)
		{
			Satellite* sat{ satPartPairs_m.at(s)->satellite.get() };
			Particles* part{ satPartPairs_m.at(s)->particle.get() };
			vperpMuConvert_d <<< part->getNumParticlesPerGPU(dev) / BLOCKSIZE, BLOCKSIZE >>> (sat->get2DDataGPUPtr(dev), 
				BFieldModel_m.at(dev)->this_dev(), part->mass(), false, part->getNumParticlesPerGPU(dev));
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
	cudaDeviceProp devProp;
	int gpuCount	 = 0;
	float computeTotal = 0;

	// Get total number of NVIDIA GPUs
	if (CUDA_API_ERRCHK(cudaGetDeviceCount(&gpuCount)))
	{
		cerr << "Cannot get GPU count.  Only default GPU used.  Assuming at least one CUDA capable device.\n";
		computeSplit_m.push_back(1);
		gpuCount_m = 1;
		return;
	}

	// Store the count of the total number of GPUs
	gpuCount_m = gpuCount;
	Log_m->createEntry("Simulation::setupGPU: Setting up " + to_string(gpuCount) + " GPU(s).");
	
	// Iterate over each GPU and determine how much data it can handle
	for (int gpu = 0; gpu < gpuCount; gpu++)
	{
		// Get the GPU Speed
		CUDA_API_ERRCHK(cudaGetDeviceProperties(&devProp, gpu));

		// For the author's machine, MP count gives a good metric to split tasks evenly
		// In future: either optimize for specific hardware create a more precise equation
		float compute = static_cast<float>(/*devProp.clockRate/1024 * */devProp.multiProcessorCount);
		computeTotal += compute;
		computeSplit_m.push_back(compute);  //need to use floats to get decimal numbers
	}
	
	// Iterate through computeSplit and get percent ratio work each device will get
	for (size_t i = 0; i < computeSplit_m.size(); ++i)
	{
		computeSplit_m.at(i) /= computeTotal;
	}
}


vector<size_t> Simulation::getSplitSize(size_t numOfParticles)
{   //returns the number of particles a device will receive based on the pct in computeSplit_m
	//returns block aligned numbers except last one, if total is not a multiple of block size
	
	vector<size_t> particleSplit;

	auto getBlockAlignedCount = [](size_t count)
	{
		size_t ret{ count };
		float bs{ static_cast<float>(BLOCKSIZE) };
		
		if (count % BLOCKSIZE)
		{
			float pct(static_cast<float>(count % BLOCKSIZE) / bs);  //find percent of blocksize of the remainder
			if (pct >= 0.5)      //if remainder is over 50% of block size, add an extra block
				ret = static_cast<size_t>((count / BLOCKSIZE + 1) * BLOCKSIZE);
			else                 //else just return the block aligned size
				ret = static_cast<size_t>((count / BLOCKSIZE) * BLOCKSIZE);
		}

		return ret;
	};

	size_t total{ 0 };
	for (const auto& comp : computeSplit_m)
	{
		size_t bsaln{ getBlockAlignedCount(static_cast<size_t>(comp * static_cast<float>(numOfParticles))) };
		total += bsaln;
		particleSplit.push_back(bsaln);
	}

	//above code creates particle count in multiples of blocksize
	size_t diff{ 0 };
	if (total < numOfParticles)  //check that we aren't missing particles or have extra blocks
	{
		diff = numOfParticles - total;

		while (diff / BLOCKSIZE)  //does not execute when diff < BLOCKSIZE
		{   //if more than one block not accounted for... (shouldn't happen)
			cerr << "Simulation::getSplitSize: total < # parts : Need " + to_string(diff / BLOCKSIZE) + " more full blocks\n";
			particleSplit.at((diff / BLOCKSIZE) % gpuCount_m) += BLOCKSIZE;  //add one full block to GPU
			diff -= BLOCKSIZE;
			//overflows are prevented by the fact that when the int div product is 0, this doesn't execute
		}
		
		//less than one full block of unaccounted for particles remains.  Add diff
		particleSplit.back() += diff;  //add diff to total
		Log_m->createEntry("Simulation::getSplitSize: Adding diff " + to_string(diff) + ".  Total: " + to_string(total+diff));
		
		for(size_t dev = 0; dev < gpuCount_m; dev++)
			Log_m->createEntry("Simulation::getSplitSize: GPU " + to_string(dev) + ": particle count: " + to_string(particleSplit.at(dev)));
	}
	else if (total > numOfParticles)
	{
		diff = total - numOfParticles;

		while (diff / BLOCKSIZE)
		{   //if one or more whole extra blocks created... (shouldn't happen)
			cerr << "Simulation::getSplitSize: total > # parts " + to_string(diff / BLOCKSIZE) + " extra blocks to create\n";
			particleSplit.at((diff / BLOCKSIZE) % gpuCount_m) -= BLOCKSIZE;  //remove one full block from GPU
			diff -= BLOCKSIZE;
		}

		//less than one full block of particles extra - subtract diff
		particleSplit.back() -= diff;  //shrink total by diff
		Log_m->createEntry("Simulation::getSplitSize: Subtracting diff " + to_string(diff) + ".  Total: " + to_string(total - diff));
		
		for (size_t dev = 0; dev < gpuCount_m; dev++)
			Log_m->createEntry("Simulation::getSplitSize: GPU " + to_string(dev) + ": particle count: " + to_string(particleSplit.at(dev)));
	}
	
	return particleSplit;
}
