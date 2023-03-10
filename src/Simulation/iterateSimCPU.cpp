//Standard Library includes
#include <string>
#include <cmath>
#include <sstream>
#include <iomanip>

//Project specific includes
#include "physicalconstants.h"
#include "utils/loopmacros.h"
#include "Simulation/Simulation.h"
#include "ErrorHandling/cudaErrorCheck.h"

//OpenMP
#include <omp.h>
#include <thread>

using std::cout;
using std::cerr;
using std::endl;
using std::setw;
using std::fixed;
using std::vector;
using std::to_string;

//forward decls
namespace physics
{
	void vperpMuConvert(const mpers vpara, flPt_t* vperpOrMu, const meters s, const seconds t_convert, BModel* BModel, const kg mass, const bool vperpToMu);
	void iterateParticle(mpers* vpara, flPt_t* mu, meters* s, seconds* t_incident, seconds* t_escape, BModel* BModel, EField* efield, const seconds simtime, const seconds dt, const kg mass, const coulomb charge, const meters simmin, const meters simmax);
}


// For the time being, CPU execution doesn't make any sense when CUDA is required to compile MEPT anyway.  It's assumed an NVIDIA GPU is present on the system due to the CUDA lib requirement.
void Simulation::__iterateSimCPU(size_t numberOfIterations, size_t checkDoneEvery)
{
	/*printSimAttributes(numberOfIterations, checkDoneEvery);

	using namespace physics;
	for (auto part = particles_m.begin(); part < particles_m.end(); part++)
	{
		(*part)->loadDataFromMem((*part)->data(true), false); //copies orig data to curr - normally this happens from CPU->GPU->CPU, but we aren't using the GPU here

		fp2Dvec& data = (*part)->__data(false);
		
		omp_set_num_threads(std::thread::hardware_concurrency());

		#pragma omp parallel for
		for (int ind = 0; ind < static_cast<int>((*part)->getNumberOfParticles()); ind++)
		{//convert vperp to mu in Particles memory
			vperpMuConvert(data.at(0).at(ind), &data.at(1).at(ind), data.at(2).at(ind), data.at(4).at(ind), BFieldModel_m.get(), (*part)->mass(), true);
		}
	}

	bool done{ false };
	size_t initEntry{ Log_m->createEntry("Iteration 1", false) };
	for (size_t cudaloopind = 0; cudaloopind < numberOfIterations; cudaloopind++)
	{
		if (cudaloopind % checkDoneEvery == 0) { done = true; }
		for (auto part = particles_m.begin(); part < particles_m.end(); part++)
		{
			fp2Dvec& data = (*part)->__data(false); //get a reference to the particle's curr data array-

			#pragma omp parallel for
			for (int ind = 0; ind < static_cast<int>((*part)->getNumberOfParticles()); ind++)
			{
				iterateParticle(&data.at(0).at(ind), &data.at(1).at(ind), &data.at(2).at(ind), &data.at(3).at(ind), &data.at(4).at(ind),
					BFieldModel_m.get(), EFieldModel_m.get(), simTime_m, dt_m, (*part)->mass(), (*part)->charge(), simMin_m, simMax_m);
				if ((cudaloopind % checkDoneEvery == 0) && done && (data.at(4).at(ind) < 0.0f))
				{
					//#pragma omp atomic
					done = false; //maybe a problem - will have at least several simultaneous attempts to write...not thread-safe, but it's simply a flag, so here it doesn't need to be maybe?
					//OpenMP 2.0 (max supported version by VS, but very old) doesn't support done = false; as a legit expression following #pragma omp atomic
				}
			}
		}

		for (auto& sat : satellites_m)  //TODO: if multiple particles added, need to loop over particles
			sat->iterateDetectorCPU(particles_m.at(0)->data(false), simTime_m, dt_m);

		if (cudaloopind % checkDoneEvery == 0)
		{
			{
				string loopStatus;
				stringstream out;

				out << setw(to_string(numberOfIterations).size()) << cudaloopind;
				loopStatus = out.str() + " / " + to_string(numberOfIterations) + "  |  Sim Time (s): ";
				out.str("");
				out.clear();

				out << setw(to_string((numberOfIterations)*dt_m).size()) << fixed << simTime_m;
				loopStatus += out.str() + "  |  Real Time Elapsed (s): " + to_string(Log_m->timeElapsedSinceEntry_s(initEntry));

				Log_m->createEntry("CPU: " + loopStatus);
				cout << loopStatus << "\n";
			}

			if (done) { std::cout << "All particles finished early.  Breaking loop." << std::endl; break; }
		}

		incTime();
	}

	for (auto part = particles_m.begin(); part < particles_m.end(); part++)
	{
		fp2Dvec tmp{ (*part)->data(false) };
		for (size_t ind = 0; ind < (*part)->getNumberOfParticles(); ind++)
		{//convert mu to vperp in Particles memory
			vperpMuConvert(tmp.at(0).at(ind), &tmp.at(1).at(ind), tmp.at(2).at(ind), tmp.at(4).at(ind), BFieldModel_m.get(), (*part)->mass(), false);
		}
		(*part)->loadDataFromMem(tmp, false);
	}

	saveReady_m = true;
	saveDataToDisk();
	simTime_m = 0.0f;

	std::cout << "Total sim time: " << Log_m->timeElapsedTotal_s() << " s" << std::endl;

	Log_m->createEntry("End Iteration of Sim:  " + std::to_string(numberOfIterations));
	*/
}