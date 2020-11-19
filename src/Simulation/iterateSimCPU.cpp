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
	void vperpMuConvert(const double vpara, double* vperpOrMu, const double s, const double t_convert, BModel* BModel, const double mass, const bool vperpToMu);
	void iterateParticle(double* vpara, double* mu, double* s, double* t_incident, double* t_escape, BModel* BModel, EField* efield, const double simtime, const double dt, const double mass, const double charge, const double simmin, const double simmax);
}

void Simulation::__iterateSimCPU(int numberOfIterations, int checkDoneEvery)
{
	printSimAttributes(numberOfIterations, checkDoneEvery, "CPU");

	using namespace physics;
	for (auto part = particles_m.begin(); part < particles_m.end(); part++)
	{
		(*part)->loadDataFromMem((*part)->data(true), false); //copies orig data to curr - normally this happens from CPU->GPU->CPU, but we aren't using the GPU here

		vector<vector<double>>& data = (*part)->__data(false);
		
		omp_set_num_threads(std::thread::hardware_concurrency());

		#pragma omp parallel for
		for (int ind = 0; ind < (*part)->getNumberOfParticles(); ind++)
		{//convert vperp to mu in Particles memory
			vperpMuConvert(data.at(0).at(ind), &data.at(1).at(ind), data.at(2).at(ind), data.at(4).at(ind), BFieldModel_m.get(), (*part)->mass(), true);
		}
	}

	bool done{ false };
	size_t initEntry{ Log_m->createEntry("Iteration 1", false) };
	for (long cudaloopind = 0; cudaloopind < numberOfIterations; cudaloopind++)
	{
		if (cudaloopind % checkDoneEvery == 0) { done = true; }
		for (auto part = particles_m.begin(); part < particles_m.end(); part++)
		{
			vector<vector<double>>& data = (*part)->__data(false); //get a reference to the particle's curr data array-

			#pragma omp parallel for
			for (int ind = 0; ind < (*part)->getNumberOfParticles(); ind++)
			{
				iterateParticle(&data.at(0).at(ind), &data.at(1).at(ind), &data.at(2).at(ind), &data.at(3).at(ind), &data.at(4).at(ind),
					BFieldModel_m.get(), EFieldModel_m.get(), simTime_m, dt_m, (*part)->mass(), (*part)->charge(), simMin_m, simMax_m);
				if ((cudaloopind % checkDoneEvery == 0) && done && (data.at(4).at(ind) < 0.0))
				{
					//#pragma omp atomic
					done = false; //maybe a problem - will have at least several simultaneous attempts to write...not thread-safe, but it's simply a flag, so here it doesn't need to be maybe?
					//OpenMP 2.0 (max supported version by VS, but very old) doesn't support done = false; as a legit expression following #pragma omp atomic
				}
			}
		}

		for (auto& sat : satPartPairs_m)
			(*sat).satellite->iterateDetectorCPU((*sat).particle->data(false), simTime_m, dt_m);

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
		vector<vector<double>> tmp{ (*part)->data(false) };
		for (int ind = 0; ind < (*part)->getNumberOfParticles(); ind++)
		{//convert mu to vperp in Particles memory
			vperpMuConvert(tmp.at(0).at(ind), &tmp.at(1).at(ind), tmp.at(2).at(ind), tmp.at(4).at(ind), BFieldModel_m.get(), (*part)->mass(), false);
		}
		(*part)->loadDataFromMem(tmp, false);
	}

	saveReady_m = true;
	saveDataToDisk();
	simTime_m = 0.0;

	std::cout << "Total sim time: " << Log_m->timeElapsedTotal_s() << " s" << std::endl;

	Log_m->createEntry("End Iteration of Sim:  " + std::to_string(numberOfIterations));
}