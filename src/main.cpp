#include <string>
#include "Simulation/Simulation.h"


int main(int argc, char* argv[])
{
	//Simulation characteristics - can be changed, but this is
	//the sample problem that we are testing speedup against
	double dt{ 0.001 };                      //time step size delta t
	double simmin{ 628565.8510817 };         //minimum altitude of the sim - when particle goes <, it's not tracked any more
	double simmax{ 19881647.2473464 };       //maximum altitude of the sim
	int numiter{ 50000 };                    //number of timesteps we do for each particle
	int numParts{ 3456000 };                 //number of particles in our simulation
	
	//create an instance of the container class that holds the entire PTEM simulation
	Simulation sim{ dt, simmin, simmax };

	//set the BField model we will use - here this is a dipole representation (hence, "DipoleB") of the
	//Earth's magnetic ("B") field stored in a lookup table ("LUT") with 1000000 entries (last parameter)
	//evenly spaced from simmin to simmax (see above variables) - so the BField is a Dipole B Lookup Table
	//We use a lookup table because it is much faster to use this than to calculate the field each time we
	//update 3.5 million particles positions.
	sim.setBFieldModel("DipoleBLUT", { 72.0, 637.12, 1000000 });

	//no E Fields are used yet, although they could be specified as below
	//sim->addEFieldModel("QSPS", { 3185500.0, 6185500.0, 0.02, 6556500.0, 9556500.0, 0.04 });

	//create a container for the particles that we will generate
	//first argument is the name of the particle, second is the mass, third is the electric charge, and fourth
	//is the number of particles that the class will have to track and allocate memory for
	sim.createParticlesType("elec", MASS_ELECTRON, -1 * CHARGE_ELEM, numParts);

	//create a distribution of particles and load the data into the Particles class, log distribution attributes
	//you shouldn't have to do anything with this class / functionality
	ParticleDistribution pd{ "./", sim.particles(0)->attributeNames(), sim.particles(0)->name(),
		sim.particles(0)->mass(), { 0.0, 0.0, 0.0, 0.0, -1.0 }, false };

	sim.getLog()->createEntry("Created Particle Distribution:");
	sim.getLog()->createEntry("     Energy: " + std::to_string(96) + " E bins, " + std::to_string(0.5) + " - " + std::to_string(4.5) + " logE");
	sim.getLog()->createEntry("     Pitch : " + std::to_string(numParts / (96)) + " E bins, " + std::to_string(180) + " - " + std::to_string(0) + " deg");

	pd.addEnergyRange(96, 0.5, 4.5);
	pd.addPitchRange(36000, 180.0, 0.0);

	sim.particles(0)->loadDistFromPD(pd, sim.simMin(), sim.simMax());

	//create our "Satellites" that track electrons as they pass by a given altitude
	//first argument is the index of the particle - here we only have one type (created right above this)
	//so the index is 0; second argument is the altitude of the satellite; third is whether or not the
	//"detector" is facing upward (if not, it's facing downward) - this determines whether the satellite
	//detects upgoing or downgoing particles, and fourth is the name of the satellite
	sim.createTempSat(0, simmin, true, "btmElec");
	sim.createTempSat(0, simmax, false, "topElec");
	sim.createTempSat(0, 4071307.04106411, false, "4e6ElecUpg");
	sim.createTempSat(0, 4071307.04106411, true, "4e6ElecDng");
	
	//create a log entry detailing what we've done so far
	sim.getLog()->createEntry("main: Simulation setup complete");

	
	//
	//
	//Now, the simulation has been defined.  Call a few setup functions and execute the sim
	//
	//
	sim.initializeSimulation();                                    //finalizes satellites
	sim.iterateSimulation(numiter, 500);  //this is the main loop that calls the CUDA kernel, also saves data to disk in folder structure

	//memory is freed, everything is cleaned up upon destruction of the classes when the program exits
	
	return 0;
}