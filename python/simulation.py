import time, csv
import os, sys

# Current Directory
from __simulationvariables import *
from __setup import *

# ./SimulationClass Directory
sys.path.append(os.path.normpath(PYROOTDIR + '/SimulationClass/'))
from Simulation import *
from __plotParticles import *

def simulationRunMain():
    #savedir, dtg = setupFolders()
    #print("================  SIMULATION ", dtg, " ================")

    sim = Simulation(DLLLOCATION, PYROOTDIR+"/..", DT, MIN_S_SIM, MAX_S_SIM)
    #sim.addEFieldModel("QSPS", "9796766.33308899, 11122579.974093, 7.54254e-5")
    sim.setupExampleSim(NUMPARTICLES)
    #sim.createSatellite(9796766.33308899, False, "QSPSBtmUpg", NUMPARTICLES)
    #sim.createSatellite(9796766.33308899, True, "QSPSBtmDng", NUMPARTICLES)
    #sim.createSatellite(11122579.974093, False, "QSPSTopUpg", NUMPARTICLES)
    #sim.createSatellite(11122579.974093, True, "QSPSTopDng", NUMPARTICLES)
    sim.run(NUMITER, ITERBTWCOUTS)

    #fields = sim.getFieldsAtAllS(0.0, 4000, (sim.simMax_m - sim.simMin_m) / (4000), sim.simMin_m)
    #for iii in range(len(fields[2])):
    #    fields[2][iii] /= RADIUS_EARTH #Normalizes location where field is measured (makes graph reading easier)
    #for iii in range(len(fields[0])):
    #    fields[0][iii] *= 1e9 #Displays B Field in nT

    #plotFields(fields[0], fields[1], fields[2], True)

    return


if __name__ == '__main__':
    simulationRunMain()

if __name__ != '__main__':
    print("Functions available:")
    print()
    print("setupFolders(): Sets up folders according to default structure.  Changes dir to the data root folder (/PROJECTROOT/distgraphs/DATE_TIME).  Returns dllLocation, savedir, rootdir, DATE_TIME string")
    print()
    print("simulationRunMain(): Calls setupFolders().  Runs the simulation with the variables specified in simulationvariables.py and ends the sim when done.  Returns nothing, and saves data to the default folder structure.")
    print()