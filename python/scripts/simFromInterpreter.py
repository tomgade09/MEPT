import numpy as np
import math
import os, sys, inspect
import matplotlib.pyplot as plt

pyscriptdir = os.path.dirname(os.path.abspath(inspect.getsourcefile(lambda:0)))
sys.path.append(os.path.normpath(pyscriptdir + '/../'))
from simulation import *

dllLocation, savedir, rootdir, dtg = setupFolders()
sim = Simulation(dllLocation, rootdir, 0.01, 8.371e6, 4 * 6.371e6, 2.5, 1000, 0.0)
finalDat, origDat, satDat = sim.runSim(200000)

MASS_ELEC = 9.10938356e-31
MASS_PROT = 1.67262190e-27
J_PER_EV = 1.60218e-19

atan2vec = np.vectorize(math.atan2)

elecpitches = atan2vec(abs(np.array(origDat[0][1])), -np.array(origDat[0][0])) * 180 / math.pi
elecenergies = 0.5 * MASS_ELEC * (np.square(np.array(origDat[0][0])) + np.square(np.array(origDat[0][1]))) / J_PER_EV
elecIonsphEnergies = elecenergies[0:50176]
elecIonsphPitches = elecpitches[0:50176]
elecIonsphVpara = np.array(origDat[0][0][0:50176])
elecIonsphVperp = np.array(origDat[0][1][0:50176])
elecIonsphZ = np.array(origDat[0][2][0:50176])
elecMagsphEnergies = elecenergies[50176:100352]
elecMagsphPitches = elecpitches[50176:100352]
elecMagsphVpara = np.array(origDat[0][0][50176:100352])
elecMagsphVperp = np.array(origDat[0][1][50176:100352])
elecMagsphZ = np.array(origDat[0][2][50176:100352])

ionspitches = atan2vec(abs(np.array(origDat[1][1])), -np.array(origDat[1][0])) * 180 / math.pi
ionsenergies = 0.5 * MASS_PROT * (np.square(np.array(origDat[1][0])) + np.square(np.array(origDat[1][1]))) / J_PER_EV
ionsIonsphEnergies = ionsenergies[0:50176]
ionsIonsphPitches = ionspitches[0:50176]
ionsIonsphVpara = np.array(origDat[1][0][0:50176])
ionsIonsphVperp = np.array(origDat[1][1][0:50176])
ionsIonsphZ = np.array(origDat[1][2])
ionsMagsphEnergies = ionsenergies[50176:100352]
ionsMagsphPitches = ionspitches[50176:100352]
ionsMagsphVpara = np.array(origDat[1][0][50176:100352])
ionsMagsphVperp = np.array(origDat[1][1][50176:100352])
ionsMagsphZ = np.array(origDat[1][2][50176:100352])

elecIonsphTimesPre = np.array([[]])
elecMagsphTimesPre = np.array([[]])
ionsIonsphTimesPre = np.array([[]])
ionsMagsphTimesPre = np.array([[]])

elecIonsphTimes = np.array([])
elecMagsphTimes = np.array([])
ionsIonsphTimes = np.array([])
ionsMagsphTimes = np.array([])

elecIonsphTimesPre = [np.array(satDat[0][3][0:50176]), np.array(satDat[2][3][0:50176])]
elecMagsphTimesPre = [np.array(satDat[0][3][50176:100352]), np.array(satDat[2][3][50176:100352])]

ionsIonsphTimesPre = [np.array(satDat[1][3][0:50176]), np.array(satDat[3][3][0:50176])]
ionsMagsphTimesPre = [np.array(satDat[1][3][50176:100352]), np.array(satDat[3][3][50176:100352])]

for iii in range(50176):
	if elecIonsphTimesPre[0][iii] == -1:
		if elecIonsphTimesPre[1][iii] == -1:
			elecIonsphTimes = np.append(elecIonsphTimes, -1)
		else:
			elecIonsphTimes = np.append(elecIonsphTimes, elecIonsphTimesPre[1][iii])
	else:
		elecIonsphTimes = np.append(elecIonsphTimes, elecIonsphTimesPre[0][iii])
	
	if elecMagsphTimesPre[0][iii] == -1:
		if elecMagsphTimesPre[1][iii] == -1:
			elecMagsphTimes = np.append(elecMagsphTimes, -1)
		else:
			elecMagsphTimes = np.append(elecMagsphTimes, elecMagsphTimesPre[1][iii])
	else:
		elecMagsphTimes = np.append(elecMagsphTimes, elecMagsphTimesPre[0][iii])

	if ionsIonsphTimesPre[0][iii] == -1:
		if ionsIonsphTimesPre[1][iii] == -1:
			ionsIonsphTimes = np.append(ionsIonsphTimes, -1)
		else:
			ionsIonsphTimes = np.append(ionsIonsphTimes, ionsIonsphTimesPre[1][iii])
	else:
		ionsIonsphTimes = np.append(ionsIonsphTimes, ionsIonsphTimesPre[0][iii])
	
	if ionsMagsphTimesPre[0][iii] == -1:
		if ionsMagsphTimesPre[1][iii] == -1:
			ionsMagsphTimes = np.append(ionsMagsphTimes, -1)
		else:
			ionsMagsphTimes = np.append(ionsMagsphTimes, ionsMagsphTimesPre[1][iii])
	else:
		ionsMagsphTimes = np.append(ionsMagsphTimes, ionsMagsphTimesPre[0][iii])

elecIonsphTimesPre = np.array([])
elecMagsphTimesPre = np.array([])
ionsIonsphTimesPre = np.array([])
ionsMagsphTimesPre = np.array([])

logVec = np.vectorize(math.log)

#elecIonsphLogTimes = logVec(elecIonsphTimes)
#elecMagsphLogTimes = logVec(elecMagsphTimes)
#ionsIonsphLogTimes = logVec(ionsIonsphTimes)
#ionsMagsphLogTimes = logVec(ionsMagsphTimes)