# SIMULATION VARIABLES
# Set variables related to the simulation here

import os, inspect
from sys import platform

#Directories
PYROOTDIR = os.path.dirname(os.path.abspath(inspect.getsourcefile(lambda:0)))
ROOTDIR = os.path.abspath(PYROOTDIR + '/../')
LIBDIR = os.path.abspath(ROOTDIR + '/lib/')

if platform == "win32":
    DLLLOCATION = os.path.abspath(LIBDIR + '/C-ADPIC-PTEM.dll')
elif platform == "linux" or platform == "linux2":
    DLLLOCATION = os.path.abspath(LIBDIR + '/C-ADPIC-PTEM.so')
elif platform == "darwin":
    DLLLOCATION = os.path.abspath(LIBDIR + '/C-ADPIC-PTEM.dylib') #not tested, but probably would work through makefile

#Physical Constants
RADIUS_EARTH = 6.3712e6
J_PER_EV = 1.6021766209e-19
MASS_ELEC = 9.10938356e-31
MASS_PROT = 1.672621898e-27
CHARGE_ELEM = 1.6021766209e-19
PI = 3.14159265358979323846

#Simulation Specific Constants
DT = 0.001
MIN_S_SIM = 628565.8510817 #101322.378940846 #z(geoc): 620km
MAX_S_SIM = 19881647.2473464 #z(geoc): 3.0 * RADIUS_EARTH
NUMITER = 50000
ITERBTWCOUTS = 500

#Distribution Quantities
LOADDIST = True
DISTINFOLDER = os.path.abspath(ROOTDIR + '/_in/data/')

NUMPARTICLES = 3456000 #172800 #115200
INITIAL_T_ION_EV = 2.5
INITIAL_T_MAG_EV = 1000

#B Field Model
DIPOLE = True
IGRF = False

ILAT = 72.0

#Satellites Data


#QSPS E


#Alfven E LUT
ALFVENLUTCSV = ""
#LUTDATAFILE = os.path.abspath(ROOTDIR + '/_in/ez.out')
