import ctypes
from utilsAPI import *

MASS_ELEC = 9.10938356e-31

name_orig = ctypes.create_string_buffer(bytes('orig dist', encoding='utf-8'))
name_comp = ctypes.create_string_buffer(bytes('comp dist', encoding='utf-8'))
fold_orig = ctypes.create_string_buffer(bytes('./../../_dataout/200826_00.12.00.hiresDown0-16/bins/particles_final', encoding='utf-8'))
fold_comp = ctypes.create_string_buffer(bytes('./../../_dataout/220727_15.49.06.RK4updatetest.vs.hiresDown0-16.200826/bins/particles_final', encoding='utf-8'))
#fold_orig = ctypes.create_string_buffer(bytes('./../../_dataout/180809_15.36.28/bins/particles_final', encoding='utf-8'))
#fold_comp = ctypes.create_string_buffer(bytes('./../../_dataout/180813_15.46.10.CPUOpenMP/bins/particles_final', encoding='utf-8'))
attrs = ctypes.create_string_buffer(bytes('vpara,vperp,s,t_inc,t_esc', encoding='utf-8'))
origname = ctypes.create_string_buffer(bytes('elec', encoding='utf-8'))
compname = ctypes.create_string_buffer(bytes('elec', encoding='utf-8'))

dist_orig = ctypes.c_void_p
dist_comp = ctypes.c_void_p
dist_orig = simDLL.DFDLoadAPI(name_orig, fold_orig, attrs, origname, MASS_ELEC)
dist_comp = simDLL.DFDLoadAPI(name_comp, fold_comp, attrs, compname, MASS_ELEC)

#simDLL.DFDCompareAPI(dist_orig, dist_comp)

def dataPtr(distptr, attrind):
	return simDLL.DFDDataAPI(distptr, attrind)
	
def printInd(distptr, dataind):
	simDLL.DFDPrintAPI(distptr, dataind)

def printDiff(dataind):
	simDLL.DFDPrintDiffAPI(dist_orig, dist_comp, dataind)
	
def printZeroes(distptr):
	simDLL.DFDZeroesAPI(distptr)

def compare():
	simDLL.DFDCompareAPI(dist_orig, dist_comp)
	
def deleteDist(distptr):
	simDLL.DFDDeleteAPI(distptr)
	
print("Distribution Class Pointers (below as distptr):")
print("1. dist_orig: original distribution")
print("2. dist_comp: distribution to compare")
print("")
print("Available Functions:")
print("dataPtr(distptr, ind of attr)  - return double* of data attr")
print("printInd(distptr, ind of data) - print index of data")
print("printDiff(ind of data)         - print difference between dists at index")
print("printZeroes(distptr)           - print how many zeroes")
print("compare()                      - print various aspects comparing two dists")
print("deleteDist(distptr)            - delete, free memory of dist")
print("")
