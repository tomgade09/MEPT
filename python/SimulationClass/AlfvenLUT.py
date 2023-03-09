from __future__ import absolute_import, division, print_function
from Simulation import *

def getElectricFieldLUT(self, cols, entries):
    self.simDLL_m.getPointerToElectricFieldDataWrapper.argtypes = (ctypes.c_void_p, ctypes.c_int)
    self.simDLL_m.getPointerToElectricFieldDataWrapper.restype = ctypes.POINTER(ctypes.c_double)

    ret = []
    for iii in range(cols):
        col_c = self.simDLL_m.getPointerToElectricFieldDataWrapper(self.simulationptr, iii)
        column = []
        for jjj in range(entries):
            column.append(col_c[jjj])
        ret.append(column)
    
    return ret
    
Simulation.getElectricFieldLUT = getElectricFieldLUT

if __name__ == '__main__':
    print("SimulationAPI.py is not meant to be called as main.  Run a simulation file and that will import this.")
