from simulation import *
savedir, dtg = setupFolders()
sim = Simulation(DLLLOCATION, savedir, DT, MIN_S_SIM, MAX_S_SIM)

sim.simDLL_m.runSingleElectronAPI.argtypes = (ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int)
sim.simDLL_m.runSingleElectronAPI.restype = None

sim.simDLL_m.runSingleElectronAPI(sim.simulationptr, -1016333.4391307, 281854.2076197, 19881647.2473464, 0.0, 75000, 500)