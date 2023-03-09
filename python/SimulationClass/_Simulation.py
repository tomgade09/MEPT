import ctypes

class _SimulationCDLL: #acts as parent class for actual Simulation class - removes some of the messy argtypes/restype functions out to make it cleaner and easier to read
    def __init__(self, dllLoc):
        self.dllLoc_m = dllLoc
        
        self.simDLL_m = ctypes.CDLL(self.dllLoc_m)
        
        #Simulation Management Functions
        self.simDLL_m.createSimulationAPI.argtypes = (ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_char_p)
        self.simDLL_m.createSimulationAPI.restype = ctypes.c_void_p
        self.simDLL_m.initializeSimulationAPI.argtypes = (ctypes.c_void_p,)
        self.simDLL_m.initializeSimulationAPI.restype = None
        self.simDLL_m.iterateSimCPUAPI.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
        self.simDLL_m.iterateSimCPUAPI.restype = None
        self.simDLL_m.iterateSimulationAPI.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
        self.simDLL_m.iterateSimulationAPI.restype = None
        self.simDLL_m.freeGPUMemoryAPI.argtypes = (ctypes.c_void_p,)
        self.simDLL_m.freeGPUMemoryAPI.restype = None
        self.simDLL_m.saveDataToDiskAPI.argtypes = (ctypes.c_void_p,)
        self.simDLL_m.saveDataToDiskAPI.restype = None
        self.simDLL_m.terminateSimulationAPI.argtypes = (ctypes.c_void_p,)
        self.simDLL_m.terminateSimulationAPI.restype = None
        self.simDLL_m.loadCompletedSimDataAPI.argtypes = (ctypes.c_char_p,)
        self.simDLL_m.loadCompletedSimDataAPI.restype = ctypes.c_void_p
        self.simDLL_m.setupExampleSimulationAPI.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p)
        self.simDLL_m.setupExampleSimulationAPI.restype = None
        self.simDLL_m.setupSingleElectronAPI.argtypes = (ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double)
        self.simDLL_m.setupSingleElectronAPI.restype = None


        #Field Management Functions
        self.simDLL_m.getBFieldAtSAPI.argtypes = (ctypes.c_void_p, ctypes.c_double, ctypes.c_double)
        self.simDLL_m.getBFieldAtSAPI.restype = ctypes.c_double
        self.simDLL_m.getEFieldAtSAPI.argtypes = (ctypes.c_void_p, ctypes.c_double, ctypes.c_double)
        self.simDLL_m.getEFieldAtSAPI.restype = ctypes.c_double
        self.simDLL_m.setBFieldModelAPI.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p)
        self.simDLL_m.setBFieldModelAPI.restype = None
        self.simDLL_m.addEFieldModelAPI.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p)
        self.simDLL_m.addEFieldModelAPI.restype = None


        #Particles Management Functions
        self.simDLL_m.createParticlesTypeAPI.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_double, ctypes.c_long, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_char_p)
        self.simDLL_m.createParticlesTypeAPI.restype = None


        #Satellite Management Functions
        self.simDLL_m.createSatelliteAPI.argtypes = (ctypes.c_void_p, ctypes.c_double, ctypes.c_bool, ctypes.c_char_p, ctypes.c_int)
        self.simDLL_m.createSatelliteAPI.restype = None
        self.simDLL_m.getNumberOfSatellitesAPI.argtypes = (ctypes.c_void_p,)
        self.simDLL_m.getNumberOfSatellitesAPI.restype = ctypes.c_int
        self.simDLL_m.getSatelliteDataPointersAPI.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
        self.simDLL_m.getSatelliteDataPointersAPI.restype = ctypes.POINTER(ctypes.c_double)
        

        #Access Functions
        self.simDLL_m.getSimTimeAPI.argtypes = (ctypes.c_void_p,)
        self.simDLL_m.getSimTimeAPI.restype = ctypes.c_double
        self.simDLL_m.getDtAPI.argtypes = (ctypes.c_void_p,)
        self.simDLL_m.getDtAPI.restype = ctypes.c_double
        self.simDLL_m.getSimMinAPI.argtypes = (ctypes.c_void_p,)
        self.simDLL_m.getSimMinAPI.restype = ctypes.c_double
        self.simDLL_m.getSimMaxAPI.argtypes = (ctypes.c_void_p,)
        self.simDLL_m.getSimMaxAPI.restype = ctypes.c_double
        self.simDLL_m.getNumberOfParticleTypesAPI.argtypes = (ctypes.c_void_p,)
        self.simDLL_m.getNumberOfParticleTypesAPI.restype = ctypes.c_int
        self.simDLL_m.getNumberOfParticlesAPI.argtypes = (ctypes.c_void_p, ctypes.c_int)
        self.simDLL_m.getNumberOfParticlesAPI.restype = ctypes.c_int
        self.simDLL_m.getNumberOfAttributesAPI.argtypes = (ctypes.c_void_p, ctypes.c_int)
        self.simDLL_m.getNumberOfAttributesAPI.restype = ctypes.c_int
        self.simDLL_m.getParticlesNameAPI.argtypes = (ctypes.c_void_p, ctypes.c_int)
        self.simDLL_m.getParticlesNameAPI.restype = ctypes.c_char_p
        self.simDLL_m.getSatelliteNameAPI.argtypes = (ctypes.c_void_p, ctypes.c_int)
        self.simDLL_m.getSatelliteNameAPI.restype = ctypes.c_char_p
        self.simDLL_m.getPointerToParticlesAttributeArrayAPI.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_bool)
        self.simDLL_m.getPointerToParticlesAttributeArrayAPI.restype = ctypes.POINTER(ctypes.c_double)


        #CSV
        self.simDLL_m.writeCommonCSVAPI.argtypes = (ctypes.c_void_p,)
        self.simDLL_m.writeCommonCSVAPI.restype = None