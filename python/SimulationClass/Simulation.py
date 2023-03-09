import ctypes, os, sys, inspect

sys.path.append(os.path.dirname(os.path.abspath(inspect.getsourcefile(lambda:0)) + './../'))
from __simulationvariables import *
import _Simulation

class Simulation(_Simulation._SimulationCDLL):
    def __init__(self, DLLloc, savedir, dt=None, simMin=None, simMax=None): ### Python ctor, dtor ###
        super().__init__(DLLloc)
        self.savedir_m = savedir
        self.dt_m = dt
        self.simMin_m = simMin
        self.simMax_m = simMax

        self.numPartTypes_m = 0
        self.numSats_m = 0
        self.numParts_m = []
        self.nameParts_m = []
        self.numAttrs_m = []
        self.nameSat_m = []

        self.origData_m  =  [[]]
        self.finalData_m =  [[]]
        self.satData_m   = [[[]]]

        savedir_c = ctypes.create_string_buffer(bytes(self.savedir_m, encoding='utf-8'))
        self.cppSimPtr_m = ctypes.c_void_p

        if dt is not None:
            self.cppSimPtr_m = self.simDLL_m.createSimulationAPI(dt, simMin, simMax, savedir_c)
        else:
            self.cppSimPtr_m = self.simDLL_m.loadCompletedSimDataAPI(savedir_c)
            self.__getSimChars()

            self.finalData_m = self.getFinalDataAllParticles()
            self.origData_m  = self.getOriginalDataAllParticles()
            #self.satData_m   = self.getSatelliteData()

        return


    def __del__(self):
        self.__terminateSimulation()
        return
    ### End ctor, dtor ###

    ### Python-Exclusive Functions ###
    def run(self, iterations, iterBtwCouts, pullData=False):
        if self.numAttrs_m == []:
            self.__getSimChars()

        self.simDLL_m.initializeSimulationAPI(self.cppSimPtr_m)
        self.simDLL_m.iterateSimulationAPI(self.cppSimPtr_m, iterations, iterBtwCouts)
        
        if (pullData):  #Returns final particle data, original particle data, satellite data
            self.finalData_m = self.getFinalDataAllParticles()
            self.origData_m  = self.getOriginalDataAllParticles()
            self.satData_m   = self.getSatelliteData()
        
        return
    
    def runCPU(self, iterations, iterBtwCouts, pullData=False):
        if self.numAttrs_m == []:
            self.__getSimChars()

        self.simDLL_m.initializeSimulationAPI(self.cppSimPtr_m)
        self.simDLL_m.iterateSimCPUAPI(self.cppSimPtr_m, iterations, iterBtwCouts)

        if (pullData):  #Returns final particle data, original particle data, satellite data
            self.finalData_m = self.getFinalDataAllParticles()
            self.origData_m  = self.getOriginalDataAllParticles()
            self.satData_m   = self.getSatelliteData()


    def __getParticleDataFromCPP(self, origData=False):
        ret = []
        partattr = []
        partdbl = []
        for iii in range(self.numPartTypes_m):
            for jjj in range(self.numAttrs_m[iii]):
                partdbl_c = self.simDLL_m.getPointerToParticlesAttributeArrayAPI(self.cppSimPtr_m, iii, jjj, origData)
                for kk in range(self.numParts_m[iii]):
                   partdbl.append(partdbl_c[kk])
                partattr.append(partdbl)
                partdbl = []
            ret.append(partattr)
            partattr = []
        return ret

    def getOriginalDataAllParticles(self):
        return self.__getParticleDataFromCPP(True)

    def getFinalDataAllParticles(self):
        return self.__getParticleDataFromCPP()

    def getFieldsAtAllS(self, time, bins, binsize, z0):
        B_z = []
        E_z = []
        B_E_z_dim = []
        for iii in range(bins):
            B_z.append(self.getBFieldatS(z0 + binsize * iii, time))
            E_z.append(self.getEFieldatS(z0 + binsize * iii, time))
            B_E_z_dim.append(z0 + binsize * iii)
        return [B_z, E_z, B_E_z_dim]

    def getSatelliteData(self):
        if self.numAttrs_m == []:
            self.__getSimChars()

        satptr = [] #constructs array of double pointers so z value can be checked before recording data
        for jjj in range(self.numSats_m):
            attrptr = []
            for kk in range(self.numAttrs_m[0] + 2):
                attrptr.append(self.simDLL_m.getSatelliteDataPointersAPI(self.cppSimPtr_m, jjj, kk))
            satptr.append(attrptr)
        
        satsdata = []
        for jjj in range(self.numSats_m):
            attr = []
            for kk in range(self.numAttrs_m[0] + 2):
                parts=[]
                for lll in range(self.numParts_m[0]):
                    parts.append(satptr[jjj][kk][lll])
                attr.append(parts)
            satsdata.append(attr)
        
        return satsdata


    ### API Function Callers ###

    ## Simulation Management Functions
    # Generally most of these shouldn't have to be called on their own.  Use run(args) instead
    # But they are here if you need them  
    def __initializeSimulation(self):
        self.simDLL_m.initializeSimulationAPI(self.cppSimPtr_m)

    def __iterateSimCPU(self, numberOfIterations, itersBtwCouts):
        print("Number Of Iterations: ", numberOfIterations)
        self.simDLL_m.iterateSimCPUAPI(self.cppSimPtr_m, numberOfIterations, itersBtwCouts)

    def __iterateSimulation(self, numberOfIterations, itersBtwCouts):
        print("Number Of Iterations: ", numberOfIterations)
        self.simDLL_m.iterateSimulationAPI(self.cppSimPtr_m, numberOfIterations, itersBtwCouts)

    def __freeGPUMemory(self):
        self.simDLL_m.freeGPUMemoryAPI(self.cppSimPtr_m)

    def __saveDataToDisk(self):
        self.simDLL_m.saveDataToDiskAPI(self.cppSimPtr_m)

    def __terminateSimulation(self):
        self.simDLL_m.terminateSimulationAPI(self.cppSimPtr_m)

    def setupExampleSim(self, numParts):
        if (LOADDIST):
            loadFileBuf = ctypes.create_string_buffer(bytes(DISTINFOLDER, encoding='utf-8'))
        else:
            loadFileBuf = ctypes.create_string_buffer(bytes("", encoding='utf-8'))

        self.simDLL_m.setupExampleSimulationAPI(self.cppSimPtr_m, numParts, loadFileBuf)
        self.__getSimChars()

    def setupSingleElectron(self, vpara, vperp, s, t_inc):
        self.simDLL_m.setupSingleElectronAPI(self.cppSimPtr_m, vpara, vperp, s, t_inc)


    ## Field Management Functions
    def getBFieldatS(self, s, time):
        return self.simDLL_m.getBFieldAtSAPI(self.cppSimPtr_m, s, time)

    def getEFieldatS(self, s, time):
        return self.simDLL_m.getEFieldAtSAPI(self.cppSimPtr_m, s, time)

    def setBFieldModel(self, name, doublesString):
        name_c = ctypes.create_string_buffer(bytes(name, encoding='utf-8'))
        doublesString_c = ctypes.create_string_buffer(bytes(doublesString, encoding='utf-8'))
        self.simDLL_m.setBFieldModelAPI(self.cppSimPtr_m, name_c, doublesString_c)

    def addEFieldModel(self, name, doublesString):
        name_c = ctypes.create_string_buffer(bytes(name, encoding='utf-8'))
        doublesString_c = ctypes.create_string_buffer(bytes(doublesString, encoding='utf-8'))
        self.simDLL_m.addEFieldModelAPI(self.cppSimPtr_m, name_c, doublesString_c)


    ## Particles Management Functions
    def createParticles(self, name, attrNames, mass, charge, numParts, posDims, velDims, normFactor, loadFileDir=""):
        nameBuf = ctypes.create_string_buffer(bytes(name, encoding='utf-8'))
        attrNamesBuf = ctypes.create_string_buffer(bytes(attrNames, encoding='utf-8'))
        loadFileDirBuf = ctypes.create_string_buffer(bytes(loadFileDir, encoding='utf-8'))

        self.simDLL_m.createParticlesTypeAPI(self.cppSimPtr_m, nameBuf, attrNamesBuf, mass, charge, numParts, posDims, velDims, normFactor, loadFileDirBuf)

        self.numPartTypes_m += 1 #eventually check to see that C++ has created properly by calling Particles access functions or don't create a python one
        self.numAttrs_m.append(posDims + velDims)
        self.numParts_m.append(numParts)
        self.nameParts_m.append(name)


    ## Satellite Management Functions
    def createSatellite(self, altitude, upwardFacing, name, particleCount):
        nameBuf = ctypes.create_string_buffer(bytes(name, encoding='utf-8'))
        self.simDLL_m.createSatelliteAPI(self.cppSimPtr_m, altitude, upwardFacing, nameBuf, particleCount)
        self.nameSat_m.append(name)

    def getNumberOfSatellites(self):
        return self.simDLL_m.getNumberOfSatellitesAPI(self.cppSimPtr_m)


    ## Access Functions
    def getTime(self):
        return self.simDLL_m.getSimTimeAPI(self.cppSimPtr_m)

    def __getSimChars(self): #call this after the simulation is set up
        if not self.numAttrs_m == []:
            return

        self.dt_m = self.simDLL_m.getDtAPI(self.cppSimPtr_m)
        self.simMin_m = self.simDLL_m.getSimMinAPI(self.cppSimPtr_m)
        self.simMax_m = self.simDLL_m.getSimMaxAPI(self.cppSimPtr_m)
        
        self.numPartTypes_m = self.simDLL_m.getNumberOfParticleTypesAPI(self.cppSimPtr_m)
        for part in range(self.numPartTypes_m):
            self.numAttrs_m.append(self.simDLL_m.getNumberOfAttributesAPI(self.cppSimPtr_m, part))
            self.numParts_m.append(self.simDLL_m.getNumberOfParticlesAPI(self.cppSimPtr_m, part))
            self.nameParts_m.append(self.simDLL_m.getParticlesNameAPI(self.cppSimPtr_m, part))
        
        self.numSats_m = self.simDLL_m.getNumberOfSatellitesAPI(self.cppSimPtr_m)
        for sat in range(self.numSats_m):
            self.nameSat_m.append(self.simDLL_m.getSatelliteNameAPI(self.cppSimPtr_m, sat))
    

    ## CSV Function
    def writeCommonCSV(self):
        self.simDLL_m.writeCommonCSVAPI(self.self.cppSimPtr_m)


if __name__ == '__main__':
    print("SimulationAPI.py is not meant to be called as main.  Run simulation.py and that will import this.")
