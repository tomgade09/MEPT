from __future__ import absolute_import, division, print_function

import os, sys, inspect
a = os.path.dirname(os.path.abspath(inspect.getsourcefile(lambda:0)))
os.chdir(a)

import numpy as np
import ctypes
import math
from time import sleep

dllLocation = './fileIO.dll'

#DLLEXPORT double* readDblBin(const std::string filename, int numOfDblsToRead)
def readBDataWrapper(filename, numDblsToRead):
    fileIO = ctypes.CDLL(dllLocation)
    fileIO.readDblBin.argtypes = (ctypes.c_char_p, ctypes.c_int)
    fileIO.readDblBin.restype = ctypes.POINTER(ctypes.c_double)

    return fileIO.readDblBin(filename, numDblsToRead)

#DLLEXPORT double* interpolateB3DPyWrapper(double* pos, double* xrng, double* yrng, double* zrng, double binSize)//, double* dataArray)
def interpolateB3DWrapper(pos, xrange, yrange, zrange, binSize, dataArray, numDblsToRead):
    fileIO = ctypes.CDLL(dllLocation)
    fileIO.interpolateB3DPyWrapper.argtypes = (ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), 
                                               ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), 
                                               ctypes.c_double)#, ctypes.POINTER(ctypes.c_double))
    fileIO.interpolateB3DPyWrapper.restype = ctypes.POINTER(ctypes.c_double)

    C_DOUBLE3A = ctypes.c_double * 3
    C_DOUBLE2A = ctypes.c_double * 2

    posc = C_DOUBLE3A()
    xrangec = C_DOUBLE2A()
    yrangec = C_DOUBLE2A()
    zrangec = C_DOUBLE2A()

    for iii in range(3):
        posc[iii] = pos[iii]
        if (iii != 2):
            xrangec[iii] = xrange[iii]
            yrangec[iii] = yrange[iii]
            zrangec[iii] = zrange[iii]

    ret = fileIO.interpolateB3DPyWrapper(posc, xrangec, yrangec, zrangec, binSize, dataArray)

    return [ret[0],ret[1],ret[2]]

#DLLEXPORT void clrDataMemory(double* dataArray)
def freeDataArrayMemory(dataArray, numDblsToRead):
    fileIO = ctypes.CDLL(dllLocation)
    fileIO.clrDataMemory.argtypes = (ctypes.POINTER(ctypes.c_double),)
    fileIO.clrDataMemory.restype = None

    fileIO.clrDataMemory(dataArray)

#DLLEXPORT int indexLambdaFcn(int xsteps, int ysteps, int zsteps, int xsize=(1+10/0.04), int ysize=(1+10/0.04), int zsize=(1+10/0.04))
def indexFromSteps(xsteps, ysteps, zsteps):
    fileIO = ctypes.CDLL(dllLocation)
    fileIO.indexLambdaFcn.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
    fileIO.indexLambdaFcn.restype = ctypes.c_int

    return fileIO.indexLambdaFcn(xsteps, ysteps, zsteps, 251, 251, 251)

################For unloading the DLL after use - example from stack overflow###############

#import ctypes

# get the module handle and create a ctypes library object
#libHandle = ctypes.windll.kernel32.LoadLibraryA('mydll.dll')
#lib = ctypes.WinDLL(None, handle=libHandle) #WinDLL would likely be CDLL

# do stuff with lib in the usual way
#lib.Foo(42, 666)

# clean up by removing reference to the ctypes library object
#del lib

# unload the DLL
#ctypes.windll.kernel32.FreeLibrary(libHandle)

#################Test code based on example - try it out

#def loadDLL(dllLoc):
	#lib = ctypes.CDLL(dllLoc)

    #return lib

#def unloadDLL(libObject, fileName):
	#libHandle = libObject._handle
    #del libObject
	#ctypes.windll.kernel32.FreeLibrary(libHandle) #Windows only
    #ctypes.cdll.LoadLibrary(fileName).dlclose(libHandle) #*NIX only