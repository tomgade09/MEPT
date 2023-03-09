import os,sys
import ctypes

sys.path.append(os.path.normpath('./../'))
from __simulationvariables import *
simDLL = ctypes.CDLL(DLLLOCATION)

#ParticleDistribution functions
simDLL.PDCreateAPI.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double)
simDLL.PDCreateAPI.restype = ctypes.c_void_p
simDLL.PDAddEnergyRangeAPI.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_bool)
simDLL.PDAddEnergyRangeAPI.restype = None
simDLL.PDAddPitchRangeAPI.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_bool)
simDLL.PDAddPitchRangeAPI.restype = None
simDLL.PDWriteAPI.argtypes = (ctypes.c_void_p, ctypes.c_double, ctypes.c_double)
simDLL.PDWriteAPI.restype = None
simDLL.PDDeleteAPI.argtypes = (ctypes.c_void_p,)
simDLL.PDDeleteAPI.restype = None

#DistributionFromDisk functions
simDLL.DFDLoadAPI.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_double)
simDLL.DFDLoadAPI.restype = ctypes.c_void_p
simDLL.DFDDataAPI.argtypes = (ctypes.c_void_p, ctypes.c_int)
simDLL.DFDDataAPI.restype = ctypes.POINTER(ctypes.c_double)
simDLL.DFDPrintAPI.argtypes = (ctypes.c_void_p, ctypes.c_int)
simDLL.DFDPrintAPI.restype = None
simDLL.DFDPrintDiffAPI.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)
simDLL.DFDPrintDiffAPI.restype = None
simDLL.DFDZeroesAPI.argtypes = (ctypes.c_void_p,)
simDLL.DFDZeroesAPI.restype = None
simDLL.DFDCompareAPI.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
simDLL.DFDCompareAPI.restype = None
simDLL.DFDDeleteAPI.argtypes = (ctypes.c_void_p,)
simDLL.DFDDeleteAPI.restype = None