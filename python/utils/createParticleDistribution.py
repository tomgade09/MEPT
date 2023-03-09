#Create out directory if it doesn't exist
import os
OUTDIR = os.path.abspath("./out")
if (not(os.path.isdir(OUTDIR))):
    os.makedirs(OUTDIR)

from utilsAPI import *

MASS_ELEC = 9.10938356e-31

#Define C-style char arrays that need to be passed into functions
save_cstr = ctypes.create_string_buffer(bytes("./out/", encoding='utf-8'))
attr_cstr = ctypes.create_string_buffer(bytes("vpara,vperp,s,t_inc,t_esc", encoding='utf-8'))
part_cstr = ctypes.create_string_buffer(bytes("elec", encoding='utf-8'))

zero_cstr = ctypes.create_string_buffer(bytes("t_inc", encoding='utf-8'))
neg1_cstr = ctypes.create_string_buffer(bytes("t_esc", encoding='utf-8'))

#Create a ParticleDistribution instance in C++
PDptr = ctypes.c_void_p
PDptr = simDLL.PDCreateAPI(save_cstr, attr_cstr, part_cstr, MASS_ELEC)

#Define and write the distribution to disk
simDLL.PDAddEnergyRangeAPI(PDptr, 96, 0.5, 4.5, True)
simDLL.PDAddPitchRangeAPI(PDptr, 36000, 180.0, 0.0, True)
#simDLL.PDAddPitchRangeAPI(PDptr, 36000, 179.9975, 0.0025, False)
#simDLL.PDAddPitchRangeAPI(PDptr, 18000, 16.0 - 8.0/18000.0, 8.0/18000.0, False)
simDLL.PDWriteAPI(PDptr, 628565.8510817, 19881647.2473464)
simDLL.PDDeleteAPI(PDptr)

PDptr = None