#ifndef UTILS_API_H
#define UTILS_API_H

//These utilities were rarely used, so I didn't feel the need to update
//Included in the code base in case it's needed in the future

//#include "dlldefines.h"
//#include "utils/fileIO.h"

//typedef utils::fileIO::ParticleDistribution PD;
//typedef utils::fileIO::DistributionFromDisk DFD;

//ParticleDistribution functions
//DLLEXP_EXTC PD*  PDCreateAPI(const char* saveFolder, const char* attrNames, const char* particleName, double mass);
//DLLEXP_EXTC void PDAddEnergyRangeAPI(PD* pd, int energyBins, double Emin, double Emax, bool logE = true);
//DLLEXP_EXTC void PDAddPitchRangeAPI(PD* pd, int pitchBins, double PAmin, double PAmax, bool midBin = true);
//DLLEXP_EXTC void PDWriteAPI(PD* pd, double s_ion, double s_mag);
//DLLEXP_EXTC void PDDeleteAPI(PD* pd);

//DistributionFromDisk functions
//DLLEXP_EXTC DFD*          DFDLoadAPI(const char* name, const char* loadFolder, const char* attrNames, const char* particleName, double mass);
//DLLEXP_EXTC const double* DFDDataAPI(DFD* dfd, int attrInd);
//DLLEXP_EXTC void          DFDPrintAPI(DFD* dfd, int at);
//DLLEXP_EXTC void          DFDPrintDiffAPI(DFD* dfd_this, DFD* dfd_other, int at);
//DLLEXP_EXTC void          DFDZeroesAPI(DFD* dfd);
//DLLEXP_EXTC void          DFDCompareAPI(DFD* dfd_this, DFD* dfd_other);
//DLLEXP_EXTC void          DFDDeleteAPI(DFD* dfd);

#endif /* UTILS_API_H */