/*
#include "API/utilsAPI.h"

#include "utils/strings.h"
#include "ErrorHandling/simExceptionMacros.h"

using std::string;
using std::vector;
using std::invalid_argument;
using utils::strings::strToStrVec;

//ParticleDistribution functions
DLLEXP_EXTC PD* PDCreateAPI(const char* saveFolder, const char* attrNames, const char* particleName, double mass)
{
	SIM_API_EXCEP_CHECK(
		string save{ saveFolder };
		vector<string> attrs{ strToStrVec(attrNames) };
		string part{ particleName };

		PD* ret{ nullptr };
		if (save == "") save = "./";
		if (attrs.size() == 0 && part == "" && mass == 0.0)
			ret = new PD(save);
		else
			ret = new PD(save, attrs, part, mass);

		return ret;
	);

	return nullptr; //if above fails
}

DLLEXP_EXTC void PDAddEnergyRangeAPI(PD* pd, int energyBins, double Emin, double Emax, bool logE) {
	SIM_API_EXCEP_CHECK(pd->addEnergyRange(energyBins, Emin, Emax, logE)); }

DLLEXP_EXTC void PDAddPitchRangeAPI(PD* pd, int pitchBins, double PAmin, double PAmax, bool midBin) {
	SIM_API_EXCEP_CHECK(pd->addPitchRange(pitchBins, PAmin, PAmax, midBin)); }

DLLEXP_EXTC void PDWriteAPI(PD* pd, double s_ion, double s_mag) {
	pd->write(s_ion, s_mag); }

DLLEXP_EXTC void PDDeleteAPI(PD* pd) {
	delete pd; }


//DistributionFromDisk functions
DLLEXP_EXTC DFD* DFDLoadAPI(const char* name, const char* loadFolder, const char* attrNames, const char* particleName, double mass)
{
	SIM_API_EXCEP_CHECK(return new DFD(name, loadFolder, particleName, strToStrVec(attrNames), mass));
	return nullptr; //if above fails
}

DLLEXP_EXTC const double* DFDDataAPI(DFD* dfd, int attrInd)
{
	SIM_API_EXCEP_CHECK(return dfd->data().at(attrInd).data());
	return nullptr; //if above fails
}

DLLEXP_EXTC void DFDPrintAPI(DFD* dfd, int at) {
	SIM_API_EXCEP_CHECK(dfd->print(at)); }

DLLEXP_EXTC void DFDPrintDiffAPI(DFD* dfd_this, DFD* dfd_other, int at) {
	SIM_API_EXCEP_CHECK(dfd_this->printdiff(*dfd_other, at)); }

DLLEXP_EXTC void DFDZeroesAPI(DFD* dfd) {
	SIM_API_EXCEP_CHECK(dfd->zeroes()); }

DLLEXP_EXTC void DFDCompareAPI(DFD* dfd_this, DFD* dfd_other) {
	SIM_API_EXCEP_CHECK(dfd_this->compare(*dfd_other)); }

DLLEXP_EXTC void DFDDeleteAPI(DFD* dfd) {
	delete dfd; }

*/