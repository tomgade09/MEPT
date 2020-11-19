#ifndef SIMULATIONCLASSEXTENSIONS_H
#define SIMULATIONCLASSEXTENSIONS_H

#include "EField/EField.h"
#include "utils/fileIO.h"
#include "physicalconstants.h"

class AlfvenLUT : public EElem
{
protected:
	std::string LUTfilename_m;
	std::vector<std::vector<double>> EFieldLUT_m;

	double*   EFieldLUT1D_d{ nullptr }; //device (GPU) electric field LUT pointer
	double**  EFieldLUT2D_d{ nullptr };

	double  omegaE_m{ 20 * PI }; //angular frequency of wave - make sure this matches the LUT passed in
	int     numOfColsLUT_m{ 3 };
	int     numOfEntrLUT_m{ 2951 };

public:
	AlfvenLUT(double omegaE, int cols, int entrs, std::string LUTfilename, bool CSV = true, const char delim = ' ') :
		EField(), omegaE_m{ omegaE }, numOfColsLUT_m{ cols }, numOfEntrLUT_m{ entrs }, LUTfilename_m { LUTfilename }
	{
		for (int iii = 0; iii < cols; iii++)
			EFieldLUT_m.push_back(std::vector<double>(entrs));

		if (CSV)
			fileIO::read2DCSV(EFieldLUT_m, LUTfilename, entrs, cols, delim);
		else
		{
			std::vector<double> tmpSerial(cols * entrs);
			fileIO::readDblBin(tmpSerial, LUTfilename, cols * entrs);
			for (int col = 0; col < cols; col++)
			{
				for (int entr = 0; entr < entrs; entr++)
					EFieldLUT_m.at(col).at(entr) = tmpSerial.at(col * entrs + entr);
			}
		}
	}

	~AlfvenLUT() {}

	void setupEnvironment();

	//One liners
	std::vector<std::vector<double>> getElectricFieldLUT() { return EFieldLUT_m; }
};

#endif //end header guard
