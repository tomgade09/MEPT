#include "utils/readIOclasses.h"
#include "utils/fileIO.h"
#include "utils/strings.h"
#include "utils/numerical.h"

#include <iostream>
#include <cmath>

namespace utils
{
	namespace fileIO
	{
		DistributionFromDisk::DistributionFromDisk(string name, string folder, string partName, vector<string> attrNames, double mass) : 
			name_m{ name }, attrNames_m{ attrNames }, mass_m{ mass }
		{
			size_t attrsize{ 0 };
			for (size_t attr = 0; attr < attrNames.size(); attr++)
			{
				vector<double> read;
				fileIO::readDblBin(read, folder + "/" + partName + "_" + attrNames.at(attr) + ".bin");
				data_m.push_back(read);
				if (attrNames_m.at(attr).size() > attrsize) { attrsize = attrNames_m.at(attr).size(); }
			}

			for (size_t attr = 0; attr < attrNames.size(); attr++)
				if (attrNames_m.at(attr).size() < attrsize) { strings::stringPadder(attrNames_m.at(attr), attrsize); }
		}

		void DistributionFromDisk::print(int at) const
		{
			std::cout << name_m << " ";
			for (size_t iii = 0; iii < attrNames_m.size(); iii++)
				std::cout << attrNames_m.at(iii) << ((iii != attrNames_m.size() - 1) ? ", " : ": ");
			for (size_t iii = 0; iii < data_m.size(); iii++)
				std::cout << data_m.at(iii).at(at) << ((iii != data_m.size() - 1) ? ", " : "");
			std::cout << std::endl;

			vector<double> E{ { 0.0 } };
			vector<double> Pitch{ { 0.0 } };
			numerical::v2DtoEPitch({ data_m.at(0).at(at) }, { data_m.at(1).at(at) }, mass_m, E, Pitch);
			std::cout << "E, Pitch: " << E.at(0) << ", " << Pitch.at(0) << "\n";
		}

		void DistributionFromDisk::printdiff(DistributionFromDisk& other, int at) const
		{
			size_t datasize{ data_m.size() };
			if (data_m.size() != other.data().size())
			{
				std::cout << "DistributionFromDisk::printdiff: Warning: data from the two distributions does not have the same dimensionality. ";
				std::cout << "Did you load up two distributions of different types? Using the smaller size." << std::endl;
				datasize = ((data_m.size() < other.data().size()) ? (data_m.size()) : (other.data().size()));
			}

			vector<double> err(datasize);
			for (size_t attr = 0; attr < datasize; attr++)
			{
				err.at(attr) = std::abs((data_m.at(attr).at(at) - other.data().at(attr).at(at)) / data_m.at(attr).at(at));
				std::cout << attrNames_m.at(attr) << " (" << name_m << ", " << other.name() << ", err): ";
				std::cout << data_m.at(attr).at(at) << ", " << other.data().at(attr).at(at) << ", " << err.at(attr) << std::endl;
			}
		}

		void DistributionFromDisk::zeroes() const
		{
			vector<int> tmp;
			zeroes(tmp, true);
		}

		void DistributionFromDisk::zeroes(vector<int>& zeroes, bool print) const //print defaults to true
		{
			if (zeroes.size() != data_m.size())
			{
				zeroes.resize(data_m.size());
			}
			
			for (auto attr = data_m.begin(); attr < data_m.end(); attr++)
			{
				for (auto part = (*attr).begin(); part < (*attr).end(); part++)
					if ((*part) == 0.0) { zeroes.at(attr - data_m.begin())++; }
			}
			
			if (print)
			{
				std::cout << name_m << " ";
				for (size_t zero = 0; zero < zeroes.size(); zero++)
					std::cout << attrNames_m.at(zero) << " zeroes: " << zeroes.at(zero) << ((zero != zeroes.size() - 1) ? ", " : "");
				std::cout << std::endl;
			}
		}

		void DistributionFromDisk::compare(const DistributionFromDisk& other) const
		{
			const vector<vector<double>>& data_other{ other.data() };
			int datasize{ (int)data_m.size() };
			if (data_m.size() != data_other.size())
			{
				std::cout << "DistributionFromDisk::compare: Warning: data from the two distributions does not have the same dimensionality. ";
				std::cout << "Did you load up two distributions of different types? Using the smaller size." << std::endl;
				datasize = (data_m.size() < data_other.size()) ? ((int)data_m.size()) : ((int)data_other.size());
			}

			vector<int> zeroes_this;
			vector<int> zeroes_other;
			zeroes(zeroes_this, false);
			other.zeroes(zeroes_other, false);

			vector<double> mean_this;
			vector<double> mean_other;
			vector<double> stdDev_this;
			vector<double> stdDev_other;

			for (int iii = 0; iii < datasize; iii++)
			{
				mean_this.push_back(numerical::calcMean(data_m.at(iii)));
				mean_other.push_back(numerical::calcMean(data_other.at(iii)));
				stdDev_this.push_back(numerical::calcStdDev(data_m.at(iii)));
				stdDev_other.push_back(numerical::calcStdDev(data_other.at(iii)));
			}

			vector<int> notsame(datasize);
			vector<double> avgErr(datasize);
			vector<double> minErr(datasize);
			vector<double> maxErr(datasize);
			for (int attr = 0; attr < datasize; attr++)
			{
				int partsize{ (int)data_m.at(attr).size() };
				if (data_m.at(attr).size() != data_other.at(attr).size()) { std::cout << "DistributionFromDisk::compare: Warning: attributes have different number of particles.  Using smaller number" << std::endl;
					partsize = (data_m.at(attr).size() < data_other.at(attr).size()) ? ((int)data_m.at(attr).size()) : ((int)data_other.at(attr).size());	}
				for (int part = 0; part < partsize; part++)
				{
					if (data_m.at(attr).at(part) != data_other.at(attr).at(part))
					{
						double err{ std::abs((data_m.at(attr).at(part) - data_other.at(attr).at(part)) / data_m.at(attr).at(part)) };
						if (minErr.at(attr) > err) { minErr.at(attr) = err; }
						if (maxErr.at(attr) < err) { maxErr.at(attr) = err; }
						avgErr.at(attr) = (avgErr.at(attr) * notsame.at(attr) + err) / (notsame.at(attr) + 1);
						notsame.at(attr)++;
					}
				}
			}

			std::cout.precision(8);
			std::cout << "================ Summary of differences: " + name_m + ", " + other.name() + " ================" << std::endl;
			std::cout << "Number of zeroes in attributes:" << std::endl << std::scientific;
			for (int attr = 0; attr < datasize; attr++)
				std::cout << attrNames_m.at(attr) << ": " << zeroes_this.at(attr) << ", " << zeroes_other.at(attr) << std::endl;
			std::cout << std::endl;

			std::cout << "Attribute means:" << std::endl;
			for (int attr = 0; attr < datasize; attr++)
				std::cout << attrNames_m.at(attr) << ": " << mean_this.at(attr) << ", " << mean_other.at(attr) << std::endl;
			std::cout << std::endl;

			std::cout << "Attribute standard deviations:" << std::endl;
			for (int attr = 0; attr < datasize; attr++)
				std::cout << attrNames_m.at(attr) << ": " << stdDev_this.at(attr) << ", " << stdDev_other.at(attr) << std::endl;
			std::cout << std::endl;

			std::cout << "Attribute error ( abs((" + name_m + " - " + other.name() + ") / " + name_m + "):" << std::endl;
			for (int attr = 0; attr < datasize; attr++)
				std::cout << attrNames_m.at(attr) << ": Min: " << minErr.at(attr) << ", Max: " << maxErr.at(attr) << ", Avg: " << avgErr.at(attr) << ", Number not same: " << notsame.at(attr) << std::endl;
			std::cout << std::endl;
		}
	}
}
