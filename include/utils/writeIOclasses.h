#ifndef UTILS_IOCLASSES_WRITE_H
#define UTILS_IOCLASSES_WRITE_H

#include <vector>
#include <string>
#include <memory>
#include <stdexcept>

#include "dlldefines.h"
#include "physicalconstants.h"
#include "utils/unitsTypedefs.h"

using std::string;
using std::vector;
using std::unique_ptr;
using std::logic_error;

#define STRVEC vector<string>

namespace utils
{
	namespace fileIO
	{
		class CSV
		{
		private:
			string filename_m;
			
			vector<vector<double>> data_m;
			STRVEC labels_m;
			bool write_m{ true };

			void write(); //defined in cpp

		public:
			CSV(string filename);
			~CSV();

			void add(vector<double> vec, string label);
			void add(vector<vector<double>> vecs, vector<string> labels);
			void addspace();
			void dontwrite();

			vector<vector<double>>& data();
		};

		class ParticleDistribution
		{
		public:
			struct Range; //energy || pitchAngle, range min, range max, step

		protected:
			string saveFolder_m;
			STRVEC attrNames_m;
			string particleName_m;
			bool   write_m{ true }; //set to false if deserialized, used to prevent overwriting a valid file
			double mass_m{ -1.0 };

			vector<Range>  ranges_m;
			vector<double> padvals_m; //pad values for each attribute (usually 0.0 or -1.0)

			void deserialize(string serialFolder, string name);

		public: //generate is dependent on vpara, vperp, and s being the first three attributes - if not, this will have to be modified
			ParticleDistribution(string saveFolder, vector<string> attrNames = { "vpara", "vperp", "s", "t_inc", "t_esc" }, string particleName = "elec", double mass = MASS_ELECTRON, vector<double> padvals = { 0.0, 0.0, 0.0, 0.0, -1.0}, bool write = true);
			ParticleDistribution(string serialFolder, string name);
			ParticleDistribution(const ParticleDistribution& PD);
			~ParticleDistribution(); //writes on destruction

			void   printRanges() const;
			const vector<Range>& ranges() const;
			string saveFolder() const;
			STRVEC attrNames() const;
			string particleName() const;
			double mass() const;

			void addEnergyRange(size_t energyBins, eV E_start, eV E_end, bool logE = true);
			void addPitchRange(size_t pitchBins, degrees PA_start, degrees PA_end, bool midBin = true);
			vector<vector<double>> generate(meters s_ion, meters s_mag) const;
			vector<vector<double>> generate(vector<meters>& s) const;
			void write(meters s_ion, meters s_mag) const;
			void write(vector<meters>& s) const;
			void serialize() const;
		};
	}
}

#undef STRVEC

#endif /* !UTILS_IOCLASSES_WRITE_H */