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

namespace utils
{
	namespace fileIO
	{
		class CSV
		{
		private:
			string filename_m;
			
			fp2Dvec data_m;
			strvec labels_m;
			bool write_m{ true };

			void write(); //defined in cpp

		public:
			CSV(string filename);
			~CSV();

			void add(fp1Dvec vec, string label);
			void add(fp2Dvec vecs, strvec labels);
			void addspace();
			void dontwrite();

			fp2Dvec& data();
		};

		class ParticleDistribution
		{
		public:
			struct Range; //energy || pitchAngle, range min, range max, step

		protected:
			string saveFolder_m;
			strvec attrNames_m;
			string particleName_m;
			bool   write_m{ true }; //set to false if deserialized, used to prevent overwriting a valid file
			kg     mass_m{ -1.0f };

			vector<Range>  ranges_m;
			fp1Dvec        padvals_m; //pad values for each attribute (usually 0.0 or -1.0)

			void deserialize(string serialFolder, string name);

		public: //generate is dependent on vpara, vperp, and s being the first three attributes - if not, this will have to be modified
			ParticleDistribution(string saveFolder, strvec attrNames = { "vpara", "vperp", "s", "t_inc", "t_esc" }, string particleName = "elec", kg mass = MASS_ELECTRON, fp1Dvec padvals = { 0.0f, 0.0f, 0.0f, 0.0f, -1.0f}, bool write = true);
			ParticleDistribution(string serialFolder, string name);
			ParticleDistribution(const ParticleDistribution& PD);
			~ParticleDistribution(); //writes on destruction

			void   printRanges() const;
			const vector<Range>& ranges() const;
			string saveFolder() const;
			strvec attrNames() const;
			string particleName() const;
			kg     mass() const;

			void    addEnergyRange(size_t energyBins, eV E_start, eV E_end, bool logE = true);
			void    addPitchRange(size_t pitchBins, degrees PA_start, degrees PA_end, bool midBin = true);
			fp2Dvec generate(meters s_ion, meters s_mag) const;
			fp2Dvec generate(vector<meters>& s) const;
			void    write(meters s_ion, meters s_mag) const;
			void    write(vector<meters>& s) const;
			void    serialize() const;
		};
	}
}

#endif /* !UTILS_IOCLASSES_WRITE_H */