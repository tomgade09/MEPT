#include <fstream>
#include <iostream>
#include <filesystem>

#include "utils/writeIOclasses.h"
#include "utils/fileIO.h"
#include "utils/numerical.h"
#include "utils/serializationHelpers.h"

using std::cerr;
using std::cout;
using std::endl;
using std::ofstream;
using std::exception;
using std::streamsize;
using std::make_unique;
using std::logic_error;
using std::invalid_argument;
using utils::numerical::generateSpacedValues;

using namespace utils::fileIO::serialize;

namespace utils
{
	namespace fileIO
	{
		//start CSV private functions
		void CSV::write() //private
		{
			if (data_m.size() == 0) return; //nothing to write to disk

			size_t largestdim{ 0 };
			for (size_t data = 0; data < data_m.size(); data++)
				if (data_m.at(data).size() > largestdim) largestdim = data_m.at(data).size();

			ofstream file(filename_m, std::ios::trunc);
			for (size_t iii = 0; iii < labels_m.size(); iii++)
				file << labels_m.at(iii) << ((iii != labels_m.size() - 1) ? "," : "\n");
			file.close();

			fileIO::write2DCSV(data_m, filename_m, largestdim, data_m.size(), ',', false);
			data_m.clear();
		}

		//public
		CSV::CSV(string filename) : filename_m{ filename }
		{

		}

		CSV::~CSV()
		{
			try
			{
				if (write_m) write();
			}
			catch (exception& e)
			{
				cerr << "CSV::~CSV: " << e.what() << ". CSV did not write to disk";
			}
		}

		void CSV::add(vector<float> vec, string label)
		{
			data_m.push_back(vec);
			labels_m.push_back(label);
		}

		void CSV::add(vector<vector<float>> vecs, vector<string> labels)
		{
			if (vecs.size() != labels.size())
				throw invalid_argument("CSV::add: vecs.size() != labels.size()");
			for (auto vec = vecs.begin(); vec < vecs.end(); vec++)
				add((*vec), labels.at(vec - vecs.begin()));
		}

		void CSV::addspace()
		{
			if (data_m.size() == 0)
			{
				cout << "CSV::addspace: data_m is empty, cannot add zeroes array\n";
				return;
			}
			data_m.push_back(vector<float>(data_m.at(0).size()));
			labels_m.push_back("");
		}

		vector<vector<float>>& CSV::data()
		{
			return data_m;
		}

		void CSV::dontwrite()
		{
			write_m = false;
		}



		// ======== ParticleDistribution::Private ======== //
		//struct ParticleDistribution::EPA
		struct EPA //not a member of ParticleDistribution - don't need outside this file
		{
			vector<eV> energies;
			vector<degrees> pitches;
		};
		
		struct ParticleDistribution::Range
		{
			enum class Type
			{
				energy,
				pitchAngle
			};

			Type   type_m{ Type::energy };
			float min_m { 0.0f };
			float max_m { 0.0f };
			float step_m{ 0.0f };
			int    bins_m{ 0 };
			bool   optn_m{ false }; //used to set logE || midBin options

			Range(Type type, float min, float max, float step, int bins, bool option) :
				type_m{ type }, min_m{ min }, max_m{ max }, step_m{ step }, bins_m{ bins }, optn_m{ option }
			{

			}

			Range(ifstream& in)
			{
				deserialize(in);
			}

			stringbuf serialize() const
			{
				stringbuf sb;
				ostream out(&sb);

				out.write(reinterpret_cast<const char*>(&type_m), sizeof(Type));
				out.write(reinterpret_cast<const char*>(&min_m), sizeof(float));
				out.write(reinterpret_cast<const char*>(&max_m), sizeof(float));
				out.write(reinterpret_cast<const char*>(&step_m), sizeof(float));
				out.write(reinterpret_cast<const char*>(&bins_m), sizeof(int));
				out.write(reinterpret_cast<const char*>(&optn_m), sizeof(bool));

				return sb;
			}

			void deserialize(ifstream& in)
			{
				auto readVal = [&](streamsize size)
				{
					vector<char> tmp(size, '\0');
					in.read(tmp.data(), size);
					return tmp;
				};

				type_m = *(reinterpret_cast<Type*>(readVal(sizeof(Type)).data()));
				min_m = *(reinterpret_cast<float*>(readVal(sizeof(float)).data()));
				max_m = *(reinterpret_cast<float*>(readVal(sizeof(float)).data()));
				step_m = *(reinterpret_cast<float*>(readVal(sizeof(float)).data()));
				bins_m = *(reinterpret_cast<int*>(readVal(sizeof(int)).data()));
				optn_m = *(reinterpret_cast<bool*>(readVal(sizeof(bool)).data()));
			}
		};


		// ======== ParticleDistribution::Public ======== //
		ParticleDistribution::ParticleDistribution(string saveFolder, vector<string> attrNames, string particleName, float mass, vector<float> padvals, bool write) :
			saveFolder_m{ saveFolder }, attrNames_m{ attrNames }, particleName_m{ particleName }, write_m{ write }, mass_m{ mass }, padvals_m{ padvals }
		{//ctor(default) for shorthand
			if (attrNames.size() == 0) throw invalid_argument("ParticleDistribution::ctor(default): invalid attributes - none specified");
			if (particleName == "") throw invalid_argument("ParticleDistribution::ctor(default): invalid name - none specified");
			if (mass <= 0.0f) throw invalid_argument("ParticleDistribution::ctor(default): invalid mass - <= 0.0f");
		}

		ParticleDistribution::ParticleDistribution(string serialFolder, string name) : write_m{ false }
		{//ctor(deserialize) for shorthand
			deserialize(serialFolder, name);
		}

		ParticleDistribution::ParticleDistribution(const ParticleDistribution& PD)
		{
			saveFolder_m = PD.saveFolder_m;
			attrNames_m = PD.attrNames_m;
			particleName_m = PD.particleName_m;
			write_m = false; //this is because a copy is being made - only one instance can have that disk location
			mass_m = PD.mass_m;

			ranges_m = PD.ranges_m;
			padvals_m = PD.padvals_m;
		}

		ParticleDistribution::~ParticleDistribution()
		{

		}

		void ParticleDistribution::printRanges() const
		{
			for (const auto& elem : ranges_m)
			{
				cout << ((elem.type_m == Range::Type::energy) ? "Energy: " : "Pitch:  ");
				cout << " [ " << elem.min_m << " (min) to " << elem.max_m << " (max) ] , step: " << elem.step_m;
				cout << " , num bins: " << elem.bins_m << " , ";
				cout << ((elem.type_m == Range::Type::energy) ? "logE: " : "midBin: ");
				cout << ((elem.optn_m) ? "true\n" : "false\n");
			}
		}

		//Access functions
		const vector<ParticleDistribution::Range>& ParticleDistribution::ranges() const
		{
			return ranges_m;
		}

		string ParticleDistribution::saveFolder() const
		{
			return saveFolder_m;
		}

		vector<string> ParticleDistribution::attrNames() const
		{
			return attrNames_m;
		}
		
		string ParticleDistribution::particleName() const
		{
			return particleName_m;
		}
		
		float ParticleDistribution::mass() const
		{
			return mass_m;
		}

		void ParticleDistribution::addEnergyRange(size_t energyBins, eV E_start, eV E_end, bool logE) //logE defaults to true
		{ //if logE is true, pass in logE_start and logE_end for E_start and E_end
		  //also E bins are mid bin and end point inclusive - |-x-|-x-|-x-|
		  //                                            E_start ^       ^ E_end

			if (energyBins == 0) throw invalid_argument("ParticleDistribution::addPitchRange: pitchBins is zero");
			float E_binsize{ (E_end - E_start) / (float)(energyBins - 1) }; //minus 1 because E bins are end point inclusive
			ranges_m.push_back(std::move(Range(Range::Type::energy, E_start, E_end, E_binsize, energyBins, logE)));
		}

		void ParticleDistribution::addPitchRange(size_t pitchBins, degrees PA_start, degrees PA_end, bool midBin) //midBin defaults to true
		{ //regardless of whether or not midBin is specified, pass in PAs as edges of the whole range
		  //          x---x---x---|          or          |-x-|-x-|-x-|
		  // PA_start ^           ^ PA_end      PA_start ^           ^ PA_end
			
			if (pitchBins == 0) throw invalid_argument("ParticleDistribution::addPitchRange: pitchBins is zero");
			float PA_binsize{ (PA_end - PA_start) / (float)pitchBins }; //no minus 1 because start and end are bin edges - not end point inclusive
			ranges_m.push_back(std::move(Range(Range::Type::pitchAngle, PA_start, PA_end, PA_binsize, pitchBins, midBin)));
		}

		vector<vector<float>> ParticleDistribution::generate(meters s_ion, meters s_mag) const
		{
			vector<meters> s({ s_ion, s_mag });

			return generate(s);
		}

		vector<vector<float>> ParticleDistribution::generate(vector<meters>& s) const
		{
			EPA epa;
			
			{ //generate every Energy and Pitch Angle, and iterate one over the other
				EPA tmp;

				for (const auto& range : ranges_m)
				{
					if (range.type_m == Range::Type::energy)
					{
						//defaults to creating mid bin values (x's ---->) This: |-x-|-x-|   Not this: x---x---|
						vector<float> E_add{ generateSpacedValues(range.min_m, range.max_m, range.bins_m, range.optn_m, true) };
						for (size_t eng = 0; eng < E_add.size(); eng++) //adds to whatever values are there now
							tmp.energies.push_back(E_add.at(eng));
					}
					else if (range.type_m == Range::Type::pitchAngle)
					{
						degrees PA_start{ (degrees)range.min_m };
						degrees PA_end  { (degrees)range.max_m };

						if (range.optn_m) //optn_m is midBin
						{ //if mid bin values are desired, adjust start and end range, and set endInclusive in generateSpacedValues
							PA_start += 0.5f * range.step_m;
							PA_end -= 0.5f * range.step_m;
						}

						//defaults to linear (non-log) scale for bins - This: x  x  x  x    Not this: xxx x x  x    x        x
						//if midBin is specified, range becomes end point inclusive : x-----x-----|  vs  |--x--|--x--|
						//                                                      start=^,end=^,not=^   start=^,end=^
						vector<float> PA_add{ generateSpacedValues(PA_start, PA_end, range.bins_m, false, range.optn_m) };
						for (size_t eng = 0; eng < PA_add.size(); eng++)
							tmp.pitches.push_back(PA_add.at(eng));
					}
					else throw invalid_argument("ParticleDistribution::generate(vector<meters>): \
						Range is not one of either possible types.");
				}

				if (tmp.energies.size() == 0 || tmp.pitches.size() == 0)
					throw invalid_argument("ParticleDistribution::generate(vector<meters>): \
						At least one of { energy, pitch } range is not specified. Cannot generate distribution.");

				for (const auto& pitch : tmp.pitches) //iterate energies over pitches, store in epa
				{
					for (const auto& energy : tmp.energies)
					{
						epa.energies.push_back(energy);
						epa.pitches.push_back(pitch);
					}
				}
			} //discard tmp

			if (s.size() == 2)
			{ //if s_ion and s_mag were passed in from generate(meters, meters)
				meters s_ion{ s.at(0) };
				meters s_mag{ s.at(1) };
				s.clear();

				for (const auto& pitch : epa.pitches)
					s.push_back(((pitch <= 90.0f) ? s_mag : s_ion));
			}
			
			vector<vector<float>> ret(attrNames_m.size(), vector<float>(epa.energies.size()));
			for (size_t otherAttr = 0; otherAttr < ret.size(); otherAttr++)
				if (padvals_m.at(otherAttr) != 0.0f) //pad vectors with non-zero pad values
					ret.at(otherAttr) = vector<float>(ret.at(0).size(), padvals_m.at(otherAttr));
			
			utils::numerical::EPitchTov2D(epa.energies, epa.pitches, mass_m, ret.at(0), ret.at(1));

			for (size_t attr = 0; attr < attrNames_m.size(); attr++)
			{
				if (attrNames_m.at(attr) == "s")
				{
					vector<float>sFloat(s.begin(), s.end());
					ret.at(attr) = sFloat;
					return ret;
				}
			}

			throw logic_error("ParticleDistribution::generate(vector<meters>): No attribute named 's'");
		}

		void ParticleDistribution::write(meters s_ion, meters s_mag) const
		{
			vector<meters> s({ s_ion, s_mag });

			write(s);
		}

		void ParticleDistribution::write(vector<meters>& s) const
		{
			vector<vector<float>> data{ generate(s) };

			try
			{
				for (size_t attr = 0; attr < data.size(); attr++)
					fileIO::writeFltBin(data.at(attr), saveFolder_m + "/" + particleName_m + "_" + attrNames_m.at(attr) + ".bin", data.at(0).size());
			}
			catch (exception& e)
			{
				cerr << "ParticleDistribution::~ParticleDistribution: " << e.what() << ". PD partially or entirely not written to disk";
			}
		}

		void ParticleDistribution::serialize() const
		{
			string filename{ saveFolder_m + string("/") + particleName_m + string(".ser") };

			if (std::filesystem::exists(filename))
				cerr << "ParticleDistribution::serialize: Warning: filename exists: " << filename << " You are overwriting an existing file.\n";

			ofstream out(filename, std::ofstream::binary);
			if (!out) throw invalid_argument("ParticleDistribution::serialize: unable to create file: " + filename);

			auto writeStrBuf = [&](const stringbuf& sb)
			{
				out.write(sb.str().c_str(), sb.str().length());
			};

			// ======== write data to file ======== //
			writeStrBuf(serializeStringVector(attrNames_m));
			writeStrBuf(serializeString(particleName_m));
			out.write(reinterpret_cast<const char*>(&mass_m), sizeof(float));

			size_t rangesSize{ ranges_m.size() };
			out.write(reinterpret_cast<char*>(&rangesSize), sizeof(size_t));
			for (const auto& range : ranges_m)
				writeStrBuf(range.serialize());

			writeStrBuf(serializeFloatVector(padvals_m));

			out.close();
		}

		void ParticleDistribution::deserialize(string serialFolder, string name)
		{
			string filename{ serialFolder + string("/") + name + string(".ser") };
			ifstream in(filename, std::ifstream::binary);
			if (!in) throw invalid_argument("ParticleDistribution::deserialize: unable to open file: " + filename);

			saveFolder_m = serialFolder;
			attrNames_m = deserializeStringVector(in);
			particleName_m = deserializeString(in);

			vector<char> masschr(sizeof(float), '\0'); // read mass float value
			in.read(masschr.data(), sizeof(float));
			mass_m = *(reinterpret_cast<float*>(masschr.data()));

			size_t rangesSize{ readSizetLength(in) };
			for (size_t range = 0; range < rangesSize; range++)
				ranges_m.push_back(std::move(Range(in)));

			padvals_m = deserializeFloatVector(in);

			in.close();
		}

	} /* end namespace utils::write */
} /* end namespace utils */
