#include <cmath>

#include "utils/numerical.h"

using std::cout;
using std::endl;
using std::to_string;
using std::invalid_argument;

namespace utils
{
	namespace numerical
	{
		DLLEXP void v2DtoEPitch(const vector<float>& vpara, const vector<float>& vperp, float mass, vector<eV>& energies, vector<degrees>& pitches)
		{
			if (vpara.size() != vperp.size())
				throw invalid_argument("utils::numerical::v2DtoEPitch: input vectors vpara and vperp are not the same size: " + to_string(vpara.size()) + ", " + to_string(vperp.size()));

			if (mass <= 0.0f) throw invalid_argument("utils::numerical::v2DtoEPitch: mass is lessthan or equal to zero " + to_string(mass));

			if (energies.size() != vpara.size()) //resize output vectors to be as big as input
				energies.resize(vpara.size());
			if (pitches.size() != vpara.size())
				pitches.resize(vpara.size());

			for (size_t part = 0; part < vpara.size(); part++)
			{
				bool nonZero{ vpara.at(part) != 0.0f || vperp.at(part) != 0.0f };

				if (nonZero) //check this or else the function can produce "NaN" in some indicies (I think atan2 is responsible) -> if false, the data at that index will be left 0
				{
					//const float test = mass * ( ( vpara.at(part) * vpara.at(part) ) + ( vperp.at(part) * vperp.at(part) ) ) / (float)JOULE_PER_EV;
					energies.at(part) = (0.5f * mass * (vpara.at(part) * vpara.at(part) + vperp.at(part) * vperp.at(part)) / (float)JOULE_PER_EV);
					pitches.at(part) = std::atan2f(std::abs(vperp.at(part)), -vpara.at(part)) / (float)RADS_PER_DEG;
				}
			}
		}

		DLLEXP void EPitchTov2D(const vector<eV>& energies, const vector<degrees>& pitches, float mass, vector<float>& vpara, vector<float>& vperp)
		{
			if (energies.size() != pitches.size())
				throw invalid_argument("utils::numerical::EPitchTov2D: input vectors vpara and vperp are not the same size: " + to_string(vpara.size()) + ", " + to_string(vperp.size()));

			if (mass <= 0.0f) throw invalid_argument("utils::numerical::v2DtoEPitch: mass is lessthan or equal to zero " + to_string(mass));

			if (energies.size() != vpara.size()) //resize output vectors to be as big as input
				vpara.resize(energies.size());
			if (energies.size() != vperp.size())
				vperp.resize(energies.size());

			for (size_t part = 0; part < energies.size(); part++)
			{
				if (energies.at(part) < 0.0f) throw invalid_argument("utils::numerical::EPitchTov2D: Energy is less than 0.  Some error has occurred.");

				bool nonZero{ energies.at(part) != 0.0f };

				if (nonZero) //check this or else the function can produce "NaN" in some indicies (I think atan2 is responsible) -> if false, the data at that index will be left 0
				{
					vpara.at(part) = -sqrtf(2 * energies.at(part) * (float)JOULE_PER_EV / mass) * cosf(pitches.at(part) * (float)RADS_PER_DEG);
					vperp.at(part) =  sqrtf(2 * energies.at(part) * (float)JOULE_PER_EV / mass) * sinf(pitches.at(part) * (float)RADS_PER_DEG);
				}
			}
		}

		DLLEXP vector<float> generateSpacedValues(float start, float end, int number, bool logSpaced, bool endInclusive)
		{
			/*
				**Note** if logSpaced is true, min and max have to be log(min) and log(max),
				min/max and in between values will be used as powers of 10 (10^(min | max))

				x -> values that are returned
			     dval
			    |-----|
				-------------------------
				x     x     x     x     x
				-------------------------
				^start ============= end^
				^min                 max^ << endInclusive = true, ("number" - 1) values
				^min           max^       << endInclusive = false, "number" values
			*/

			if (number <= 0)
				throw invalid_argument("utils::numerical::generateSpacedValues: number of values is less than / equal to zero");

			vector<float> ret(number);

			float dval{ (end - start) / ((endInclusive) ? (number - 1) : number) };
			for (int iter = 0; iter < number; iter++)
				ret.at(iter) = ((logSpaced) ? powf(10, iter * dval + start) : (iter * dval + start));

			return ret;
		}

		DLLEXP void normalize(vector<float>& normalizeMe, float normFactor, bool inverse) //inverse defaults to false
		{
			if (normFactor == 1.0f)
				return;

			for (auto& elem : normalizeMe) //normalize -> divide by normalization factor
				elem *= (inverse ? (normFactor) : (1 / normFactor));
		}

		DLLEXP float calcMean(const vector<float>& calcMyMean, bool absValue) //absValue defaults to false
		{
			float sum{ 0 };
			for (size_t iii = 0; iii < calcMyMean.size(); iii++)
			{
				if (absValue)
					sum += std::abs(calcMyMean.at(iii));
				else
					sum += calcMyMean.at(iii);
			}
			return sum / calcMyMean.size();
		}

		DLLEXP float calcStdDev(const vector<float>& calcMyStdDev)
		{
			float stdDev{ 0 };
			float mean{ calcMean(calcMyStdDev, false) };
			for (size_t iii = 0; iii < calcMyStdDev.size(); iii++)
			{
				stdDev += powf(calcMyStdDev.at(iii) - mean, 2);
			}
			stdDev = sqrt(stdDev / calcMyStdDev.size());
			return stdDev;
		}

		DLLEXP void coutMinMaxErr(const vector<float>& basevals, const vector<float>& testvals, string label, bool skipzeroes) //label defaults to "", skipzeroes to true
		{
			if (basevals.size() != testvals.size())
				throw invalid_argument("coutMinMaxErr: vectors are not the same size");

			float maxerr{ 0.0f };
			float minerr{ 3.0e38f };
			for (size_t iii = 0; iii < basevals.size(); iii++)
			{
				if (basevals.at(iii) == 0.0f && skipzeroes) { continue; }
				if (testvals.at(iii) == 0.0f && skipzeroes) { continue; }
				float err{ std::abs((basevals.at(iii) - testvals.at(iii)) / basevals.at(iii)) };
				if (err > maxerr) { maxerr = err; }
				if (err < minerr) { minerr = err; }
			}

			cout << label << " min err: " << minerr << ", max err: " << maxerr << endl;
		}

		DLLEXP void coutNumAboveErrEps(const vector<float>& basevals, const vector<float>& testvals, float errEps, string label, bool skipzeroes) //label defaults to "", skipzeroes to true
		{
			if (basevals.size() != testvals.size())
				throw invalid_argument("coutNumAboveEps: vectors are not the same size");

			int above{ 0 };
			for (size_t iii = 0; iii < basevals.size(); iii++)
			{
				if (basevals.at(iii) == 0.0f && skipzeroes) { continue; }
				if (testvals.at(iii) == 0.0f && skipzeroes) { continue; }
				if (std::abs((basevals.at(iii) - testvals.at(iii)) / basevals.at(iii)) > errEps) { above++; };
			}

			cout << label << " error above " << errEps << ": " << above << endl;
		}
	}
}
