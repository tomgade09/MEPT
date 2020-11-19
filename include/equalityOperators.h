#ifndef EQUALITYOPERATORS_H
#define EQUALITYOPERATORS_H

#include "Simulation/Simulation.h"

inline bool operator==(const Simulation& x, const Simulation& y)
{
	try
	{
		return (x.dt() == y.dt()) &&
			(x.simMin() == y.simMin()) &&
			(x.simMax() == y.simMax()) &&
			(x.getNumberOfParticleTypes() == y.getNumberOfParticleTypes() &&
			(x.getNumberOfSatellites() == y.getNumberOfSatellites()));
	}
	catch (...)
	{
		return false;
	}
}

inline bool operator==(const Particles& x, const Particles& y)
{
	try
	{
		bool same{ (x.name() == y.name()) &&
			(x.mass() == y.mass()) &&
			(x.charge() == y.charge()) &&
			(x.getNumberOfParticles() == y.getNumberOfParticles()) };

		for (int iii = 0; iii < x.getNumberOfAttributes(); iii++)
			same &= (x.getAttrNameByInd(iii) == y.getAttrNameByInd(iii));

		same &= (x.data(true ) == y.data(true ));
		same &= (x.data(false) == x.data(false));

		return same;
	}
	catch (...)
	{
		return false;
	}
}

inline bool operator==(const Satellite& x, const Satellite& y)
{
	try
	{
		return (x.name() == y.name()) &&
			   (x.altitude() == y.altitude()) &&
		       (x.upward() == y.upward()) &&
		       (x.data() == y.data());
	}
	catch (...)
	{
		return false;
	}
}

inline bool operator==(const BModel& x, const BModel& y)
{
	try
	{
		if (x.getAllAttributes() != y.getAllAttributes())
			return false;

		if (!(
			(x.getBFieldAtS(4.0e6, 0.0) == y.getBFieldAtS(4.0e6, 0.0)) &&
			(x.getGradBAtS(4.0e6, 0.0) == y.getGradBAtS(4.0e6, 0.0))
			))
			return false;
	}
	catch (...)
	{
		return false;
	}

	return true;
}

inline bool operator==(const EModel& x, const EModel& y)
{
	try
	{
		if (x.getAllAttributes() != y.getAllAttributes())
		{
			return false;
		}
	}
	catch (...)
	{
		return false;
	}

	return true;
}

inline bool operator==(const EField& x, const EField& y)
{
	try
	{
		for (int emodel = 0; emodel < x.size(); emodel++) //maybe shouldn't check elems here??
		{
			EModel& xm{ *(x.emodel(emodel)) };
			EModel& ym{ *(y.emodel(emodel)) };
			
			if (!(xm == ym)) //if the elements (references) are not equal - operator defined above
				return false;
		}
	}
	catch (...)
	{
		return false;
	}

	return true;
}

#endif