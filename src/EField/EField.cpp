#include "EField/EField.h"
#include "EField/QSPS.h"

#include <iostream>
#include <sstream>

#include "utils/serializationHelpers.h"

using std::move;
using std::make_unique;
using std::runtime_error;
using namespace utils::fileIO::serialize;

void EField::serialize(ofstream& out) const
{
	writeSizetLength(out, emodels_m.size());

	for (const auto& emodel : emodels_m)
	{
		out.write(reinterpret_cast<char*>(&(emodel->type_m)), sizeof(EModel::Type)); //write type of emodel
		emodel->serialize(out);
	}
}

void EField::deserialize(ifstream& in)
{
	size_t len{ readSizetLength(in) };

	for (size_t emodel = 0; emodel < len; emodel++)
	{
		EModel::Type type{ 6 };
		in.read(reinterpret_cast<char*>(&type), sizeof(EModel::Type));

		if (type == EModel::Type::QSPS) emodels_m.push_back(move(make_unique<QSPS>(in)));
		//else if (type == EModel::Type::AlfvenLUT) elements_m.push_back(move(make_unique<AlfvenLUT>(in)));
		else throw runtime_error("EField::deserialize: unknown EModel Type");
	}
}