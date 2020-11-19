#include "utils/fileIO.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

//file read/write exception checking (probably should mostly wrap fileIO functions)
#define FILE_RDWR_EXCEP_CHECK(x) \
	try{ x; } \
	catch(const std::invalid_argument& a) { std::cerr << __FILE__ << ":" << __LINE__ << " : " << "Invalid argument error: " << a.what() << ": continuing without loading file" << std::endl; std::cout << "FileIO exception: check log file for details" << std::endl; } \
	catch(...)                            { throw; }

namespace utils
{
	namespace fileIO
	{
		DLLEXP void readDblBin(vector<double>& arrayToReadInto, string filename)
		{
			std::ifstream binFile{ filename, std::ios::binary };
			if (!binFile.is_open())
				throw std::invalid_argument("fileIO::readDblBin: could not open file " + filename + " for reading");

			binFile.seekg(0, binFile.end);
			size_t length{ (size_t)binFile.tellg() / 8 };
			binFile.seekg(0, binFile.beg);
			binFile.close();
			//std::cout << "fileIO::readDblBin: (unknown size): num elements determined from disk: " << length << std::endl;

			arrayToReadInto.resize(length);

			readDblBin(arrayToReadInto, filename, length);
		}

		DLLEXP void readDblBin(vector<double>& arrayToReadInto, string filename, size_t numOfDblsToRead)
		{
			if (arrayToReadInto.size() < numOfDblsToRead)
				throw std::invalid_argument("fileIO::readDblBin: vector is not big enough to contain the data being read from file " + filename);

			std::ifstream binFile{ filename, std::ios::binary };
			if (!binFile.is_open())
				throw std::invalid_argument("fileIO::readDblBin: could not open file " + filename + " for reading");

			binFile.seekg(0, binFile.end);
			size_t length{ (size_t)binFile.tellg() };
			binFile.seekg(0, binFile.beg);

			if (length < numOfDblsToRead * 8)
			{
				binFile.close();
				throw std::invalid_argument("fileIO::readDblBin: filesize of \"" + filename + "\" is smaller than specified number of doubles to read");
			}
			if (length > numOfDblsToRead * 8)
				std::cerr << "fileIO::readDblBin: warning: size of data read is less than the size of all data in file " << filename << ": continuing" << std::endl;

			binFile.read(reinterpret_cast<char*>(arrayToReadInto.data()), std::streamsize(numOfDblsToRead * sizeof(double)));
			binFile.close();
		}

		DLLEXP void read2DCSV(vector<vector<double>>& array2DToReadInto, string filename, size_t numofentries, size_t numofcols, const char delim)
		{
			std::ifstream csv{ filename };
			if (!csv.is_open())
				throw std::invalid_argument("fileIO::read2DCSV: could not open file " + filename + " for reading");
			if (array2DToReadInto.size() < numofcols)
				throw std::invalid_argument("fileIO::read2DCSV: vector outer vector is not big enough to contain the data being read from file " + filename);
			if (array2DToReadInto.size() > numofcols)
				std::cerr << "fileIO::read2DCSV: vector outer vector is bigger than numofcols, some data in the vector will remain unmodified" << std::endl;
			for (size_t col = 0; col < array2DToReadInto.size(); col++)
			{
				if (array2DToReadInto.at(col).size() < numofentries)
					throw std::invalid_argument("fileIO::read2DCSV: vector inner vector is not big enough to contain the data being read from file " + filename);
				if (array2DToReadInto.at(col).size() > numofentries)
					std::cerr << "fileIO::read2DCSV: vector inner vector is bigger than numofentries, some data in the vector will remain unmodified" << std::endl;
			}

			try
			{
				for (size_t iii = 0; iii < numofentries; iii++)
				{
					string in;
					std::getline(csv, in);

					std::stringstream in_ss(in);

					for (size_t jjj = 0; jjj < numofcols; jjj++)
					{
						string val;
						std::getline(in_ss, val, delim);
						std::stringstream convert(val);
						convert >> array2DToReadInto.at(jjj).at(iii);
					}
				}
			}
			catch (...)
			{
				csv.close();
				throw;
			}

			csv.close();
		}

		DLLEXP void readTxtFile(string& readInto, string filename)
		{
			std::ifstream txt(filename);
			if (!txt.is_open())
			{
				txt.close();
				throw std::invalid_argument("fileIO::readTxtFile: could not open file " + filename + " for reading " + std::to_string(txt.is_open()));
			}

			std::stringstream buf;
			buf << txt.rdbuf();
			txt.close();

			readInto = buf.str();
		}


		//write functions
		DLLEXP void writeDblBin(const vector<double>& dataarray, string filename, size_t numelements, bool overwrite)//overwrite defaults to true
		{
			std::ofstream binfile{ filename, std::ios::binary | (overwrite ? (std::ios::trunc) : (std::ios::app)) };
			if (!binfile.is_open())
				throw std::invalid_argument("fileIO::writeDblBin: could not open file " + filename + " for writing");
			if (dataarray.size() < numelements)
			{
				binfile.close();
				throw std::invalid_argument("fileIO::writeDblBin: size of data vector is less than the number of doubles requested from it for filename " + filename);
			}

			binfile.write(reinterpret_cast<const char*>(dataarray.data()), std::streamsize(numelements * sizeof(double)));
			binfile.close();
		}

		DLLEXP void write2DCSV(const vector<vector<double>>& dataarray, string filename, size_t numofentries, size_t numofcols, const char delim, bool overwrite, int precision)//overwrite defaults to true, precision to 20
		{
			std::ofstream csv(filename, overwrite ? (std::ios::trunc) : (std::ios::app));
			if (!csv.is_open())
				throw std::invalid_argument("fileIO::write2DCSV: could not open file " + filename + " for writing");
			if (dataarray.size() < numofcols)
			{
				csv.close();
				throw std::invalid_argument("fileIO::write2DCSV: size of data vector is less than the doubles requested from it for filename " + filename);
			}

			for (size_t iii = 0; iii < numofentries; iii++)
			{
				for (size_t jjj = 0; jjj < numofcols; jjj++)
				{
					try
					{
						csv << std::setprecision(precision) << dataarray.at(jjj).at(iii) << delim;
					}
					catch (std::out_of_range) //if the csv being written has uneven columns, this just fills in a blank where there's no data
					{
						csv << "" << delim;
					}
				}

				csv << "\n";
			}

			csv.close();
		}

		DLLEXP void writeTxtFile(string textToWrite, string filename, bool overwrite)//overwrite defaults to false
		{
			std::ofstream txt(filename, overwrite ? (std::ios::trunc) : (std::ios::app));
			if (!txt.is_open())
			{
				txt.close();
				throw std::invalid_argument("fileIO::writeTxtFile: could not open file " + filename + " for writing " + std::to_string(txt.is_open()));
			}

			txt << textToWrite;

			txt.close();
		}
	} /* END fileIO */
} /* END utils */