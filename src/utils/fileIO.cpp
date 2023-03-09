#include "utils/fileIO.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <typeinfo>

//file read/write exception checking (probably should mostly wrap fileIO functions)
#define FILE_RDWR_EXCEP_CHECK(x) \
	try{ x; } \
	catch(const std::invalid_argument& a) { std::cerr << __FILE__ << ":" << __LINE__ << " : " << "Invalid argument error: " << a.what() << ": continuing without loading file" << std::endl; std::cout << "FileIO exception: check log file for details" << std::endl; } \
	catch(...)                            { throw; }

namespace utils
{
	namespace fileIO
	{
		//bin functions
		template <typename T1>
		DLLEXP void readBin(vector<T1>& arrayToReadInto, string filename, size_t numOfNumsToRead) //numOfNumsToRead defaults to 0
		{//read binary files from disk, interpreting them as type T1
			static_assert(std::is_arithmetic_v<T1>, "readBin is only usable with aritmetic types");

			std::ifstream binFile{ filename, std::ios::binary };
			if (!binFile.is_open())
				throw std::invalid_argument("fileIO::readBin: could not open file " + filename + " for reading");

			binFile.seekg(0, binFile.end);
			size_t flen{ (size_t)binFile.tellg() / sizeof(T1) };  //file length in number of numbers stored in file
			binFile.seekg(0, binFile.beg);
			
			if (numOfNumsToRead == 0)
				numOfNumsToRead = flen;
			if (arrayToReadInto.size() != numOfNumsToRead) //resize array to be able to hold number of numbers being read
				arrayToReadInto.resize(numOfNumsToRead);
			if (flen < numOfNumsToRead * sizeof(T1))
			{
				binFile.close();
				throw std::invalid_argument("fileIO::readBin: filesize of \"" + filename + "\" is smaller than the number of numbers to read specified by the user");
			}
			if (flen > numOfNumsToRead * sizeof(T1))
				std::cerr << "fileIO::readBin: warning: size of data read is less than the size of all data in file " << filename << ": continuing" << std::endl;

			binFile.read(reinterpret_cast<char*>(arrayToReadInto.data()), std::streamsize(numOfNumsToRead * sizeof(T1)));
			binFile.close();
		}

		//These template instantiations are a required part of implementing template definitions in .cpp files (with forward decls in headers)
		//see: https://isocpp.org/wiki/faq/templates#separate-template-fn-defn-from-decl
		template DLLEXP void readBin<flPt_t>(fp1Dvec& arrayToReadInto, string filename, size_t numOfNumsToRead);
		template DLLEXP void readBin<int>(vector<int>& arrayToReadInto, string filename, size_t numOfNumsToRead);

		template <typename T1>
		DLLEXP void writeBin(const vector<T1>& dataarray, string filename, size_t numelements, bool overwrite)//overwrite defaults to true
		{
			static_assert(std::is_arithmetic_v<T1>, "writeBin is only usable with aritmetic types");
			
			std::ofstream binfile{ filename, std::ios::binary | (overwrite ? (std::ios::trunc) : (std::ios::app)) };
			if (!binfile.is_open())
				throw std::invalid_argument("fileIO::writeBin: could not open file " + filename + " for writing");
			if (dataarray.size() < numelements)
			{
				binfile.close();
				throw std::invalid_argument("fileIO::writeBin: size of data vector is less than the number of numbers requested from it for filename " + filename);
			}

			binfile.write(reinterpret_cast<const char*>(dataarray.data()), std::streamsize(numelements * sizeof(T1)));
			binfile.close();
		}

		template DLLEXP void writeBin<flPt_t>(const fp1Dvec&, string, size_t, bool);
		template DLLEXP void writeBin<int>(const vector<int>&, string, size_t, bool);

		
		//CSV functions
		template <typename T1>
		DLLEXP void read2DCSV(vector<vector<T1>>& array2DToReadInto, string filename, size_t numofentries, size_t numofcols, const char delim)
		{
			static_assert(std::is_arithmetic_v<T1>, "read2DCSV is only usable with aritmetic types");
			
			std::ifstream csv{ filename };
			if (!csv.is_open())
				throw std::invalid_argument("fileIO::read2DCSV: could not open file " + filename + " for reading");
			if (array2DToReadInto.size() != numofcols)  //reshape vector to exactly contain all data
				array2DToReadInto.resize(numofcols);
			for (size_t col = 0; col < array2DToReadInto.size(); col++)
			{
				if (array2DToReadInto.at(col).size() != numofentries)
					array2DToReadInto.at(col).resize(numofentries);
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

		template <typename T1>
		DLLEXP void write2DCSV(const vector<vector<T1>>& dataarray, string filename, size_t numofentries, size_t numofcols, const char delim, bool overwrite, int precision)//overwrite defaults to true, precision to 20
		{
			static_assert(std::is_arithmetic_v<T1>, "write2DCSV is only usable with aritmetic types");
			
			std::ofstream csv(filename, overwrite ? (std::ios::trunc) : (std::ios::app));
			if (!csv.is_open())
				throw std::invalid_argument("fileIO::write2DCSV: could not open file " + filename + " for writing");
			if (dataarray.size() < numofcols)
			{
				csv.close();
				throw std::invalid_argument("fileIO::write2DCSV: size of data vector is less than the numbers requested from it for filename " + filename);
			}

			for (size_t iii = 0; iii < numofentries; iii++)
			{
				for (size_t jjj = 0; jjj < numofcols; jjj++)
				{
					try
					{
						csv << std::setprecision(precision) << dataarray.at(jjj).at(iii) << delim;
					}
					catch (std::out_of_range&) //if the csv being written has uneven columns, this just fills in a blank where there's no data
					{
						csv << "" << delim;
					}
				}

				csv << "\n";
			}

			csv.close();
		}

		template DLLEXP void write2DCSV<flPt_t>(const fp2Dvec&, string, size_t, size_t, const char, bool, int);

		
		//txt file functions
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