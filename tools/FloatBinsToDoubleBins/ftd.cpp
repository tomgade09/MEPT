#include <vector>
#include <fstream>
#include <iostream>
#include <filesystem>

using std::cout;
using std::string;
using std::vector;
namespace fs = std::filesystem;

#define DIRCREATEIFNOTEXIST(x) \
	if (!fs::exists(fs::path(x))) \
	{\
		try\
		{\
			fs::create_directory(fs::path(x));\
			cout << "Created: " << x << "\n";\
		}\
		catch (std::exception& e)\
		{\
			cout << "Exception: " << e.what() << "\n";\
			exit(1);\
		}\
	}

void readFltBin(vector<float>& arrayToReadInto, const fs::path filename, size_t numToRead)
{
	if (arrayToReadInto.size() != numToRead)
		throw std::invalid_argument("readFltBin: vector is not the same size as data read from file " + filename.string());

	std::ifstream binFile{ filename, std::ios::binary };
	if (!binFile.is_open())
		throw std::invalid_argument("readFltBin: could not open file " + filename.string() + " for reading");

	binFile.seekg(0, binFile.end);
	size_t length{ (size_t)binFile.tellg() };
	binFile.seekg(0, binFile.beg);

	if (length < numToRead * sizeof(float))
	{
		binFile.close();
		throw std::invalid_argument("fileIO::readFltBin: filesize of \"" + filename.string() + "\" is smaller than specified number of floats to read");
	}
	if (length > numToRead * sizeof(float))
		std::cerr << "fileIO::readFltBin: warning: size of data read is less than the size of all data in file " << filename.string() << ": continuing" << std::endl;

	binFile.read(reinterpret_cast<char*>(arrayToReadInto.data()), std::streamsize(numToRead * sizeof(float)));
	binFile.close();
}

size_t readFltBin(vector<float>& arrayToReadInto, const fs::path filename)
{
	std::ifstream binFile{ filename, std::ios::binary };
	if (!binFile.is_open())
		throw std::invalid_argument("readFltBin: could not open file " + filename.string() + " for reading");

	binFile.seekg(0, binFile.end);
	size_t length{ (size_t)binFile.tellg() / sizeof(float) };
	binFile.seekg(0, binFile.beg);
	binFile.close();

	arrayToReadInto.resize(length);

	readFltBin(arrayToReadInto, filename, length);
	
	return length;
}

void writeDblBin(const vector<double>& dataarray, const fs::path filename, size_t numelements, bool overwrite)
{
	std::ofstream binfile{ filename, std::ios::binary | (overwrite ? (std::ios::trunc) : (std::ios::app)) };
	if (!binfile.is_open())
		throw std::invalid_argument("fileIO::writeDblBin: could not open file " + filename.string() + " for writing");
	if (dataarray.size() < numelements)
	{
		binfile.close();
		throw std::invalid_argument("fileIO::writeDblBin: size of data vector is less than the number of floats requested from it for filename " + filename.string());
	}

	binfile.write(reinterpret_cast<const char*>(dataarray.data()), std::streamsize(numelements * sizeof(double)));
	binfile.close();
}

int main(int argc, char* argv[])
{
	if (argc < 4)
	{
		cout
			<< "\nftd Usage:\n\n" << "\tftd.exe datadir partname satname [satname satname...]\n"
			<< "\n\tdatadir:\tThe root directory where the data to process resides.\n"
			<< "\t\t\tThis is usually named by the date in the following format: YYMMDD_HH.MM.SS\n\n"
			<< "\tpartname:\tThe name of the particle.  This is used to read and name the files on disk.\n"
			<< "\tsatname:\tThe name of the satellite.  This is used to read and name the files on disk.\n"
			<< "\t\t\tNote: multiple satellites can be specified here.\n\n"
			<< "\tData is output to folder \"out\"\n\n"
			<< "Exiting.\n";
		exit(1);
	}

	string datadir{ argv[1] };
	string partnm{ argv[2] };
	vector<string> satnames;
	for (size_t arg = 3; arg < argc; arg++)  //grab satellite names from passed in args
		satnames.push_back(string(argv[arg]));
	
	vector<string> dirs{
		"/particles_final/",
		"/particles_init/",
		"/satellites/" };

	cout << "Creating directories and checking for properly formed input directory...\n";
	DIRCREATEIFNOTEXIST("./out");
	for (size_t i = 0; i < dirs.size(); i++)   //check if all folders exist
	{
		fs::path name{ datadir + "/bins" + dirs.at(i) };
		
		if (!fs::exists(name))
		{
			cout << "Directory: " << datadir << " cannot be resolved, or does not contain appropriate data.  Exiting.\n";
			exit(1);
		}
		DIRCREATEIFNOTEXIST("./out" + dirs.at(i));
	}

	vector<string> partfnames{
		partnm + "_s.bin",
		partnm + "_t_esc.bin",
		partnm + "_t_inc.bin",
		partnm + "_vpara.bin",
		partnm + "_vperp.bin"
	};

	vector<string> satfnames{
		"_index.bin",
		"_s.bin",
		"_time.bin",
		"_vpara.bin",
		"_vperp.bin"
	};
	size_t satfnmCount{ satfnames.size() };
	
	for (size_t sat = 0; sat < satnames.size(); sat++)  //append one each of both sat and fname yielding sat_fname.bin for each sat and fname
		for (size_t f = 0; f < satfnmCount; f++)
			satfnames.push_back(satnames.at(sat) + satfnames.at(f));

	for (size_t del = 0; del < satfnmCount; del++)      //remove first names without sat name
		satfnames.erase(satfnames.begin());
	
	vector<float> fltsFromDisk;
	cout << "Reading floats from disk and saving to double files...\n";
	try
	{   //Read files from disk as float vectors, write to disk as double vectors
		for (size_t dir = 0; dir < dirs.size(); dir++) //iterate over directories
		{
			bool sats{ dirs.at(dir) == "/satellites/" };
			
			vector<string>& names{ (sats) ? satfnames : partfnames };

			for (size_t nm = 0; nm < names.size(); nm++) //iterate over names within directory
			{
				fs::path inname{ datadir + "/bins" + dirs.at(dir) + names.at(nm) };
				fs::path outname{ "./out" + dirs.at(dir) + names.at(nm) };
				
				fltsFromDisk.clear();
				size_t read{ readFltBin(fltsFromDisk, inname) };

				if (read * sizeof(float) != static_cast<size_t>(fs::file_size(inname)))
					cout << "Warning: " << inname << " number of bytes read " << read * sizeof(float)
					<< " does not equal filesize " << fs::file_size(inname) << "\n";

				vector<double> dblsToDisk;
				for (const auto& flt : fltsFromDisk)
					dblsToDisk.push_back(static_cast<double>(flt));

				//overwrites existing files
				writeDblBin(dblsToDisk, outname, dblsToDisk.size(), true);
			}
		}
	}
	catch(std::exception& e)
	{
		cout << "Exception: " << e.what() << "  Exiting.\n";
		exit(1);
	}
	cout << "Done.  Files are located in './out'\n";

	return 0;
}