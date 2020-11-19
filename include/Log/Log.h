#ifndef LOG_H
#define LOG_H

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <thread>
#include <iostream>
#include <sstream>
#include <fstream>

using std::string;
using std::vector;
using std::ofstream;
using std::streambuf;
using std::unique_ptr;
using std::stringstream;

class Log
{
private:
	struct Entry
	{
		std::chrono::steady_clock::time_point time_m;
		string message_m;
		bool error_m{ false };
		bool write_m{ true };

		Entry(std::chrono::steady_clock::time_point time, string message, bool write = true, bool error = false);
	};

	string logFilePath_m;
	vector<Entry> entries_m;

	stringstream cerr_m; //stringstream to capture cerr so we can check it / write to log
	streambuf*   cerrBufferBackup_m{ std::cerr.rdbuf() };
	std::thread  cerrReadThread_m;
	bool         check_m{ true };
	bool         writing_m{ false };

	void saveEntry(const Entry& entr);
	void cerrCheck();

public:
	Log(string logFileName);
	~Log();

	size_t createEntry(string message, bool write = true);
	
	double timeElapsedTotal_s() const;
	double timeElapsedSinceEntry_s(size_t index) const;
};

#endif