#ifndef LOG_H
#define LOG_H

#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <thread>
#include <iostream>
#include <sstream>
#include <mutex>
#include <condition_variable>

using std::string;
using std::vector;
using std::ofstream;
using std::streambuf;
using std::unique_ptr;
using std::stringstream;
typedef std::condition_variable condvar;

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
	stringstream clog_m; //same facility for clog
	streambuf*   cerrBufferBackup_m{ std::cerr.rdbuf() }; //save old buffer so we can restore it
	streambuf*   clogBufferBackup_m{ std::clog.rdbuf() };

	//Threads and concurrency associated variables
	std::thread  strmRead_m;
	std::thread  logWrite_m;
	bool         check_m{ true };                         //check cerr and clog for log entries while this is true
	bool         write_m{ true };                         //set to true while writing to file
	condvar      writeReady_m;                            //condition variable that is signaled when a write to file is ready
	std::mutex   mutex_m;                                 //mutex associated with locks acquired

	void writeEntries();
	void streamCheck();

public:
	Log(string logFileName);
	~Log();

	size_t createEntry(string message, bool write = true);
	
	double timeElapsedTotal_s() const;
	double timeElapsedSinceEntry_s(size_t index) const;
};

#endif