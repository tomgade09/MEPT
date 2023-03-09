#include <iomanip>
#include <fstream>

#include "Log/Log.h"
#include "ErrorHandling/simExceptionMacros.h"

using std::cout;
using std::clog;
using std::cerr;
using std::setw;
using std::to_string;
using std::make_unique;
using std::setprecision;
using std::stringstream;
using std::invalid_argument;

using namespace std::chrono;

Log::Entry::Entry(std::chrono::steady_clock::time_point time, string message, bool write, bool error) :
	time_m{ time }, message_m{ message }, error_m{ error }, write_m{ write }
{
	
}

void Log::writeEntries()
{
	//Runs in its own thread - checks if there are any entries ready for writing
	//waits on condition variable, woken by other threads
	seconds delay{ 2 };
	
	while (write_m)
	{		
		auto write = [&](Entry& entry, ofstream& logFile)
		{
			if (!entry.write_m) return;

			duration<double, std::nano> dur{ entry.time_m - entries_m.at(0).time_m };

			string writeTxt;
			stringstream timess;

			timess << setprecision(10) << setw(11) << dur.count() / 1000000;
			string timestr{ timess.str() };
			while (timestr.length() > 11)
				timestr.erase(std::ios::end);
			writeTxt = "[ " + timestr + " ] : " + ((entry.error_m) ? "ERROR: " : "") + entry.message_m + "\n";

			logFile << writeTxt;
		};

		std::unique_lock<std::mutex> ulock(mutex_m);
		writeReady_m.wait_for(ulock, delay);  //wait until an entry is ready for writing
											     //wait_for is used to prevent the race condition when the dtor
											     //signals the condition variable before this thread waits on it
		
		ofstream logFile(logFilePath_m, std::ios::app);
		if (!logFile) cout << "Log::saveEntry: unable to open log\n";

		for (auto& entry : entries_m)
		{   //scans the list of entries every time for writeable entries, but it works
			if (entry.write_m) write(entry, logFile);
			entry.write_m = false;
		}
		
		logFile.close();
	}
}

void Log::streamCheck()
{
	//Runs in its own thread - checks if there are any entries ready for writing
	//sleeps to ensure checking the streams every so often
	
	while (check_m)
	{
		auto check = [&](stringstream& stream, bool error)
		{
			if (stream.str().length() > 0)
			{
				size_t tmplen{ stream.str().length() };

				do
				{
					tmplen = stream.str().length();
					std::this_thread::sleep_for(microseconds(10));
				}
				while (tmplen != stream.str().length()); //block until cerr is not being written to anymore

				entries_m.push_back(Entry(steady_clock::now(), stream.str(), true, error));
				stream.str(string());
				stream.clear();

				writeReady_m.notify_one();
			}
		};

		check(clog_m, false);
		check(cerr_m, true);
		
		std::this_thread::sleep_for(milliseconds(10)); //check cerr every __ <<
	}
}

Log::Log(string logFilePath) : logFilePath_m{ logFilePath }
{
	cerr.rdbuf(cerr_m.rdbuf()); //set cerr read buffer to this class' stringstream
	clog.rdbuf(clog_m.rdbuf());
	
	{   //lock block - when block goes out of scope, lock is released
		std::lock_guard<std::mutex> lock(mutex_m);  //grab the lock so we can write to file
		
		ofstream logFile(logFilePath_m, std::ios::trunc); //overwrites old log - specifically for reloading sims
		if (!logFile) throw invalid_argument("Log::Log: unable to create log");
		logFile << string("[  Time (ms)  ] : Log Message\n");
		logFile.close();
	}

	strmRead_m = std::thread([&]() { streamCheck(); });
	logWrite_m = std::thread([&]() { writeEntries(); });
	
	createEntry("Simulation initialized");
}

Log::~Log()
{
	check_m = false;
	write_m = false;

	writeReady_m.notify_one();      //logWriteThread_m
	
	SIM_API_EXCEP_CHECK(strmRead_m.join());
	SIM_API_EXCEP_CHECK(logWrite_m.join());
	cerr.rdbuf(cerrBufferBackup_m); //restore cerr to normal
	clog.rdbuf(clogBufferBackup_m);
}

size_t Log::createEntry(string label, bool write)
{
	entries_m.push_back(Entry(steady_clock::now(), label, write));

	if (write) writeReady_m.notify_one();

	return entries_m.size() - 1;
}

double Log::timeElapsedTotal_s() const
{
	return timeElapsedSinceEntry_s(0);
}

double Log::timeElapsedSinceEntry_s(size_t index) const
{
	duration<double, std::nano> elapsed = steady_clock::now() - entries_m.at(index).time_m;
	return elapsed.count() / 1000000000.0; //convert to s
}