#include "API/LogAPI.h"

DLLEXP_EXTC void writeLogFileEntryAPI(Log* log, const char* logMessage) {
	log->createEntry(logMessage); }

//DLLEXP_EXTC void writeTimeDiffFromNowAPI(Log* log, int startTSind, const char* nowLabel) {
//	log->writeTimeDiffFromNow(startTSind, nowLabel); }

//DLLEXP_EXTC void writeTimeDiffAPI(Log* log, int startTSind, int endTSind) {
//	log->writeTimeDiff(startTSind, endTSind); }