#ifndef LOGFILEAPI_H
#define LOGFILEAPI_H

#include "dlldefines.h"
#include "Log/Log.h"

DLLEXP_EXTC void writeLogFileEntryAPI(Log* log, const char* logMessage);
//DLLEXP_EXTC void writeTimeDiffFromNowAPI(Log* log, int startTSind, const char* nowLabel);
//DLLEXP_EXTC void writeTimeDiffAPI(Log* log, int startTSind, int endTSind);

#endif