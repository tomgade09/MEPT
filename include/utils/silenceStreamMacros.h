#ifndef UTILS_SILENCESTREAM_H
#define UTILS_SILENCESTREAM_H

#include <sstream>
#include <iostream>

//takes a stream (cout, cerr) and dumps whatever is written to it to a stringstream which is discarded after execution of code "x"
//don't use this macro directly unless you need to specify another stream (e.g. clog), use the below
#define _SILENCE_STREAM(x, stream) { std::stringstream nullss; std::streambuf* bak(stream.rdbuf()); stream.rdbuf(nullss.rdbuf()); x; stream.rdbuf(bak); }

#define SILENCE_COUT(x) { _SILENCE_STREAM(x, std::cout); }
#define SILENCE_CERR(x) { _SILENCE_STREAM(x, std::cerr); }

#endif /* !UTILS_SILENCESTREAM_H */