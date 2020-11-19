#ifndef SIMEXCEPHANDLER_H
#define SIMEXCEPHANDLER_H

#include <iostream>

//exception checking for API functions
#define SIM_API_EXCEP_CHECK(x) \
	try{ x; } \
	catch(const std::invalid_argument& a) { std::cerr << __FILE__ << ":" << __LINE__ << " : " << "Invalid argument error: " << a.what() << std::endl; std::cout << "Exception: check log for details" << std::endl; } \
	catch(const std::out_of_range& oor)   { std::cerr << __FILE__ << ":" << __LINE__ << " : " << "Vector out of range exception: " << oor.what() << std::endl; std::cout << "Exception: check log for details" << std::endl; } \
	catch(const std::logic_error& log)    { std::cerr << __FILE__ << ":" << __LINE__ << " : " << "Logic error exception: " << log.what() << std::endl; std::cout << "Exception: check log for details" << std::endl; } \
	catch(const std::runtime_error& rte)  { std::cerr << __FILE__ << ":" << __LINE__ << " : " << "Generic runtime error: " << rte.what() << std::endl; std::cout << "Exception: check log for details" << std::endl; } \
	catch(const std::exception& exp)      { std::cerr << __FILE__ << ":" << __LINE__ << " : " << "Unhandled specific std::exception: " << exp.what() << std::endl; std::cout << "Exception: check log for details" << std::endl; } \
	catch(...)                            { std::cerr << __FILE__ << ":" << __LINE__ << " : " << "Other unhandled exception - exiting out of precaution"  << std::endl; std::cout << "Exception: check log for details" << std::endl; exit(EXIT_FAILURE);}

#endif /* SIMEXCEPHANDLER_H */