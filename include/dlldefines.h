#ifndef DLLIMPEXP_DEFINES_H
#define DLLIMPEXP_DEFINES_H

//Windows defines
#ifdef _WIN32

	#ifdef DLLFILE /* DLLFILE is defined for GPS, not TESTS */
		#define DLLEXP_EXTC extern "C" __declspec(dllexport) /* for python use of the library */
		#define DLLEXP __declspec(dllexport)
		#define DLLCLEXP __declspec(dllexport)
	#else
		#ifndef _DEBUG /* not _DEBUG, no DLLFILE */
			#define DLLEXP_EXTC
			#define DLLEXP
			#define DLLCLEXP __declspec(dllimport)
		#else /* _DEBUG, no DLLFILE (not defined with DEBUG) */
			#define DLLEXP_EXTC
			#define DLLEXP
			#define DLLCLEXP
		#endif /* _DEBUG */
	#endif /* DLLFILE */

#else /* !_WIN32 - non Windows defines */

	#define DLLEXP_EXTC extern "C"
	#define DLLEXP
	#define DLLCLEXP
	#define FLT_EPSILON 1.192092896e-7F

#endif /* _WIN32 */

#include <cmath> //include and using necessary for linux
using std::log10;
using std::pow;
using std::exp;
using std::sqrt;
using std::sin;
using std::cos;

#endif /* !DLLIMPEXP_DEFINES_H */