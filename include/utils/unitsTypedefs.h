#ifndef UTILS_UNITSTYPEDEFS_H
#define UTILS_UNITSTYPEDEFS_H

#include <vector>
#include <string>

#define USE_DP_FLTPT

#ifdef USE_SP_FLTPT
typedef float flPt_t;                      //floating point type (can select float or double at compile time)
#else
typedef double flPt_t;
#endif

using std::vector;
using std::string;

typedef flPt_t eV;                          //electron volts
typedef flPt_t kg;                          //kilograms
typedef flPt_t cm;                          //centimeters
typedef flPt_t Vperm;                       //volts per meter (E Field)
typedef flPt_t tesla;                       //tesla (B Field)
typedef flPt_t ratio;                       //no units
typedef flPt_t mpers;                       //meters per second
typedef flPt_t mpers2;                      //meters per second squared
typedef flPt_t percm3;                      //per centimeters cubed (1/volume)
typedef flPt_t meters;                      //meters
typedef flPt_t dNflux;                      //differential number flux
typedef flPt_t dEflux;                      //differential energy flux
typedef flPt_t degrees;                     //degrees (measure of angle)
typedef flPt_t coulomb;                     //coulomb (measure of electric charge)
typedef flPt_t percent;                     //no units
typedef flPt_t seconds;                     //seconds
typedef vector<dNflux> dNflux_v1D;          //1D std::vector of differential number flux
typedef vector<dEflux> dEflux_v1D;          //1D std::vector of differential energy flux
typedef vector<string> strvec;              //string vector
typedef vector<flPt_t> fp1Dvec;             //1D std::vector of floating point type
typedef vector<vector<flPt_t>> fp2Dvec;     //2D std::vector of floating point type (std::vector of std::vectors)

#endif
