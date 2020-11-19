#ifndef UTILS_LOOPMACROS_H
#define UTILS_LOOPMACROS_H

//Cannot nest these - outer loops all use iii as a variable name
#define LOOP_OVER_3D_ARRAY(d1, d2, d3, x) for (int iii = 0; iii < d1; iii++)\
	{ for (int jjj = 0; jjj < d2; jjj++) \
		{ for (int kk = 0; kk < d3; kk++) {x;} } }

#define LOOP_OVER_2D_ARRAY(d1, d2, x) for (int iii = 0; iii < d1; iii++)\
	{ for (int jjj = 0; jjj < d2; jjj++) {x;} }

#define LOOP_OVER_1D_ARRAY(d1, x) for (int iii = 0; iii < d1; iii++) {x;}

#endif /* !UTILS_LOOPMACROS_H */