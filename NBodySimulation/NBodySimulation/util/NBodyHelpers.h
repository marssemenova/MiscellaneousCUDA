
#ifndef _NBODY_HELPERS_
#define _NBODY_HELPERS_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX(X, Y) ((X) < (Y) ? (Y) : (X))


/* pi */
static const double _PI = 3.1415926535897932;

/* A floating-point equality comparison epsilon. */
static const double _EPS = 1e-20;

/**
 * Get a floating point random number in the range [0,1]
 */
static inline double frand() {
    return ((double) rand()) / (double) RAND_MAX;
}

/**
 * Possible error codes on exit.
 */
enum ERROR_CODES {
	ALLOC_ERROR = 0x00000001,
	INIT_ERROR  = 0x00000002
};


/**
 * Scale an array of n doubles by scale.
 *
 * @param n: the number of bodies.
 * @param m: an array of n doubles.
 * @param scale: the multiplicative scale factor.
 */
static inline void scaleArray_NB(long n, double* m, double scale) {
	for (long i = 0; i < n; ++i) {
		m[i] *= scale;
	}
}


/**
 * Scale an array of 3*i doubles by scale.
 *
 * @param n: the number of bodies.
 * @param m: an array of 3*i doubles.
 * @param scale: the multiplicative scale factor.
 */
static inline void scale3NArray_NB(long n, double* m, double scale) {
	for (long i = 0; i < 3*n; ++i) {
		m[i] *= scale;
	}
}


/**
 * Print an array of 3*i doubles.
 *
 * @param n: the number of bodies.
 * @param m: an array of 3*i doubles.
 */
static inline void print3NArray_NB(long n, double* m) {
	for (long i = 0; i < n; ++i) {
		fprintf(stderr, "%.15f, %.15g, %.15g\n", m[3*i], m[3*i+1], m[3*i+2]);
	}
}


/**
 * Compute the total kinetic energy of n masses with certain velocities.
 * @param n: the number of bodies.
 * @param m: an array of n masses.
 * @param v: an array of 3*i doubles (vx, vy, vz);
 *
 * @return the total kinetic energy
 */
static inline double computeEkin_NB(long n, double* m, double* v) {
	double Ekin = 0.0;
	long i;
	for (i = 0; i < n; ++i) {
		Ekin += 0.5 * m[i] * (v[3*i]*v[3*i] + v[3*i+1]*v[3*i+1] + v[3*i+2]*v[3*i+2]);
	}
	return Ekin;
}


/**
 * Compute the total gravitational potential energy of n masses with
 * certain positions.
 * @param n: the number of bodies.
 * @param m: an array of n masses.
 * @param v: an array of 3*i doubles (vx, vy, vz);
 *
 * @return the total kinetic energy
 */
static inline double computeEpot_NB(long n, double* m, double* r) {
	double Epot = 0.0;
	double D, x, y, z;
	long i, j;
	for (i = 0; i < n; ++i) {
		for (j = i+1; j < n; ++j) {
			x = r[3*i + 0] - r[3*j + 0];
			y = r[3*i + 1] - r[3*j + 1]	;
			z = r[3*i + 2] - r[3*j + 2];
			D = sqrt(x*x + y*y + z*z);
			Epot += -1.0*m[i]*m[j] / D;
		}
	}
	return Epot;
}


/**
 * Compute a very rough bounding size of all the positions.
 * Each components of the resulting positions is guaranteed
 * to be within [-R, R] for the return value R.
 *
 * @param n: the number of bodies
 * @param r: a 3*n array of positions.
 */
static inline double computeDomainSize_NB(long n, double* r) {
	long i = 0;

	double max = fabs(r[0]);
	for (i = 1; i < 3*n; ++i) {
		max = MAX(max, fabs(r[i]));
	}
	return ceil(max);
}

#ifdef __cplusplus
}
#endif


#endif
