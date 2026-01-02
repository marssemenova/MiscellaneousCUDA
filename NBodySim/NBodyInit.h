
#ifndef _NBODY_INIT_
#define _NBODY_INIT_

#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "util/NBodyHelpers.h"

#ifdef __cplusplus
extern "C" {
#endif


/**
 * A softening parameter for the gravitational potential calculation
 * to avoid huge fluctuations caused by close encounters.
 */
static const double _SOFTENING = 0.025;


/**
 * Compute the acceleration of body i from the mutual graviational
 * potential of the other n-1 bodies.
 *
 * @param n, the number of bodies
 * @param i, the index of the body of which to compute acceleration.
 * @param m, an array of n doubles holding the masses of the bodies
 * @param r, an array of 3*n doubles holding the positions of the bodies
 * @param a, an array of 3*n doubles to hold the resulting acceleration of the bodies.
 */
void computeForce(long n, long i,
    const double* m,
    const double* r,
    double* a)
{
    //         a[k] = -r[k] / (r2*sqrtr2);
    double* ai = a + 3 * i;
    ai[0] = 0.0; ai[1] = 0.0; ai[2] = 0.0;
    double rx = r[3 * i + 0];
    double ry = r[3 * i + 1];
    double rz = r[3 * i + 2];
    double dx, dy, dz, D;
    long j;
    for (j = 0; j < i; ++j) {
        //really dx is other way around, be this way we can avoid -1.0* later.
        dx = r[3 * j] - rx;
        dy = r[3 * j + 1] - ry;
        dz = r[3 * j + 2] - rz;
        D = dx * dx + dy * dy + dz * dz;
        D += _SOFTENING * _SOFTENING;
        D = 1.0 / (D * sqrt(D));
        ai[0] += m[j] * dx * D;
        ai[1] += m[j] * dy * D;
        ai[2] += m[j] * dz * D;
    }
    for (j = i + 1; j < n; ++j) {
        dx = r[3 * j] - rx;
        dy = r[3 * j + 1] - ry;
        dz = r[3 * j + 2] - rz;
        D = dx * dx + dy * dy + dz * dz;
        D += _SOFTENING * _SOFTENING;
        D = 1.0 / (D * sqrt(D));
        ai[0] += m[j] * dx * D;
        ai[1] += m[j] * dy * D;
        ai[2] += m[j] * dz * D;
    }
}


/**
 * Using gravitational potential, compute and update the accelerations
 * of the n bodies in place.
 *
 * @param n: the number of bodies
 * @param m: an array of n doubles holding the masses of the bodies
 * @param r: an array of 3*n doubles holding the positions of the bodies
 * @param[out] a: an array of 3*n doubles to hold the resulting acceleration of the bodies.
 */
static inline void computeForces(long n,
	const double* m,
	const double* r,
	double* a) {
	for (long i = 0; i < n; ++i) {
		computeForce(n, i, m, r, a);
	}
}


/**
 * Allocate an array of doubles of size n*3.
 *
 * @param n: the array length multiplier.
 * @param[out] d_p: a pointer pointing to the array of size 3*n.
 * @return 0 iff the allocation was successful.
 */
int allocData3N_NB(long n, double** d_p) {
    if (d_p != NULL) {
        *d_p = (double*) malloc(sizeof(double)*3*n);
        if (*d_p == NULL) {
            return 1;
        }
    }

    return 0;
}

/**
 * Allocate an array of doubles of size n.
 *
 * @param n: the array length.
 * @param[out] d_p: a pointer pointing to the array of size n.
 * @return 0 iff the allocation was successful.
 */
int allocDataN_NB(long n, double** d_p) {
    if (d_p != NULL) {
        *d_p = (double*) malloc(sizeof(double)*n);
        if (*d_p == NULL) {
            return 1;
        }
    }

    return 0;
}


/**
 * Allocate the necessary arrays describing the n bodies.
 * All arrays returned are 3*n in size, except mass which is n.
 *
 * @param n, the number of bodies to allocate space for.
 * @param[out] r_p: return pointer for the position array.
 * @param[out] v_p: return pointer for the velocity array.
 * @param[out] a_p: return pointer for the acceleration array.
 * @param[out] m_p: return pointer for the mass array.
 * @param[out] work_p: return pointer for the work estiamtes array.
 * @return 0 iff the allocations was successful.
 */
int allocData_NB(long n, double** r_p, double** v_p, double** a_p, double** m_p, double** work_p) {
    //TODO consider using an alternating array here which
    //stores them all interleaved.

    int err = 0;
    err |= allocData3N_NB(n, r_p);
    err |= allocData3N_NB(n, v_p);
    err |= allocData3N_NB(n, a_p);
    err |= allocDataN_NB(n, m_p);
    err |= allocDataN_NB(n, work_p);

    return err;
}


/**
 * Initialize n masses so their masses are equal and sum is 1.0.
 *
 * @param n: the number of bodies.
 * @param m: an array of size n to store the masses.
 * @param M: the total mass of all bodies.
 *
 * @return 0 iff the initialization was successful.
 */
int _initMassEqual(long n, double* m, double M) {
    double mi = M / n;
    long i;

    for (i = 0; i < n; ++i) {
        m[i] = mi;
    }

    return 0;
}


/**
 * Initialize n positions based on the Plummer model.
 * This method is based on Vol. 9 of Hut and Makino, 2005.
 *
 * @param n: the number of bodies.
 * @param r: an array of size 3*i to store the positions.
 * @param Pr: Plummer radius
 *
 * @return 0 iff the initialization was successful.
 */
int _initPositionsPlummer(long n, double* r, double Pr) {
    double R, X, Y;
    double n23 = -2.0 / 3.0;
    double Minv = 1.0; //assume we scale the mass so total mass is 1.0
    long i;
    Pr = Pr*Pr; //We only ever need it squared

    for (i = 0; i < n; ++i) {
        X = pow(frand()*Minv, n23);
        R = Pr / sqrt(X - 1);
        X = acos(1.0 - 2.0*frand()); //acos(-1..1) to get correct random dist.
        Y = frand() * 2.0 * _PI; //phi

        r[3*i + 2] = R * cos(X); //z
        X = sin(X);
        r[3*i + 0] = R * X * cos(Y); //x
        r[3*i + 1] = R * X * sin(Y); //y
    }

    return 0;
}


/**
 * Initialize n positions uniformly throughout the unit sphere.
 *
 * @param n: the number of bodies.
 * @param r: an array of size 3*n to store the positions.
 *
 * @return 0 iff the initialization was successful.
 */
int _initPositionsUniform(long n, double* r) {
    double R, X, Y;
    long i;
    for (i = 0; i < n; ++i) {
        R = frand();
        X = acos(1.0 - 2.0*frand());
        Y = frand()*2.0*_PI;
        r[3*i + 0] = R*sin(X)*cos(Y);
        r[3*i + 1] = R*sin(X)*sin(Y);
        r[3*i + 2] = R*cos(X);
    }

    return 0;
}

/**
 * Initialize n velocities based on the Plummer model
 * and Aaresh's escape velocity criteria.
 * This method is based on Vol. 9 of Hut and Makino, 2005.
 *
 * @param n: the number of bodies.
 * @param r: an array of size 3*i which already stores the positions.
 * @param v: an array of size 3*i to store the velocities.
 *
 * @return 0 iff the initialization was successful.
 */
int _initVelocitiesPlummer(long n, const double* r, double* v) {

    double R2, X, Y, vel;
    double sqrt2 = sqrt(2.0);
    long i;
    for (i = 0; i < n; ++i) {
        R2 = r[3*i]*r[3*i] + r[3*i + 1]*r[3*i + 1] + r[3*i + 2]*r[3*i + 2];

        do {
            X = frand();
            Y = 0.1*frand();
        } while (Y > X*X*pow(1.0-X*X, 3.5));
        vel = sqrt2 * X * pow(1 + R2, -0.25);

        X = acos(1.0 - 2.0*frand()); //acos(-1..1) to get correct random dist.
        Y = frand()*2.0*_PI; //phi
        v[3*i + 2] = vel * cos(X); //z
        X = sin(X);
        v[3*i + 0] = vel * X * cos(Y); //x
        v[3*i + 1] = vel * X * sin(Y); //y
    }

    return 0;
}


/**
 * Initialize the n velocities randomly and uniformly in [-1,1] for each dimension.
 *
 * @param n: the number of velocities.
 * @param v: an array of 3*n doubles for n velocities.
 * @return 0 iff the initialization was successful.
 */
int _initVelocitiesUniform(long n, double* v) {
    long i;
    for (i = 0; i < n; ++i) {
        v[3*i] = (1.0 - 2.0*frand());
        v[3*i + 1] = (1.0 - 2.0*frand());
        v[3*i + 2] = (1.0 - 2.0*frand());
    }

    return 0;
}


/**
 * Normalizes bodies into a reference frame where the center of mass
 * is at the origin with 0 velocity.
 *
 * @param n: the number of bodies
 * @param r: an array of 3*n doubles of positions.
 * @param v: an array of 3*n doubles of velocities.
 * @param m: an array of n doubles of masses.
 * @param M: the sum of masses.
 * @return 0 iff the adjustment was successful.
 */
int _centerOfMassAdjustment(long n, double* r, double* v, double* m, double M) {

    double rx = 0.0, ry = 0.0, rz = 0.0;
    double vx = 0.0, vy = 0.0, vz = 0.0;
    double mi;
    long i;

    for (i = 0; i < n; ++i) {
        mi = m[i];
        rx += r[3*i]*mi;
        ry += r[3*i + 1]*mi;
        rz += r[3*i + 2]*mi;

        vx += v[3*i]*mi;
        vy += v[3*i + 1]*mi;
        vz += v[3*i + 2]*mi;
    }

    rx /= M;
    ry /= M;
    rz /= M;
    vx /= M;
    vy /= M;
    vz /= M;

    for (i = 0; i < n; ++i) {
        r[3*i] -= rx;
        r[3*i + 1] -= ry;
        r[3*i + 2] -= rz;
        v[3*i] -= vx;
        v[3*i + 1] -= vy;
        v[3*i + 2] -= vz;

    }

    return 0;
}


/**
 * Prepare the initial conditions of
 * position, velocity, acceleration, and mass
 * for n bodies.
 *
 * @param n: the number of bodies
 * @param seed: the random seed. If <= 0, use current time.
 * @param r: an array of 3*i doubles to store the positions.
 * @param v: an array of 3*i doubles to store the velocities.
 * @param a: an array of 3*i doubles to store the accelerations.
 * @param m: an array of n doubles to store the masses.
 *
 * @return 0 iff the initialization was successful.
 */
int initData_NB(long n, time_t seed, double* r, double* v, double* a, double* m) {
    if (seed <= 0) {
        seed = time(NULL);
    }
    fprintf(stderr, "seed: %ld\n", seed);
    srand(seed);

    double M = 1.0;
    double Pr = 1.0;
	double virialScale = 16.0 / (_PI * 3.0); //for plummer model

    int error = _initMassEqual(n, m, M);

    if (error) {
        fprintf(stderr, "NBODY: Could not init masses.\n");
        exit(INIT_ERROR);
    }

    if (0) {
        error = _initPositionsPlummer(n, r, Pr);
        error = _initVelocitiesPlummer(n, r, v);
    } else { // uniform
        error = _initPositionsUniform(n, r);
        error = _initVelocitiesUniform(n, v);
    }

    if (error) {
        fprintf(stderr, "NBODY: Could not init nbody positions and velocities.\n");
        exit(INIT_ERROR);
    }

    error = _centerOfMassAdjustment(n, r, v, m, M);

    //Aarseth, 2003, Algorithm 7.2.
    double Epot =  computeEpot_NB(n, m, r);
    double Ekin =  computeEkin_NB(n, m, v);
    double virialRatio = 0.5;
    double Qv = sqrt(virialRatio*fabs(Epot)/Ekin);
    scale3NArray_NB(n, v, Qv);
    double beta = fabs((1 - virialRatio)*Epot/(Epot+Ekin));

    scale3NArray_NB(n, r, beta);
    scale3NArray_NB(n, v, 1.0/(sqrt(beta)));

    //After first scale Ekin is -0.5Epot but E0 != -0.25.
    //So just scale up or down as needed.
    Epot = computeEpot_NB(n, m, r);
    beta = Epot / -0.5;
    scale3NArray_NB(n, r, beta);
    scale3NArray_NB(n, v, 1.0/sqrt(beta));

    if (error) {
        fprintf(stderr, "NBODY: Could not scale to standard units.\n");
        exit(INIT_ERROR);
    }

    computeForces(n, m, r, a);

    long i;
    for (i = 0; i < n; ++i) {
        a[3*i + 0] = 0.0;
        a[3*i + 1] = 0.0;
        a[3*i + 2] = 0.0;
    }

    return (error == 0);
}


/**
 * Chapter 5: Art of computational science vol_1 v1_web.
 */
int initData3BodyChaotic(double* r, double* v, double* a, double* m) {
    m[0] = 1.0;
    m[1] = 1.0;
    m[2] = 1.0;
    long i;
    for (i = 0; i < 3; ++i) {
		double phi = i * 2 * _PI / 3.0;
        r[3*i + 0] = cos(phi);
        r[3*i + 1] = sin(phi);
        r[3*i + 2] = 0.0;
    }

    computeForces(3, m, r, a);

    double v_abs = sqrt(-1.0*a[0]);
    for (i = 0; i < 3; ++i) {
        double phi = i*2*_PI / 3.0;
        v[3*i + 0] = -1.0 * v_abs * sin(phi);
        v[3*i + 1] = v_abs * cos(phi);
        v[3*i + 2] = 0.0;
    }
    v[0] += 0.0001;

    return 0;
}


/**
 * 3Body figure eight stable orbit init.
 */
int initData3BodyFigureEight(double* r, double* v, double* a, double* m) {

    m[0] = 1.0;
    m[1] = 1.0;
    m[2] = 1.0;

    r[3*0 + 0] = 0.9700436;
    r[3*0 + 1] = -0.24308753;
    r[3*0 + 2] = 0.0;
    v[3*0 + 0] = 0.466203685;
    v[3*0 + 1] = 0.43236573;
    v[3*0 + 2] = 0.0;

    r[3*1 + 0] = -r[3*0 + 0];
    r[3*1 + 1] = -r[3*0 + 1];
    r[3*1 + 2] = -r[3*0 + 2];
    v[3*1 + 0] = v[3*0 + 0];
    v[3*1 + 1] = v[3*0 + 1];
    v[3*1 + 2] = v[3*0 + 2];

    r[3*2 + 0] = 0.0;
    r[3*2 + 1] = 0.0;
    r[3*2 + 2] = 0.0;

    v[3*2 + 0] = -2.0 * v[3*0 + 0];
    v[3*2 + 1] = -2.0 * v[3*0 + 1];
    v[3*2 + 2] = -2.0 * v[3*0 + 2];

    v[0] += 0.001;

    computeForces(3, m, r, a);

    return 0;
}



/**
 * Initialize the colors of the bodies for rendering.
 * These colors are based an interpolation of the apparent color
 * of main sequence stars of stellar type O through M.
 *
 * @param n: the number of colors to create.
 * @return an array of n RGBA colors as 4n floats.
 */
static inline float* createColors(long n) {
    //Polynomial fit coefs for color of main sequence stars, index equals monomial degree
    static float Rcoefs[5] = {1.0044296822813448, -0.34970836,  3.01217486, -7.03979628,  4.01061921};
    static float Gcoefs[5] = {0.7105430630750322, 0.35529946,  3.21427637, -8.17669968,  4.63094998};
    static float Bcoefs[6] = {0.47835976470247865, -0.88069571,  16.68565124, -44.11776039,  44.98967698, -16.16596219};
    float* color_data = (float*) malloc(sizeof(float)*4*n);
    float X[6];
    int k;
    for (long i = 0; i < n; ++i) {
        //X acts as an interpolation of spectral type between M and O as [0,1]
        X[1] = frand();
        for (k = 2; k < 6; ++k) {
            X[k] = X[k-1]*X[1];
        }
        color_data[4*i + 0] = Rcoefs[0] + X[1]*Rcoefs[1] + X[2]*Rcoefs[2] + X[3]*Rcoefs[3] + X[4]*Rcoefs[4];
        color_data[4*i + 1] = Gcoefs[0] + X[1]*Gcoefs[1] + X[2]*Gcoefs[2] + X[3]*Gcoefs[3] + X[4]*Gcoefs[4];
        color_data[4*i + 2] = Bcoefs[0] + X[1]*Bcoefs[1] + X[2]*Bcoefs[2] + X[3]*Bcoefs[3] + X[4]*Bcoefs[4] + X[5]*Bcoefs[5];
        color_data[4*i + 3] = 1.0;
    }
    return color_data;
}

#ifdef __cplusplus
}
#endif

#endif
