#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <lib_rng/lib_rng.h>
#include <blas/blas.h>
#include "lib_gpr.h"

#define CHUNK (100)

void get_krn_se_ard(double *krn, const double *x, const double *xp, unsigned long nx,
		    unsigned long nxp, unsigned long dim, const double *p, int npar)
{
	double sig_y, l, l2, r2, x_xp;
	unsigned long i, j, k;

	assert(npar == dim + 1);

	sig_y = p[dim];

#pragma omp parallel
	{
#pragma omp parallel for collapse(2) default(none)                                                 \
    shared(nx, nxp, dim, x, xp, p, krn, sig_y) private(i, j, k, r2, x_xp) schedule(dynamic, CHUNK)
		for (i = 0; i < nx; i++)
			for (j = 0; j < nxp; j++) {

				r2 = 0;
				for (k = 0; k < dim; k++) {
					x_xp = x[dim * i + k] - xp[dim * j + k];
					r2 += x_xp * x_xp / (p[k] * p[k]);
				}

				krn[i * nxp + j] = sig_y * sig_y * exp(-r2);
			}
	}
}

void get_dkrn_se_ard(double *dK, int k, const double *x, const double *kxx, unsigned long nx,
		     unsigned int dim, const double *p, int np)
{
	double ld3, xij_d;
	unsigned long i, j, nx2;

	nx2 = nx * nx;

	assert(np == dim + 1);

	if (k < np - 1) {

		/* dK/dl_k */

		ld3 = p[k] * p[k] * p[k];

#pragma omp parallel for collapse(2) default(none)                                                 \
    shared(nx, x, k, dim, dK, ld3, kxx) private(i, j, xij_d) schedule(dynamic, CHUNK)
		for (i = 0; i < nx; ++i) {
			for (j = 0; j < nx; ++j) {

				xij_d = x[i * dim + k] - x[j * dim + k];
				dK[i * nx + j] = 2.0 * (xij_d * xij_d / ld3) * kxx[i * nx + j];
			}
		}

	} else {

		/* dK/dsigma */

#pragma omp parallel for default(none) shared(nx2, p, k, dK, kxx) private(i)                       \
    schedule(dynamic, CHUNK)
		for (i = 0; i < nx2; i++) {
			dK[i] = (2.0 / p[k]) * kxx[i];
		}
	}
}

void get_krn_rat_quad(double *krn, const double *x, const double *xp, unsigned long nx,
		      unsigned long nxp, unsigned long dim, const double *par, int npar,
		      const double *hpar, int nhpar)
{
	double sig_y, l, l2, r2, x_xp, alph;
	unsigned long i, j, k;

	assert(npar == 1);
	assert(nhpar == 1);

	sig_y = 1.0;
	l = hpar[0];
	alph = par[0];
	l2 = l * l;

	for (i = 0; i < nx; i++)
		for (j = 0; j < nxp; j++) {

			r2 = 0;
			for (k = 0; k < dim; k++) {
				x_xp = x[dim * i + k] - xp[dim * j + k];
				r2 += x_xp * x_xp;
			}

			krn[i * nxp + j] = sig_y * sig_y * pow((1 + r2 / (2 * alph * l2)), -alph);
		}
}

/* TESTS */
double test_get_dkrn_se_ard(unsigned int m, unsigned int dim, unsigned long nx, double eps,
			    int seed)
{
	double *x, *kxx, *kxx_eps, *dk, *dk_num, *p, xmax, err_norm;
	unsigned int np;
	unsigned long i, j, nx2;
	int N, INCX, INCY;
	dsfmt_t drng;

	np = dim + 1;
	assert(m <= np);

	nx2 = nx * nx;

	x = malloc(dim * nx * sizeof(double));
	assert(x);
	p = malloc(np * sizeof(double));
	assert(p);

	kxx = malloc(nx * nx * sizeof(double));
	assert(kxx);
	kxx_eps = malloc(nx * nx * sizeof(double));
	assert(kxx_eps);
	dk = malloc(nx * nx * sizeof(double));
	assert(dk);
	dk_num = malloc(nx * nx * sizeof(double));
	assert(dk_num);

	dsfmt_init_gen_rand(&drng, seed);

	xmax = 2.0;

	for (i = 0; i < nx; i++) {
		for (j = 0; j < dim; j++) {
			x[i * dim + j] = xmax * dsfmt_genrand_close_open(&drng);
		}
	}

	dsfmt_init_gen_rand(&drng, seed + 34);

	for (i = 0; i < np; i++) {
		p[i] = 0.5 + dsfmt_genrand_close_open(&drng);
	}

	get_krn_se_ard(kxx, x, x, nx, nx, dim, p, np);

	get_dkrn_se_ard(dk, m, x, kxx, nx, dim, p, np);

	p[m] += eps;

	get_krn_se_ard(kxx_eps, x, x, nx, nx, dim, p, np);

	for (i = 0; i < nx2; i++) {

		dk_num[i] = (kxx_eps[i] - kxx[i]) / eps;
	}

	err_norm = 0;
	for (i = 0; i < nx2; i++) {
		err_norm += (dk_num[i] - dk[i]) * (dk_num[i] - dk[i]);
	}

	if (DEBUG == 1) {
		for (i = 0; i < nx2; i++) {
			printf("%+.15E %+.15E\n", dk[i], dk_num[i]);
		}
	}

	free(kxx);
	free(kxx_eps);
	free(dk);
	free(dk_num);
	free(p);
	free(x);

	return sqrt(err_norm);
}
