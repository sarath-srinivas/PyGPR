#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <lib_rng/lib_rng.h>
#include <blas/blas.h>
#include "lib_gpr.h"

#define CHUNK (100)

void get_krn_sin_ard(double *krn, const double *x, const double *xp, unsigned long nx,
		     unsigned long nxp, unsigned int dim, const double *p, unsigned int npar,
		     void *dat)
{
	double sig_y, l, l2, r2, sin_x_xp, pexp, w2, c2;
	unsigned long i, j, k;

	assert(npar == 2 * dim + 1);

	sig_y = p[2 * dim];

#pragma omp parallel
	{
#pragma omp parallel for collapse(2) default(none)                                                 \
    shared(nx, nxp, dim, x, xp, p, krn, sig_y) private(i, j, k, w2, c2, r2, sin_x_xp)              \
	schedule(dynamic, CHUNK)
		for (i = 0; i < nx; i++)
			for (j = 0; j < nxp; j++) {

				r2 = 0;
				for (k = 0; k < dim; k++) {

					c2 = p[k] * p[k];
					w2 = p[dim + k] * p[dim + k];

					sin_x_xp = sin(w2 * (x[dim * i + k] - xp[dim * j + k]));
					r2 += c2 * sin_x_xp * sin_x_xp;
				}

				krn[i * nxp + j] = sig_y * sig_y * exp(-r2);
			}
	}
}

void get_dkrn_sin_ard(double *dK, unsigned int k, const double *x, const double *kxx,
		      unsigned long nx, unsigned int dim, const double *p, unsigned int np,
		      void *dat)
{
	double pk, sin_xij_d, xij_d, c, c2, w, w2;
	unsigned long i, j, nx2;

	nx2 = nx * nx;

	assert(np == 2 * dim + 1);

	if (k < dim) {

		/* dK/dl_k */

		c = p[k];
		c2 = c * c;
		w = p[dim + k];
		w2 = w * w;

#pragma omp parallel for collapse(2) default(none)                                                 \
    shared(nx, x, k, dim, dK, c, c2, w, w2, kxx) private(i, j, sin_xij_d) schedule(dynamic, CHUNK)
		for (i = 0; i < nx; ++i) {
			for (j = 0; j < nx; ++j) {

				sin_xij_d = sin(w2 * (x[i * dim + k] - x[j * dim + k]));
				dK[i * nx + j] = -2.0 * c * sin_xij_d * sin_xij_d * kxx[i * nx + j];
			}
		}

	} else if (k < 2 * dim) {

		c = p[k - dim];
		c2 = c * c;
		w = p[k];
		w2 = w * w;

#pragma omp parallel for collapse(2) default(none)                                                 \
    shared(nx, x, k, dim, dK, c, c2, w, w2, kxx) private(i, j, xij_d, sin_xij_d)                   \
	schedule(dynamic, CHUNK)
		for (i = 0; i < nx; ++i) {
			for (j = 0; j < nx; ++j) {

				xij_d = x[i * dim + k - dim] - x[j * dim + k - dim];
				sin_xij_d = sin(2 * w2 * xij_d);
				dK[i * nx + j]
				    = -2.0 * c2 * w * xij_d * sin_xij_d * kxx[i * nx + j];
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

double test_get_dkrn_sin_ard(unsigned int m, unsigned int dim, unsigned long nx, double eps,
			     int seed)
{
	double *x, *kxx, *kxx_eps, *dk, *dk_num, *p, xmax, err_norm;
	unsigned int np;
	unsigned long i, j, nx2;
	int N, INCX, INCY;
	dsfmt_t drng;

	printf("test_get_dkrn_sin_ard(m = %u, dim = %u, nx = %lu, eps = %.1E):\n", m, dim, nx, eps);

	np = 2 * dim + 1;
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

	get_krn_sin_ard(kxx, x, x, nx, nx, dim, p, np, NULL);

	get_dkrn_sin_ard(dk, m, x, kxx, nx, dim, p, np, NULL);

	p[m] += eps;

	get_krn_sin_ard(kxx_eps, x, x, nx, nx, dim, p, np, NULL);

	for (i = 0; i < nx2; i++) {

		dk_num[i] = (kxx_eps[i] - kxx[i]) / eps;
	}

	err_norm = 0;
	for (i = 0; i < nx2; i++) {
		err_norm += (dk_num[i] - dk[i]) * (dk_num[i] - dk[i]);
	}

	err_norm /= nx2;

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
