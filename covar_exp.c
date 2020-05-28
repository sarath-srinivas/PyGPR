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
		    unsigned long nxp, unsigned int dim, const double *p, unsigned int npar,
		    void *dat)
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
					r2 += x_xp * x_xp * (p[k] * p[k]);
				}

				krn[i * nxp + j] = sig_y * sig_y * exp(-r2);
			}
	}
}

void get_dkrn_se_ard(double *dK, unsigned int k, const double *x, const double *kxx,
		     unsigned long nx, unsigned int dim, const double *p, unsigned int np,
		     void *dat)
{
	double pk, xij_d;
	unsigned long i, j, nx2;

	nx2 = nx * nx;

	assert(np == dim + 1);

	if (k < np - 1) {

		/* dK/dl_k */

		pk = p[k];

#pragma omp parallel for collapse(2) default(none)                                                 \
    shared(nx, x, k, dim, dK, pk, kxx) private(i, j, xij_d) schedule(dynamic, CHUNK)
		for (i = 0; i < nx; ++i) {
			for (j = 0; j < nx; ++j) {

				xij_d = x[i * dim + k] - x[j * dim + k];
				dK[i * nx + j] = -2.0 * (xij_d * xij_d * pk) * kxx[i * nx + j];
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

/* TESTS */
double test_get_dkrn_se_ard(unsigned int m, unsigned int dim, unsigned long nx, double eps,
			    int seed)
{
	double *x, *kxx, *kxx_eps, *dk, *dk_num, *p, xmax, err_norm;
	unsigned int np;
	unsigned long i, j, nx2;
	int N, INCX, INCY;
	dsfmt_t drng;

	printf("test_get_dkrn_se_ard(m = %u, dim = %u, nx = %lu, eps = %.1E):\n", m, dim, nx, eps);

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

	get_krn_se_ard(kxx, x, x, nx, nx, dim, p, np, NULL);

	get_dkrn_se_ard(dk, m, x, kxx, nx, dim, p, np, NULL);

	p[m] += eps;

	get_krn_se_ard(kxx_eps, x, x, nx, nx, dim, p, np, NULL);

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