#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <omp.h>
#include <lib_rng/lib_rng.h>
#include <lib_flow/lib_flow.h>
#include "lib_gpr.h"

#define CHUNK (100)

void get_asymm_covar(double *krn, const double *x, const double *xp, double *ax, double *axp,
		     unsigned long nx, unsigned long nxp, unsigned long dim, const double *p,
		     int npar)
{
	double sig2, l, l2, r2[4], d[4];
	unsigned long i, j, k;

	assert(npar == dim + 1);

	sig2 = p[dim] * p[dim];

#pragma omp parallel
	{
#pragma omp parallel for collapse(2) default(none)                                                 \
    shared(nx, nxp, dim, x, xp, ax, axp, p, krn, sig2) private(i, j, k, r2, d)                     \
	schedule(dynamic, CHUNK)
		for (i = 0; i < nx; i++)
			for (j = 0; j < nxp; j++) {

				r2[0] = 0;
				r2[1] = 0;
				r2[2] = 0;
				r2[3] = 0;

				for (k = 0; k < dim; k++) {

					d[0] = x[dim * i + k] - xp[dim * j + k];
					d[1] = x[dim * i + k] - axp[dim * j + k];
					d[2] = ax[dim * i + k] - xp[dim * j + k];
					d[3] = ax[dim * i + k] - axp[dim * j + k];

					r2[0] += d[0] * d[0] * (p[k] * p[k]);
					r2[1] += d[1] * d[1] * (p[k] * p[k]);
					r2[2] += d[2] * d[2] * (p[k] * p[k]);
					r2[3] += d[3] * d[3] * (p[k] * p[k]);
				}

				krn[i * nxp + j]
				    = sig2
				      * (exp(-r2[0]) - exp(-r2[1]) - exp(-r2[2]) + exp(-r2[3]));
			}
	}
}

void get_asymm_covar_jac(double *dK, unsigned int m, const double *kxx, const double *x,
			 const double *ax, unsigned long nx, unsigned long dim, const double *p,
			 int npar)
{
	double sig2, sig, l, l2, r2[4], d[4], dm[4], pm, pm2;
	unsigned long i, j, k, nx2;

	assert(npar == dim + 1);

	nx2 = nx * nx;

	sig = p[dim];
	sig2 = p[dim] * p[dim];

	if (m < npar - 1) {

		pm2 = p[m] * p[m];
		pm = p[m];

#pragma omp parallel
		{
#pragma omp parallel for collapse(2) default(none)                                                 \
    shared(m, nx, dim, x, ax, p, pm, dK, sig2) private(i, j, k, r2, d, dm)                         \
	schedule(dynamic, CHUNK)
			for (i = 0; i < nx; i++)
				for (j = 0; j < nx; j++) {

					r2[0] = 0;
					r2[1] = 0;
					r2[2] = 0;
					r2[3] = 0;

					for (k = 0; k < dim; k++) {

						d[0] = x[dim * i + k] - x[dim * j + k];
						d[1] = x[dim * i + k] - ax[dim * j + k];
						d[2] = ax[dim * i + k] - x[dim * j + k];
						d[3] = ax[dim * i + k] - ax[dim * j + k];

						r2[0] += d[0] * d[0] * p[k] * p[k];
						r2[1] += d[1] * d[1] * p[k] * p[k];
						r2[2] += d[2] * d[2] * p[k] * p[k];
						r2[3] += d[3] * d[3] * p[k] * p[k];
					}

					dm[0] = x[dim * i + m] - x[dim * j + m];
					dm[1] = x[dim * i + m] - ax[dim * j + m];
					dm[2] = ax[dim * i + m] - x[dim * j + m];
					dm[3] = ax[dim * i + m] - ax[dim * j + m];

					dK[i * nx + j] = -2.0 * sig2 * pm
							 * (dm[0] * dm[0] * exp(-r2[0])
							    - dm[1] * dm[1] * exp(-r2[1])
							    - dm[2] * dm[2] * exp(-r2[2])
							    + dm[3] * dm[3] * exp(-r2[3]));
				}
		}
	}

	else {
		/* dK/dsigma */

#pragma omp parallel for default(none) shared(nx2, sig, dK, kxx) private(i) schedule(dynamic, CHUNK)
		for (i = 0; i < nx2; i++) {
			dK[i] = (2.0 / sig) * kxx[i];
		}
	}
}

/* TESTS */

double test_asymm_covar(unsigned int dim, unsigned long nx, unsigned long ns, int seed)
{
	double *x, *ax, *xs, *axs, *kxx, *akxx, *p, xmax, err_norm;
	unsigned int np;
	unsigned long i, j, nx2;
	int N, INCX, INCY;
	dsfmt_t drng;

	printf("test_asymm_covar(dim = %u, nx = %lu, ns = %lu):\n", dim, nx, ns);

	np = dim + 1;

	nx2 = nx * ns;

	x = malloc(dim * nx * sizeof(double));
	assert(x);
	ax = malloc(dim * nx * sizeof(double));
	assert(ax);
	xs = malloc(dim * ns * sizeof(double));
	assert(xs);
	axs = malloc(dim * ns * sizeof(double));
	assert(axs);
	p = malloc(np * sizeof(double));
	assert(p);

	kxx = malloc(nx * ns * sizeof(double));
	assert(kxx);
	akxx = malloc(nx * ns * sizeof(double));
	assert(akxx);

	dsfmt_init_gen_rand(&drng, seed);

	xmax = 2.0;

	for (i = 0; i < ns; i++) {
		for (j = 0; j < dim; j++) {
			xs[i * dim + j] = xmax * dsfmt_genrand_close_open(&drng);
		}
	}

	for (i = 0; i < nx; i++) {
		for (j = 0; j < dim; j++) {
			x[i * dim + j] = xmax * dsfmt_genrand_close_open(&drng);
		}
	}

	get_ph_ex_sample(ax, x, nx, dim);
	get_ph_ex_sample(axs, xs, ns, dim);

	dsfmt_init_gen_rand(&drng, seed + 34);

	for (i = 0; i < np; i++) {
		p[i] = 0.5 + dsfmt_genrand_close_open(&drng);
	}

	get_asymm_covar(kxx, x, xs, ax, axs, nx, ns, dim, p, np);

	get_asymm_covar(akxx, ax, xs, x, axs, nx, ns, dim, p, np);

	err_norm = 0;
	for (i = 0; i < nx2; i++) {
		err_norm += fabs(kxx[i] + akxx[i]);
	}

	err_norm /= nx2;

	free(akxx);
	free(kxx);
	free(p);
	free(axs);
	free(xs);
	free(ax);
	free(x);

	return sqrt(err_norm);
}

double test_asymm_covar_jac(unsigned int m, unsigned int dim, unsigned long nx, double eps,
			    int seed)
{
	double *x, *ax, *kxx, *kxx_eps, *dk, *dk_num, *p, xmax, err_norm;
	unsigned int np;
	unsigned long i, j, nx2;
	int N, INCX, INCY;
	dsfmt_t drng;

	printf("test_asymm_covar_jac(m = %u, dim = %u, nx = %lu, eps = %.1E):\n", m, dim, nx, eps);

	np = dim + 1;
	assert(m <= np);

	nx2 = nx * nx;

	x = malloc(dim * nx * sizeof(double));
	assert(x);
	ax = malloc(dim * nx * sizeof(double));
	assert(ax);
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

	get_ph_ex_sample(ax, x, nx, dim);

	dsfmt_init_gen_rand(&drng, seed + 34);

	for (i = 0; i < np; i++) {
		p[i] = 0.5 + dsfmt_genrand_close_open(&drng);
	}

	get_asymm_covar(kxx, x, x, ax, ax, nx, nx, dim, p, np);

	get_asymm_covar_jac(dk, m, kxx, x, ax, nx, dim, p, np);

	p[m] += eps;

	get_asymm_covar(kxx_eps, x, x, ax, ax, nx, nx, dim, p, np);

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
