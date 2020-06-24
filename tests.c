#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_rng/lib_rng.h>
#include "lib_gpr.h"

static int verify(double terr, double tol)
{
	int ret;
	if (terr > tol) {
		ret = 1;
		printf("T-ERROR: %+.15E TOL: %+.0E TEST FAILED  ***\n\n", terr, tol);
	} else {
		ret = 0;
		printf("T-ERROR: %+.15E TOL: %+.0E TEST PASSED\n\n", terr, tol);
	}

	return ret;
}

static void f_nd(double *y, const double *x, unsigned long ns, unsigned int dim)
{
	double x1, x2, r;
	unsigned long i, k;

	for (i = 0; i < ns; i++) {

		r = 0;
		for (k = 0; k < dim; k++) {

			r += x[dim * i + k];

			y[i] = exp(-r * r);
		}
	}
}

double test_gpr_interpolate(unsigned long ns, unsigned long np, unsigned int dim, int seed)
{
	double *x, *xp, *y, *yp, *yt, *p, *var_yp, err;
	unsigned int npar;
	unsigned long i, j, k;

	printf("test_gpr_interpolate(nke = %lu, nq = %lu, dim = %u):\n", ns, np, dim);

	npar = dim + 1;

	x = malloc(dim * ns * sizeof(double));
	assert(x);

	xp = malloc(dim * np * sizeof(double));
	assert(xp);

	y = malloc(ns * sizeof(double));
	assert(y);

	yp = malloc(np * sizeof(double));
	assert(yp);

	yt = malloc(np * sizeof(double));
	assert(yt);

	p = malloc(npar * sizeof(double));
	assert(p);

	var_yp = malloc(np * np * sizeof(double));
	assert(p);

	srand(seed);
	for (i = 0; i < ns; i++) {
		for (k = 0; k < dim; k++) {
			x[dim * i + k] = rand() / (RAND_MAX + 1.0);
		}
	}

	f_nd(y, x, ns, dim);

	srand(seed + 1232);
	for (i = 0; i < np; i++) {
		for (k = 0; k < dim; k++) {
			xp[dim * i + k] = rand() / (RAND_MAX + 1.0);
		}
	}

	f_nd(yt, xp, np, dim);

	for (i = 0; i < npar; i++) {
		p[i] = 1.0;
	}

	/*
	gpr_interpolate(xp, yp, np, x, y, ns, dim, p, npar, var_yp, 1);
	*/

	printf("GPR INTERPOLATE:\n");

	gpr_interpolate_wrap(xp, yp, np, x, y, ns, dim, p, npar, var_yp, 1, get_krn_se_ard,
			     get_dkrn_se_ard, NULL);

	err = 0;
	for (i = 0; i < np; i++) {
		err += fabs(yp[i] - yt[i]);
	}

	if (DEBUG) {
		for (i = 0; i < np; i++) {
			printf("%+.15E %+.15E %+.15E %+.15E %+.15E %+.15E\n", xp[dim * i + 0],
			       xp[dim * i + 1], yp[i], yt[i], sqrt(var_yp[i * np + i]),
			       fabs(yp[i] - yt[i]));
		}
	}

	free(x);
	free(xp);
	free(y);
	free(yp);
	free(yt);
	free(p);
	free(var_yp);

	return err;
}

double test_gpr_interpolate_experts(unsigned long nsc, unsigned long nc, unsigned long np,
				    unsigned int dim, double min_dist, unsigned int gate, int seed)
{
	double *x, *gx, *xp, *y, *yp, *yt, *p, *var_yp, err, *xc, *st, *en;
	unsigned int npar;
	unsigned long i, j, k, ns;
	long count;

	printf(
	    "test_gpr_interpolate_experts(nsc = %lu, nc = %lu, np = %lu, dim = %u, gate = %u):\n",
	    nsc, nc, np, dim, gate);

	npar = nc * (dim + 1);

	ns = nsc * nc;

	x = malloc(3 * dim * ns * sizeof(double));
	assert(x);

	xc = malloc(dim * nc * sizeof(double));
	assert(xc);

	gx = malloc(dim * ns * sizeof(double));
	assert(gx);

	xp = malloc(dim * np * sizeof(double));
	assert(xp);

	y = malloc(ns * sizeof(double));
	assert(y);

	yp = malloc(np * sizeof(double));
	assert(yp);

	yt = malloc(np * sizeof(double));
	assert(yt);

	p = malloc(npar * sizeof(double));
	assert(p);

	st = malloc(dim * sizeof(double));
	assert(st);
	en = malloc(dim * sizeof(double));
	assert(en);

	var_yp = malloc(np * np * sizeof(double));
	assert(var_yp);

	for (i = 0; i < dim; i++) {
		st[i] = 0;
		en[i] = 1.0;
	}

	fill_random(x, 3 * ns, dim, st, en, seed);

	count = get_gibbs_hrdcr_samples(xc, nc, dim, st, en, min_dist, seed + 3535, 5000);

	printf("COUNT:%ld\n", count);

	fill_random_groups(gx, ns, xc, nc, x, 3 * ns, dim, st, en);

	f_nd(y, gx, ns, dim);

	fill_random(xp, np, dim, st, en, seed);

	f_nd(yt, xp, np, dim);

	for (i = 0; i < npar; i++) {
		p[i] = 1.0;
	}

	/*
	gpr_interpolate(xp, yp, np, x, y, ns, dim, p, npar, var_yp, 1);
	*/

	printf("GPR INTERPOLATE:\n");

	gpr_interpolate_experts(yp, var_yp, xp, np, gx, y, ns, nc, dim, p, npar, 1, get_krn_se_ard,
				get_dkrn_se_ard, NULL, gate);

	err = 0;
	for (i = 0; i < np; i++) {
		err += fabs(yp[i] - yt[i]);
	}

	if (DEBUG) {
		for (i = 0; i < np; i++) {
			printf("%+.15E %+.15E %+.15E %+.15E %+.15E %+.15E\n", xp[dim * i + 0],
			       xp[dim * i + 1], yp[i], yt[i], sqrt(var_yp[i * np + i]),
			       fabs(yp[i] - yt[i]));
		}
	}

	free(var_yp);
	free(en);
	free(st);
	free(p);
	free(yt);
	free(yp);
	free(y);
	free(xp);
	free(gx);
	free(xc);
	free(x);

	return err;
}

void test_lib_gpr(void)
{
	unsigned int dim;
	unsigned long nx, ns, nsc, np, nc;

	dim = 7;
	nx = 50;
	ns = 100;

	verify(test_gpr_interpolate(10, 100, 2, 34366), 1E-7);

	nsc = 10;
	nc = 4;
	np = 100;
	dim = 2;

	verify(test_gpr_interpolate_experts(nsc, ns, np, dim, 0.3, 0, 34366), 1E-7);
	verify(test_gpr_interpolate_experts(nsc, ns, np, dim, 0.3, 1, 34366), 1E-7);
	verify(test_gpr_interpolate_experts(nsc, ns, np, dim, 0.3, 2, 34366), 1E-7);

	dim = 7;

	verify(test_get_dkrn_se_ard(0, dim, nx, 1e-6, 343), 1E-6);
	verify(test_get_dkrn_se_ard(1, dim, nx, 1e-6, 343), 1E-6);
	verify(test_get_dkrn_se_ard(2, dim, nx, 1e-6, 343), 1E-6);
	verify(test_get_dkrn_se_ard(3, dim, nx, 1e-6, 343), 1E-6);
	verify(test_get_dkrn_se_ard(4, dim, nx, 1e-6, 343), 1E-6);
	verify(test_get_dkrn_se_ard(5, dim, nx, 1e-6, 343), 1E-6);
	verify(test_get_dkrn_se_ard(6, dim, nx, 1e-6, 343), 1E-6);
	verify(test_get_dkrn_se_ard(7, dim, nx, 1e-6, 343), 1E-6);

	dim = 3;

	verify(test_get_dkrn_sin_ard(0, dim, nx, 1e-6, 343), 1E-6);
	verify(test_get_dkrn_sin_ard(1, dim, nx, 1e-6, 343), 1E-6);
	verify(test_get_dkrn_sin_ard(2, dim, nx, 1e-6, 343), 1E-6);
	verify(test_get_dkrn_sin_ard(3, dim, nx, 1e-6, 343), 1E-6);
	verify(test_get_dkrn_sin_ard(4, dim, nx, 1e-6, 343), 1E-6);
	verify(test_get_dkrn_sin_ard(5, dim, nx, 1e-6, 343), 1E-6);
	verify(test_get_dkrn_sin_ard(6, dim, nx, 1e-6, 343), 1E-6);

	dim = 7;

	verify(test_jac_cost_fun_ard(0, dim, nx, 1e-6, 363), 1E-6);
	verify(test_jac_cost_fun_ard(1, dim, nx, 1e-6, 363), 1E-6);
	verify(test_jac_cost_fun_ard(2, dim, nx, 1e-6, 363), 1E-6);
	verify(test_jac_cost_fun_ard(3, dim, nx, 1e-6, 363), 1E-6);
	verify(test_jac_cost_fun_ard(4, dim, nx, 1e-6, 363), 1E-6);
	verify(test_jac_cost_fun_ard(5, dim, nx, 1e-6, 363), 1E-6);
	verify(test_jac_cost_fun_ard(6, dim, nx, 1e-6, 363), 1E-6);
	verify(test_jac_cost_fun_ard(7, dim, nx, 1e-6, 363), 1E-6);

	verify(test_symm_covar(1, dim, nx, ns, 66), 1E-7);
	verify(test_symm_covar(-1, dim, nx, ns, 66), 1E-7);

	verify(test_symm_covar_jac(1, 0, dim, nx, 1e-6, 363), 1E-6);
	verify(test_symm_covar_jac(1, 1, dim, nx, 1e-6, 363), 1E-6);
	verify(test_symm_covar_jac(1, 2, dim, nx, 1e-6, 363), 1E-6);
	verify(test_symm_covar_jac(1, 3, dim, nx, 1e-6, 363), 1E-6);
	verify(test_symm_covar_jac(1, 4, dim, nx, 1e-6, 363), 1E-6);
	verify(test_symm_covar_jac(1, 5, dim, nx, 1e-6, 363), 1E-6);
	verify(test_symm_covar_jac(1, 6, dim, nx, 1e-6, 363), 1E-6);
	verify(test_symm_covar_jac(1, 7, dim, nx, 1e-6, 363), 1E-6);

	verify(test_symm_covar_jac(-1, 0, dim, nx, 1e-6, 363), 1E-6);
	verify(test_symm_covar_jac(-1, 1, dim, nx, 1e-6, 363), 1E-6);
	verify(test_symm_covar_jac(-1, 2, dim, nx, 1e-6, 363), 1E-6);
	verify(test_symm_covar_jac(-1, 3, dim, nx, 1e-6, 363), 1E-6);
	verify(test_symm_covar_jac(-1, 4, dim, nx, 1e-6, 363), 1E-6);
	verify(test_symm_covar_jac(-1, 5, dim, nx, 1e-6, 363), 1E-6);
	verify(test_symm_covar_jac(-1, 6, dim, nx, 1e-6, 363), 1E-6);
	verify(test_symm_covar_jac(-1, 7, dim, nx, 1e-6, 363), 1E-6);

	test_get_subsample_cv_holdout(100, 10, 2, 3, 4546);
	test_get_gpr_cv_holdout(100, 3, 10, 5, 0, 356);
}
