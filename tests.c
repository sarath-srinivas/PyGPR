#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
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

static void f_2d(const double *x, unsigned long ns, double *y, int fno)
{
	double x1, x2;
	unsigned long i;
	unsigned int dim;

	dim = 2;

	for (i = 0; i < ns; i++) {
		x1 = x[dim * i + 0];
		x2 = x[dim * i + 1];

		y[i] = x1 * sin(x2) + x2 * sin(x1);
	}
}

double test_gpr_interpolate(unsigned long ns, unsigned long np, int fno, int seed)
{
	double *x, *xp, *y, *yp, *yt, *p, *var_yp;
	unsigned int dim, npar;
	unsigned long i;

	dim = 2;
	npar = 3;

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
		x[dim * i + 0] = rand() / (RAND_MAX + 1.0);
		x[dim * i + 1] = rand() / (RAND_MAX + 1.0);
	}

	f_2d(x, ns, y, 1);

	srand(seed + 1232);
	for (i = 0; i < np; i++) {
		xp[dim * i + 0] = rand() / (RAND_MAX + 1.0);
		xp[dim * i + 1] = rand() / (RAND_MAX + 1.0);
	}

	f_2d(xp, np, yt, 1);

	for (i = 0; i < npar; i++) {
		p[i] = 1.0;
	}

	gpr_interpolate(xp, yp, np, x, y, ns, dim, p, npar, var_yp, 1);

	for (i = 0; i < np; i++) {
		printf("%+.15E %+.15E %+.15E %+.15E %+.15E %+.15E\n", xp[dim * i + 0],
		       xp[dim * i + 1], yp[i], yt[i], sqrt(var_yp[i * np + i]),
		       fabs(yp[i] - yt[i]));
	}

	free(x);
	free(xp);
	free(y);
	free(yp);
	free(yt);
	free(p);
	free(var_yp);

	return 0;
}

void test_lib_gpr(void)
{
	unsigned int dim;
	unsigned long nx, ns;

	dim = 7;
	nx = 50;
	ns = 100;

	verify(test_get_dkrn_se_ard(0, dim, nx, 1e-6, 343), 1E-6);
	verify(test_get_dkrn_se_ard(1, dim, nx, 1e-6, 343), 1E-6);
	verify(test_get_dkrn_se_ard(2, dim, nx, 1e-6, 343), 1E-6);
	verify(test_get_dkrn_se_ard(3, dim, nx, 1e-6, 343), 1E-6);
	verify(test_get_dkrn_se_ard(4, dim, nx, 1e-6, 343), 1E-6);
	verify(test_get_dkrn_se_ard(5, dim, nx, 1e-6, 343), 1E-6);
	verify(test_get_dkrn_se_ard(6, dim, nx, 1e-6, 343), 1E-6);

	verify(test_jac_cost_fun_ard(0, dim, nx, 1e-6, 363), 1E-6);
	verify(test_jac_cost_fun_ard(1, dim, nx, 1e-6, 363), 1E-6);
	verify(test_jac_cost_fun_ard(2, dim, nx, 1e-6, 363), 1E-6);
	verify(test_jac_cost_fun_ard(3, dim, nx, 1e-6, 363), 1E-6);
	verify(test_jac_cost_fun_ard(4, dim, nx, 1e-6, 363), 1E-6);
	verify(test_jac_cost_fun_ard(5, dim, nx, 1e-6, 363), 1E-6);
	verify(test_jac_cost_fun_ard(6, dim, nx, 1e-6, 363), 1E-6);

	verify(test_asymm_covar(dim, nx, ns, 66), 1E-7);

	verify(test_asymm_covar_jac(0, dim, nx, 1e-6, 363), 1E-6);
	verify(test_asymm_covar_jac(1, dim, nx, 1e-6, 363), 1E-6);
	verify(test_asymm_covar_jac(2, dim, nx, 1e-6, 363), 1E-6);
	verify(test_asymm_covar_jac(3, dim, nx, 1e-6, 363), 1E-6);
	verify(test_asymm_covar_jac(4, dim, nx, 1e-6, 363), 1E-6);
	verify(test_asymm_covar_jac(5, dim, nx, 1e-6, 363), 1E-6);
	verify(test_asymm_covar_jac(6, dim, nx, 1e-6, 363), 1E-6);

	verify(test_jac_cost_fun_ard_asymm(0, dim, nx, 1e-6, 363), 1E-6);
	verify(test_jac_cost_fun_ard_asymm(1, dim, nx, 1e-6, 363), 1E-6);
	verify(test_jac_cost_fun_ard_asymm(2, dim, nx, 1e-6, 363), 1E-6);
	verify(test_jac_cost_fun_ard_asymm(3, dim, nx, 1e-6, 363), 1E-6);
	verify(test_jac_cost_fun_ard_asymm(4, dim, nx, 1e-6, 363), 1E-6);
	verify(test_jac_cost_fun_ard_asymm(5, dim, nx, 1e-6, 363), 1E-6);
	verify(test_jac_cost_fun_ard_asymm(6, dim, nx, 1e-6, 363), 1E-6);
}
