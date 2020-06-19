#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_rng/lib_rng.h>
#include <blas/blas.h>
#include <blas/lapack.h>
#include "lib_gpr.h"

double sin_nd(const double *x, unsigned int dim)
{
	unsigned int i;
	double y;

	y = 0;
	for (i = 0; i < dim; i++) {
		y += (1.0 + i) / 2.0 * x[i];
	}

	y = sin(y);

	return y;
}

double get_rel_rmse(const double *y, const double *y_pred, unsigned long n)
{
	double rel_diff, rel_rmse;
	unsigned long i;

	rel_rmse = 0;
	for (i = 0; i < n; i++) {

		rel_diff = (y[i] - y_pred[i]) / y[i];
		rel_rmse += rel_diff * rel_diff;
	}

	rel_rmse = sqrt(rel_rmse) / n;

	return rel_rmse;
}

double get_chi_sq(const double *y, const double *y_pred, const double *covar, unsigned long n)
{
	double rel_diff, chi_sq;
	unsigned long i;

	chi_sq = 0;
	for (i = 0; i < n; i++) {

		rel_diff = (y[i] - y_pred[i]);
		chi_sq += rel_diff * rel_diff / covar[n * i + i];
	}

	chi_sq /= 1;

	return chi_sq;
}

double get_mhlbs_dist(const double *y, const double *y_pred, const double *covar, unsigned long n)
{
	double *diff, *kdiff, *covar_chd, mhlbs_dist;
	unsigned long i;
	long int Nl;
	int N, NRHS, LDA, LDB, info, INCX, INCY;
	unsigned char UPLO;

	diff = malloc(n * sizeof(double));
	assert(diff);
	kdiff = malloc(n * sizeof(double));
	assert(kdiff);
	covar_chd = malloc(n * n * sizeof(double));
	assert(covar_chd);

	for (i = 0; i < n; i++) {
		diff[i] = y[i] - y_pred[i];
		kdiff[i] = diff[i];
	}

	Nl = n * n;
	INCX = 1;
	INCY = 1;
	dcopy_(&Nl, covar, &INCX, covar_chd, &INCY);

	UPLO = 'L';
	N = (int)n;
	Nl = N;
	LDA = (int)n;
	LDB = (int)n;
	NRHS = 1;
	dposv_(&UPLO, &N, &NRHS, covar_chd, &LDA, kdiff, &LDB, &info);
	assert(info == 0);

	mhlbs_dist = ddot_(&Nl, diff, &INCX, kdiff, &INCY);

	mhlbs_dist /= 1;

	return mhlbs_dist;
}

void get_subsample_cv_holdout(double *ytst, double *xtst, unsigned long ntst, double *ytrn,
			      double *xtrn, unsigned long ntrn, const double *y, const double *x,
			      unsigned long n, unsigned int dim, unsigned long k)
{
	unsigned long nsub, i, j, l, m, d;

	assert(n % ntst == 0 && "Subsample does not divide sample!!");
	assert(ntrn == n - ntst);

	/*
	nsub = n / ntst;
	*/

	for (j = 0; j < ntst; j++) {

		for (d = 0; d < dim; d++) {
			xtst[j * dim + d] = x[(k * ntst + j) * dim + d];
		}

		ytst[j] = y[k * ntst + j];
	}

	m = 0;
	for (l = 0; l < n; l++) {

		for (d = 0; d < dim; d++) {
			xtrn[m * dim + d] = x[l * dim + d];
		}

		ytrn[m] = y[l];

		if (l < k * ntst || (k + 1) * ntst <= l) {
			m++;
		}
	}
}

/* GPR WITHOUT MEAN */

void get_gpr_cv_holdout_batch(double *cv, unsigned long k, unsigned long ntst, const double *x,
			      const double *y, unsigned long n, unsigned int dim, double *hp,
			      unsigned long nhp, const enum estimator est)
{
	double *ytrn, *xtrn, *ytst, *ytst_gpr, *xtst, diff, *covar, cv_holdout;
	unsigned long ntrn, i;

	ntrn = n - ntst;

	ytrn = malloc(ntrn * sizeof(double));
	assert(ytrn);
	ytst = malloc(ntst * sizeof(double));
	assert(ytst);
	ytst_gpr = malloc(ntst * sizeof(double));
	assert(ytst_gpr);

	xtrn = malloc(ntrn * dim * sizeof(double));
	assert(xtrn);
	xtst = malloc(ntst * dim * sizeof(double));
	assert(xtst);

	covar = malloc(ntst * ntst * sizeof(double));
	assert(covar);

	get_subsample_cv_holdout(ytst, xtst, ntst, ytrn, xtrn, ntrn, y, x, n, dim, k);

	gpr_interpolate(xtst, ytst_gpr, ntst, xtrn, ytrn, ntrn, dim, hp, nhp, covar, 0);

	cv[0] = get_rel_rmse(ytst, ytst_gpr, ntst);
	cv[1] = get_chi_sq(ytst, ytst_gpr, covar, ntst);
	cv[2] = get_mhlbs_dist(ytst, ytst_gpr, covar, ntst);

	free(xtrn);
	free(xtst);
	free(ytrn);
	free(ytst);
	free(ytst_gpr);
	free(covar);
}

void get_gpr_cv_holdout(double *cv_btch, const double *x, const double *y, unsigned long n,
			unsigned int dim, double *hp, unsigned long nhp, unsigned long ntst,
			unsigned long nbtch, enum estimator est)
{
	double cv[3];
	unsigned long k;

	assert(n % ntst == 0);

	cv[0] = 0;
	cv[1] = 0;
	cv[2] = 0;
	for (k = 0; k < nbtch; k++) {

		get_gpr_cv_holdout_batch(cv, k, ntst, x, y, n, dim, hp, nhp, est);

		cv_btch[0] += cv[0];
		cv_btch[1] += cv[1];
		cv_btch[2] += cv[2];
	}

	cv_btch[0] /= nbtch;
	cv_btch[1] /= nbtch;
	cv_btch[2] /= nbtch;
}

/* GPR WITH MEAN */

double get_gpr_mean_cv_holdout_batch(unsigned long k, unsigned long ntst, const double *x,
				     const double *y, const double *y_mn, unsigned long n,
				     unsigned int dim, double *hp, unsigned long nhp,
				     enum estimator est)
{
	double *ytrn, *xtrn, *ytst, *ytst_gpr, *xtst, *ytst_mn, *ytrn_mn, diff, *covar, cv_holdout;
	unsigned long ntrn, i;

	ntrn = n - ntst;

	ytrn = malloc(ntrn * sizeof(double));
	assert(ytrn);
	ytst = malloc(ntst * sizeof(double));
	assert(ytst);
	ytst_gpr = malloc(ntst * sizeof(double));
	assert(ytst_gpr);

	ytrn_mn = malloc(ntrn * sizeof(double));
	assert(ytrn_mn);
	ytst_mn = malloc(ntst * sizeof(double));
	assert(ytst_mn);

	xtrn = malloc(ntrn * dim * sizeof(double));
	assert(xtrn);
	xtst = malloc(ntst * dim * sizeof(double));
	assert(xtst);

	covar = malloc(ntst * ntst * sizeof(double));
	assert(covar);

	get_subsample_cv_holdout(ytst_mn, xtst, ntst, ytrn_mn, xtrn, ntrn, y_mn, x, n, dim, k);

	get_subsample_cv_holdout(ytst, xtst, ntst, ytrn, xtrn, ntrn, y, x, n, dim, k);

	cv_holdout = -1;

	gpr_interpolate_mean(xtst, ytst_gpr, ytst_mn, ntst, xtrn, ytrn, ytrn_mn, ntrn, dim, hp, nhp,
			     covar, 0);
	if (est == CHI_SQ) {
		cv_holdout = get_chi_sq(ytst, ytst_gpr, covar, ntst);
	}

	if (est == MAHALANOBIS) {
		cv_holdout = get_mhlbs_dist(ytst, ytst_gpr, covar, ntst);
	}

	free(xtrn);
	free(xtst);
	free(ytrn);
	free(ytst);
	free(ytrn_mn);
	free(ytst_mn);
	free(ytst_gpr);
	free(covar);

	return cv_holdout;
}

double get_gpr_mean_cv_holdout(const double *x, const double *y, const double *y_mn,
			       unsigned long n, unsigned int dim, double *hp, unsigned long nhp,
			       unsigned long ntst, unsigned long nbtch, enum estimator est)
{
	double cv_btch_avg;
	unsigned long k;

	assert(n % ntst == 0);

	cv_btch_avg = 0;
	for (k = 0; k < nbtch; k++) {

		cv_btch_avg
		    += get_gpr_mean_cv_holdout_batch(k, ntst, x, y, y_mn, n, dim, hp, nhp, est);
	}

	return cv_btch_avg;
}

void test_get_subsample_cv_holdout(unsigned long n, unsigned long ntst, unsigned long k,
				   unsigned int dim, int seed)
{
	double *ytst, *ytrn, *xtst, *xtrn, *y, *x, tmp;
	unsigned long ntrn, i, d;
	dsfmt_t drng;

	printf("test_get_subsample_cv_holdout(n = %lu, ntst = %lu, k = %lu, dim = %u):\n", n, ntst,
	       k, dim);

	ntrn = n - ntst;

	x = malloc(n * dim * sizeof(double));
	assert(x);
	y = malloc(n * sizeof(double));
	assert(y);

	xtst = malloc(ntst * dim * sizeof(double));
	assert(xtst);
	ytst = malloc(ntst * sizeof(double));
	assert(ytst);

	xtrn = malloc(ntrn * dim * sizeof(double));
	assert(xtrn);
	ytrn = malloc(ntrn * sizeof(double));
	assert(ytrn);

	dsfmt_init_gen_rand(&drng, seed);

	for (i = 0; i < n; i++) {

		tmp = 0;
		for (d = 0; d < dim; d++) {
			x[i * dim + d] = dsfmt_genrand_close_open(&drng);
			tmp += x[i * dim + d];
		}

		y[i] = tmp;
	}

	get_subsample_cv_holdout(ytst, xtst, ntst, ytrn, xtrn, ntrn, y, x, n, dim, k);

	printf("## x \t y \n\n");

	for (i = 0; i < n; i++) {
		printf("%3ld ", i + 1);
		for (d = 0; d < dim; d++) {
			printf("%+.15E ", x[i * dim + d]);
		}
		printf("%+.15E\n", y[i]);
	}

	printf("\n\n## xtst \t ytst \n\n");

	for (i = 0; i < ntst; i++) {
		printf("%3ld ", i + 1);
		for (d = 0; d < dim; d++) {
			printf("%+.15E ", xtst[i * dim + d]);
		}
		printf("%+.15E\n", ytst[i]);
	}

	printf("\n\n## xtrn \t ytrn \n\n");

	for (i = 0; i < ntrn; i++) {
		printf("%3ld ", i + 1);
		for (d = 0; d < dim; d++) {
			printf("%+.15E ", xtrn[i * dim + d]);
		}
		printf("%+.15E\n", ytrn[i]);
	}

	free(x);
	free(y);
	free(xtst);
	free(ytst);
	free(xtrn);
	free(ytrn);
}

void test_get_gpr_cv_holdout(unsigned long n, unsigned int dim, unsigned long ntst,
			     unsigned long nbtch, enum estimator est, int seed)
{
	double cv[3], *y, *x, *st, *en, *hpar, *covar;
	unsigned long nhpar, i;
	dsfmt_t drng;

	printf("test_get_gpr_cv_holdout(n = %lu, ntst = %lu, nbtch = %lu, dim = %u):\n", n, ntst,
	       nbtch, dim);

	y = malloc(n * sizeof(double));
	assert(y);
	x = malloc(n * dim * sizeof(double));
	assert(x);

	st = malloc(dim * sizeof(double));
	assert(st);
	en = malloc(dim * sizeof(double));
	assert(en);

	nhpar = dim + 1;

	hpar = malloc(nhpar * sizeof(double));
	assert(hpar);

	for (i = 0; i < dim; i++) {
		st[i] = 0.01;
		en[i] = 3.0;
	}

	fill_random(x, n, dim, st, en, seed);

	for (i = 0; i < n; i++) {
		y[i] = sin_nd(&x[dim * i], dim);
	}

	for (i = 0; i < nhpar; i++) {
		hpar[i] = 1.0;
	}

	get_hyper_param_ard(hpar, nhpar, x, y, n, dim, get_krn_se_ard, get_dkrn_se_ard, NULL);

	get_gpr_cv_holdout(cv, x, y, n, dim, hpar, nhpar, ntst, nbtch, est);

	free(hpar);
	free(en);
	free(st);
	free(x);
	free(y);

	printf("%+.15E REL_RMSE\n", cv[0]);
	printf("%+.15E CHI_SQ\n", cv[1]);
	printf("%+.15E MAHALANABIS\n", cv[2]);
}
