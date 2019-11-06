#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_rng/lib_rng.h>
#include <atlas/blas.h>
#include <atlas/lapack.h>
#include "lib_gpr.h"

double get_rmse_rel(const double *y, const double *y_pred, unsigned long n)
{
	double rel_diff, rmse_rel;
	unsigned long i;

	rmse_rel = 0;
	for (i = 0; i < n; i++) {

		rel_diff = (y[i] - y_pred[i]) / fabs(y[i]);
		rmse_rel += rel_diff;
	}

	rmse_rel /= n;

	return rmse_rel;
}

double get_mhlbs_dist(const double *y, const double *y_pred, const double *covar, unsigned long n)
{
	double *diff, *kdiff, *covar_chd, mhlbs_dist;
	unsigned long i;
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

	N = n * n;
	INCX = 1;
	INCY = 1;
	dcopy_(&N, covar, &INCX, covar_chd, &INCY);

	UPLO = 'L';
	N = (int)n;
	LDA = (int)n;
	LDB = (int)n;
	NRHS = 1;
	dposv_(&UPLO, &N, &NRHS, covar_chd, &LDA, kdiff, &LDB, &info);
	assert(info == 0);

	mhlbs_dist = ddot_(&N, diff, &INCX, kdiff, &INCY);

	return mhlbs_dist;
}

void get_subsample_cv_holdout(double *ytst, double *xtst, unsigned long ntst, double *ytrn,
			      double *xtrn, unsigned long ntrn, const double *y, const double *x,
			      unsigned long n, unsigned int dim, unsigned long k)
{
	unsigned long nsub, i, j, l, m, d;

	assert(n % ntst == 0 && "Subsample does not divide sample!!");
	assert(ntrn == n - ntst);

	nsub = n / ntst;

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

double get_gpr_cv_holdout_batch(unsigned long k, unsigned long ntst, const double *x,
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

	cv_holdout = -1;

	if (est == REL_RMSE) {

		gpr_interpolate(xtst, ytst_gpr, ntst, xtrn, ytrn, ntrn, dim, hp, nhp, NULL, 0);

		cv_holdout = get_rmse_rel(ytst, ytst_gpr, ntst);
	}

	if (est == MAHALANOBIS) {

		gpr_interpolate(xtst, ytst_gpr, ntst, xtrn, ytrn, ntrn, dim, hp, nhp, covar, 0);

		cv_holdout = get_mhlbs_dist(ytst, ytst_gpr, covar, ntst);
	}

	free(xtrn);
	free(xtst);
	free(ytrn);
	free(ytst);
	free(ytst_gpr);
	free(covar);

	return cv_holdout;
}

double get_gpr_cv_holdout(const double *x, const double *y, unsigned long n, unsigned int dim,
			  double *hp, unsigned long nhp, unsigned long ntst, unsigned long nbtch,
			  enum estimator est)
{
	double cv_btch_avg;
	unsigned long k;

	assert(n % ntst == 0);

	cv_btch_avg = 0;
	for (k = 0; k < nbtch; k++) {

		cv_btch_avg += get_gpr_cv_holdout_batch(k, ntst, x, y, n, dim, hp, nhp, est);
	}

	cv_btch_avg /= nbtch;

	return cv_btch_avg;
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

	if (est == REL_RMSE) {

		gpr_interpolate_mean(xtst, ytst_gpr, ytst_mn, ntst, xtrn, ytrn, ytrn_mn, ntrn, dim,
				     hp, nhp, NULL, 0);

		cv_holdout = get_rmse_rel(ytst, ytst_gpr, ntst);
	}

	if (est == MAHALANOBIS) {

		gpr_interpolate_mean(xtst, ytst_gpr, ytst_mn, ntst, xtrn, ytrn, ytrn_mn, ntrn, dim,
				     hp, nhp, covar, 0);

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
