#include <assert.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <atlas/lapack.h>
#include <atlas/blas.h>
#include "lib_gpr.h"

#define PI (3.14159265358979)

void get_gpr_weights(double *wt, double *krn_chd, const double *krn, unsigned long ns,
		     unsigned long dim, const double *y)
{
	double eps;
	int N, NRHS, LDA, LDB, info;
	unsigned long i;
	unsigned char UPLO;

	for (i = 0; i < ns * ns; i++) {
		krn_chd[i] = krn[i];
	}

	eps = 1E-7;

	for (i = 0; i < ns; i++) {
		krn_chd[i * ns + i] += eps;
	}

	for (i = 0; i < ns; i++) {
		wt[i] = y[i];
	}

	UPLO = 'L';
	N = (int)ns;
	LDA = (int)ns;
	LDB = (int)ns;
	NRHS = 1;
	dposv_(&UPLO, &N, &NRHS, krn_chd, &LDA, wt, &LDB, &info);
	if (info != 0)
		fprintf(stderr, "info: %d\n", info);
	assert(info == 0);
}

void gpr_predict(double *yp, const double *wt, const double *krnp, unsigned long np,
		 const unsigned long ns)
{
	unsigned char tr;
	int N, M, LDA, incx, incy;
	double alph, bet;

	tr = 'T';
	M = (int)ns;
	N = (int)np;
	LDA = M;
	incx = 1;
	incy = 1;
	alph = 1.0;
	bet = 0;

	dgemv_(&tr, &M, &N, &alph, krnp, &LDA, wt, &incx, &bet, yp, &incy);
}

void get_var_mat(double *var, double *krnpp, double *krnp, double *krn, unsigned long np,
		 unsigned long ns)
{
	unsigned char tra, trb, UPLO;
	int N, M, NRHS, LDA, LDB, LDC, info;
	unsigned long i, j, k;
	double *V, alph, bet, tmp, eps;

	V = malloc(np * ns * sizeof(double));
	assert(V);

	eps = 1E-7;

	for (i = 0; i < ns; i++) {
		krn[i * ns + i] += eps;
	}

	for (i = 0; i < ns * np; i++) {
		V[i] = krnp[i];
	}

	UPLO = 'L';
	N = ns;
	NRHS = np;
	LDA = N;
	LDB = N;

	dposv_(&UPLO, &N, &NRHS, krn, &LDA, V, &LDB, &info);
	assert(info == 0);

	for (i = 0; i < np * np; i++) {
		var[i] = krnpp[i];
	}

	N = ns;
	M = np;
	LDA = N;
	LDB = N;
	LDC = M;
	alph = -1.0;
	bet = 1.0;
	tra = 'T';
	trb = 'N';

	dgemm_(&tra, &trb, &M, &M, &N, &alph, krnp, &LDA, V, &LDB, &bet, var, &LDC);

	free(V);

	assert(info == 0);
}

void get_var_mat_chd(double *var, const double *krnpp, const double *krnp, const double *krn_chd,
		     unsigned long np, unsigned long ns)
{
	unsigned char tra, trb, UPLO, SIDE, DIAG;
	int N, M, NRHS, LDA, LDB, LDC, info;
	unsigned long i, j, k;
	double *V, alph, bet, tmp, eps;

	V = malloc(np * ns * sizeof(double));
	assert(V);

	for (i = 0; i < ns * np; i++) {
		V[i] = krnp[i];
	}

	SIDE = 'L';
	UPLO = 'L';
	tra = 'N';
	DIAG = 'N';
	N = ns;
	NRHS = np;
	LDA = N;
	LDB = N;
	alph = 1.0;

	dtrsm_(&SIDE, &UPLO, &tra, &DIAG, &N, &NRHS, &alph, krn_chd, &LDA, V, &LDB);

	for (i = 0; i < np * np; i++) {
		var[i] = krnpp[i];
	}

	N = ns;
	M = np;
	LDA = N;
	LDB = N;
	LDC = M;
	alph = -1.0;
	bet = 1.0;
	tra = 'T';
	UPLO = 'L';

	dsyrk_(&UPLO, &tra, &M, &N, &alph, V, &N, &bet, var, &LDC);

	free(V);
}

double get_log_likelihood(const double *wt, const double *y, unsigned long ns,
			  const double *krn_chd, double *ret)
{
	double llhd, ywt, log_det_k;
	int N, incx, incy;
	unsigned long i;

	N = ns;
	incx = 1;
	incy = 1;

	ywt = ddot_(&N, y, &incx, wt, &incy);

	log_det_k = 0;
	for (i = 0; i < ns; i++) {
		log_det_k += 2 * log(krn_chd[i * ns + i]);
	}

	llhd = -0.5 * ywt - 0.5 * log_det_k - 0.5 * ns * log(2 * PI);

	if (ret) {
		ret[0] = -0.5 * ywt;
		ret[1] = -log_det_k;
		ret[2] = -0.5 * ns * log(2 * PI);
	}

	return llhd;
}

void gpr_interpolate(double *xp, double *yp, unsigned long np, double *x, double *y,
		     unsigned long ns, unsigned int dim, double *p, unsigned int npar,
		     double *var_yp, int is_opt)
{
	double *krxx, *lkrxx, *krpx, *krpp, *wt;
	int info;

	krxx = malloc(ns * ns * sizeof(double));
	assert(krxx);

	lkrxx = malloc(ns * ns * sizeof(double));
	assert(lkrxx);

	krpx = malloc(np * ns * sizeof(double));
	assert(krpx);

	krpp = malloc(np * np * sizeof(double));
	assert(krpp);

	wt = malloc(ns * sizeof(double));
	assert(wt);

	if (is_opt) {
		get_hyper_param_ard(p, npar, x, y, ns, dim);
	}

	get_krn_se_ard(krxx, x, x, ns, ns, dim, p, npar);

	get_gpr_weights(wt, lkrxx, krxx, ns, dim, y);

	get_krn_se_ard(krpx, xp, x, np, ns, dim, p, npar);

	gpr_predict(yp, wt, krpx, np, ns);

	if (var_yp) {

		get_krn_se_ard(krpp, xp, xp, np, np, dim, p, npar);

		get_var_mat_chd(var_yp, krpp, krpx, lkrxx, np, ns);
	}

	free(wt);
	free(krpx);
	free(lkrxx);
	free(krxx);
	free(krpp);
}
