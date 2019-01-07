#include <assert.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "lib_gpr.h"

#define PI (3.14159265358979)

void get_krn_se(double *krn, const double *x, const double *xp, unsigned long nx, unsigned long nxp,
		unsigned long dim, const double *p, int npar)
{
	double sig_y, l, l2, r2, x_xp;
	unsigned long i, j, k;

	assert(npar == 1);

	sig_y = 1.0;
	l = p[0];
	l2 = l * l;

	for (i = 0; i < nx; i++)
		for (j = 0; j < nxp; j++) {

			r2 = 0;
			for (k = 0; k < dim; k++) {
				x_xp = x[dim * i + k] - xp[dim * j + k];
				r2 += x_xp * x_xp;
			}

			krn[i * nxp + j] = sig_y * sig_y * exp(-0.5 * r2 / l2);
		}
}

void get_krn_se_ard(double *krn, const double *x, const double *xp, unsigned long nx, unsigned long nxp,
		    unsigned long dim, const double *p, int npar)
{
	double sig_y, l, l2, r2, x_xp;
	unsigned long i, j, k;

	assert(npar == dim);

	sig_y = 1;

	for (i = 0; i < nx; i++)
		for (j = 0; j < nxp; j++) {

			r2 = 0;
			for (k = 0; k < dim; k++) {
				x_xp = x[dim * i + k] - xp[dim * j + k];
				r2 += x_xp * x_xp / (p[k] * p[k]);
			}

			krn[i * nxp + j] = sig_y * sig_y * exp(-0.5 * r2);
		}
}

void get_gpr_weights(double *wt, double *krn_chd, const double *krn, unsigned long ns, unsigned long dim,
		     const double *y)
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
	assert(info == 0);
}

void gpr_predict(double *yp, const double *wt, const double *krnp, unsigned long np, const unsigned long ns)
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

void get_var_mat(double *var, double *krnpp, double *krnp, double *krn, unsigned long np, unsigned long ns)
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

double get_log_likelihood(const double *wt, const double *y, unsigned long ns, const double *krn_chd,
			  double *ret)
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
		log_det_k += log(krn_chd[i * ns + i]);
	}

	llhd = -0.5 * ywt - log_det_k - 0.5 * ns * log(2 * PI);

	if (ret) {
		ret[0] = -0.5 * ywt;
		ret[1] = -log_det_k;
		ret[2] = -0.5 * ns * log(2 * PI);
	}

	/*
	    printf("%+.15E %+.15E %+.15E %+.15E\n", -0.5 * ywt, -log_det_k, -1.0 * ns * log(2 * PI), llhd);
	    */

	return llhd;
}

void get_var_mat_chd(double *var, double *krnpp, double *krnp, double *krn_chd, unsigned long np,
		     unsigned long ns)
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

static double cost_fun(const gsl_vector *pv, void *data)
{
	struct gpr_dat *gp;
	unsigned long ns, np;
	int i, info;
	double *wt, *krn, *krn_chd, f, *p;

	p = pv->data;
	np = pv->size;
	assert(np == 1);

	gp = (struct gpr_dat *)data;
	ns = gp->ns;

	wt = malloc(ns * sizeof(double));
	assert(wt);

	krn = malloc(ns * ns * sizeof(double));
	assert(krn);

	krn_chd = malloc(ns * ns * sizeof(double));
	assert(krn_chd);

	get_krn_se(krn, gp->x, gp->x, ns, ns, gp->dim, p, np);

	get_gpr_weights(wt, krn_chd, krn, ns, gp->dim, gp->y);

	f = -1.0 * get_log_likelihood(wt, gp->y, ns, krn_chd, NULL);

	free(wt);
	free(krn);
	free(krn_chd);

	return f;
}

static void jac_cost_fun(const gsl_vector *pv, void *data, gsl_vector *jac)
{
	unsigned char tra, uplo;
	unsigned long np;
	struct gpr_dat *gp;
	int i, j, ns, N, M, LDA, LDB, incx, incy, info;
	double *B, *kl, alph, bet, tr_wt[2], tr_krn[2], sig_f, l, l3, jac_sig_f, jac_l;
	double *wt, *krn, *krn_chd, *p;

	p = pv->data;
	np = pv->size;
	assert(np == 1);

	sig_f = 1.0;
	l = p[0];
	l3 = l * l * l;

	gp = (struct gpr_dat *)data;
	ns = gp->ns;

	wt = malloc(ns * sizeof(double));
	assert(wt);

	krn = malloc(ns * ns * sizeof(double));
	assert(krn);

	krn_chd = malloc(ns * ns * sizeof(double));
	assert(krn_chd);

	get_krn_se(krn, gp->x, gp->x, ns, ns, gp->dim, p, np);

	get_gpr_weights(wt, krn_chd, krn, ns, gp->dim, gp->y);

	/* GETTING B^T = WT^T * K OR B = K^T * WT  */

	N = gp->ns;
	M = gp->ns;
	LDA = gp->ns;
	alph = 1.0;
	bet = 0;
	tra = 'N';
	incx = 1;
	incy = 1;

	B = malloc(N * sizeof(double));
	assert(B);

	/* KL(i,j) = R(i,j)^2 / l^3 */

	kl = malloc(N * N * sizeof(double));
	assert(kl);

	for (i = 0; i < N * N; ++i) {
		kl[i] = (gp->r2)[i] * krn[i];
	}

	dgemv_(&tra, &M, &N, &alph, kl, &LDA, wt, &incx, &bet, B, &incy);

	/* TR(WT*WT^T*K) */
	tr_wt[1] = 0;
	for (i = 0; i < N; ++i) {
		tr_wt[1] += wt[i] * B[i];
	}

	/* K^-1 * KL = L\L^T\KL */

	uplo = 'L';
	LDA = gp->ns;
	LDB = gp->ns;

	dpotrs_(&uplo, &N, &M, krn_chd, &LDA, kl, &LDB, &info);
	assert(info == 0);

	/* TR(K^-1 * KL) */

	tr_krn[1] = 0;
	for (i = 0; i < N; ++i) {
		tr_krn[1] += kl[i * N + i];
	}

	jac_l = (0.5 / l3) * (tr_wt[1] - tr_krn[1]);

	gsl_vector_set(jac, 0, -1.0 * jac_l);

	/*
	    printf("%+.15E %+.15E\n", p[0], -jac_l);
	    */

	free(B);
	free(kl);
	free(wt);
	free(krn);
	free(krn_chd);
}

static void fdf_cost_fun(const gsl_vector *pv, void *data, double *f, gsl_vector *jac)
{

	unsigned char tra, uplo;
	unsigned long np;
	struct gpr_dat *gp;
	int i, j, ns, N, M, LDA, LDB, incx, incy, info;
	double *B, *kl, alph, bet, tr_wt[2], tr_krn[2], sig_f, l, l3, jac_sig_f, jac_l;
	double *wt, *krn, *krn_chd, *p;

	p = pv->data;
	np = pv->size;
	assert(np == 1);

	sig_f = 1.0;
	l = p[0];
	l3 = l * l * l;

	gp = (struct gpr_dat *)data;
	ns = gp->ns;

	wt = malloc(ns * sizeof(double));
	assert(wt);

	krn = malloc(ns * ns * sizeof(double));
	assert(krn);

	krn_chd = malloc(ns * ns * sizeof(double));
	assert(krn_chd);

	get_krn_se(krn, gp->x, gp->x, ns, ns, gp->dim, p, np);

	get_gpr_weights(wt, krn_chd, krn, ns, gp->dim, gp->y);

	*f = -1.0 * get_log_likelihood(wt, gp->y, ns, krn_chd, NULL);

	/* GETTING B^T = WT^T * K OR B = K^T * WT  */

	N = gp->ns;
	M = gp->ns;
	LDA = gp->ns;
	alph = 1.0;
	bet = 0;
	tra = 'N';
	incx = 1;
	incy = 1;

	B = malloc(N * sizeof(double));
	assert(B);

	/* KL(i,j) = R(i,j)^2 / l^3 */

	kl = malloc(N * N * sizeof(double));
	assert(kl);

	for (i = 0; i < N * N; ++i) {
		kl[i] = (gp->r2)[i] * krn[i];
	}

	dgemv_(&tra, &M, &N, &alph, kl, &LDA, wt, &incx, &bet, B, &incy);

	/* TR(WT*WT^T*K) */
	tr_wt[1] = 0;
	for (i = 0; i < N; ++i) {
		tr_wt[1] += wt[i] * B[i];
	}

	/* K^-1 * KL = L\L^T\KL */

	uplo = 'L';
	LDA = gp->ns;
	LDB = gp->ns;

	dpotrs_(&uplo, &N, &M, krn_chd, &LDA, kl, &LDB, &info);
	assert(info == 0);

	/* TR(K^-1 * KL) */

	tr_krn[1] = 0;
	for (i = 0; i < N; ++i) {
		tr_krn[1] += kl[i * N + i];
	}

	jac_l = (0.5 / l3) * (tr_wt[1] - tr_krn[1]);

	gsl_vector_set(jac, 0, -1.0 * jac_l);

	/*
	    printf("%+.15E %+.15E\n", p[0], -jac_l);
	    */

	free(B);
	free(kl);
	free(wt);
	free(krn);
	free(krn_chd);
}

static double cost_fun_ard(const gsl_vector *pv, void *data)
{
	struct gpr_dat *gp;
	unsigned long ns, np;
	int i, dim, info;
	double *wt, *krn, *krn_chd, f, *p;

	gp = (struct gpr_dat *)data;
	ns = gp->ns;
	dim = gp->dim;

	p = pv->data;
	np = pv->size;
	assert(np == dim);

	wt = malloc(ns * sizeof(double));
	assert(wt);

	krn = malloc(ns * ns * sizeof(double));
	assert(krn);

	krn_chd = malloc(ns * ns * sizeof(double));
	assert(krn_chd);

	get_krn_se_ard(krn, gp->x, gp->x, ns, ns, gp->dim, p, np);

	get_gpr_weights(wt, krn_chd, krn, ns, gp->dim, gp->y);

	f = -1.0 * get_log_likelihood(wt, gp->y, ns, krn_chd, NULL);

	free(wt);
	free(krn);
	free(krn_chd);

	return f;
}

static void jac_cost_fun_ard(const gsl_vector *pv, void *data, gsl_vector *jac)
{
	unsigned char tra, uplo;
	unsigned long np;
	struct gpr_dat *gp;
	int i, j, k, ns, N, M, LDA, LDB, incx, incy, info, dim;
	double *B, *kl, alph, bet, tr_wt, tr_krn, sig_f, l, ld3, jac_l, xij_d;
	double *wt, *krn, *krn_chd, *p;

	gp = (struct gpr_dat *)data;
	ns = gp->ns;
	dim = gp->dim;

	p = pv->data;
	np = pv->size;
	dim = gp->dim;
	assert(np == dim);

	sig_f = 1.0;

	wt = malloc(ns * sizeof(double));
	assert(wt);

	krn = malloc(ns * ns * sizeof(double));
	assert(krn);

	krn_chd = malloc(ns * ns * sizeof(double));
	assert(krn_chd);

	get_krn_se_ard(krn, gp->x, gp->x, ns, ns, gp->dim, p, np);

	get_gpr_weights(wt, krn_chd, krn, ns, gp->dim, gp->y);

	N = gp->ns;
	M = gp->ns;
	LDA = gp->ns;
	alph = 1.0;
	bet = 0;
	tra = 'N';
	incx = 1;
	incy = 1;
	uplo = 'L';
	LDA = gp->ns;
	LDB = gp->ns;

	B = malloc(N * sizeof(double));
	assert(B);

	kl = malloc(N * N * sizeof(double));
	assert(kl);

	for (k = 0; k < np; k++) {

		ld3 = p[k] * p[k] * p[k];

		for (i = 0; i < N; ++i) {
			for (j = 0; j < N; ++j) {
				xij_d = (gp->x)[i * dim + k] - (gp->x)[j * dim + k];
				kl[i * N + j] = (xij_d * xij_d / ld3) * krn[i * N + j];
			}
		}

		dgemv_(&tra, &M, &N, &alph, kl, &LDA, wt, &incx, &bet, B, &incy);

		tr_wt = 0;
		for (i = 0; i < N; ++i) {
			tr_wt += wt[i] * B[i];
		}

		dpotrs_(&uplo, &N, &M, krn_chd, &LDA, kl, &LDB, &info);
		assert(info == 0);

		tr_krn = 0;
		for (i = 0; i < N; ++i) {
			tr_krn += kl[i * N + i];
		}

		jac_l = 0.5 * (tr_wt - tr_krn);

		gsl_vector_set(jac, k, -1.0 * jac_l);
	}

	free(B);
	free(kl);
	free(wt);
	free(krn);
	free(krn_chd);
}

static void fdf_cost_fun_ard(const gsl_vector *pv, void *data, double *f, gsl_vector *jac)
{
	unsigned char tra, uplo;
	unsigned long np;
	struct gpr_dat *gp;
	int i, j, k, ns, N, M, LDA, LDB, incx, incy, info, dim;
	double *B, *kl, alph, bet, tr_wt, tr_krn, sig_f, l, ld3, xij_d, jac_l;
	double *wt, *krn, *krn_chd, *p;

	gp = (struct gpr_dat *)data;
	ns = gp->ns;
	dim = gp->dim;

	p = pv->data;
	np = pv->size;
	dim = gp->dim;
	assert(np == dim);

	sig_f = 1.0;

	wt = malloc(ns * sizeof(double));
	assert(wt);

	krn = malloc(ns * ns * sizeof(double));
	assert(krn);

	krn_chd = malloc(ns * ns * sizeof(double));
	assert(krn_chd);

	get_krn_se_ard(krn, gp->x, gp->x, ns, ns, gp->dim, p, np);

	get_gpr_weights(wt, krn_chd, krn, ns, gp->dim, gp->y);

	*f = -1.0 * get_log_likelihood(wt, gp->y, ns, krn_chd, NULL);

	N = gp->ns;
	M = gp->ns;
	LDA = gp->ns;
	alph = 1.0;
	bet = 0;
	tra = 'N';
	incx = 1;
	incy = 1;
	uplo = 'L';
	LDA = gp->ns;
	LDB = gp->ns;

	B = malloc(N * sizeof(double));
	assert(B);

	kl = malloc(N * N * sizeof(double));
	assert(kl);

	for (k = 0; k < np; k++) {

		ld3 = p[k] * p[k] * p[k];

		for (i = 0; i < N; ++i) {
			for (j = 0; j < N; ++j) {
				xij_d = (gp->x)[i * dim + k] - (gp->x)[j * dim + k];
				kl[i * N + j] = (xij_d * xij_d / ld3) * krn[i * N + j];
			}
		}

		dgemv_(&tra, &M, &N, &alph, kl, &LDA, wt, &incx, &bet, B, &incy);

		tr_wt = 0;
		for (i = 0; i < N; ++i) {
			tr_wt += wt[i] * B[i];
		}

		dpotrs_(&uplo, &N, &M, krn_chd, &LDA, kl, &LDB, &info);
		assert(info == 0);

		tr_krn = 0;
		for (i = 0; i < N; ++i) {
			tr_krn += kl[i * N + i];
		}

		jac_l = 0.5 * (tr_wt - tr_krn);

		gsl_vector_set(jac, k, -1.0 * jac_l);
	}

	free(B);
	free(kl);
	free(wt);
	free(krn);
	free(krn_chd);
}

void get_hyper_param(double *p, int np, double *x, double *y, unsigned long ns, int dim)
{
	struct gpr_dat *gp;
	double *r2, r2ij, x_xp, *f, ret;
	unsigned long i, j, iter, max_iter;
	int k, status;
	double step, tol;

	const gsl_multimin_fdfminimizer_type *T;
	gsl_multimin_fdfminimizer *s;
	gsl_vector *pv;
	gsl_multimin_function_fdf fun;

	gp = malloc(1 * sizeof(struct gpr_dat));
	assert(gp);

	/* R2(i,j) = || x_i - xj || */

	r2 = malloc(ns * ns * sizeof(double));
	assert(r2);

	for (i = 0; i < ns; ++i) {
		for (j = 0; j < ns; ++j) {
			r2ij = 0;
			for (k = 0; k < dim; k++) {
				x_xp = x[dim * i + k] - x[dim * j + k];
				r2ij += x_xp * x_xp;
			}
			r2[i * ns + j] = r2ij;
		}
	}

	gp->ns = ns;
	gp->dim = dim;
	gp->x = x;
	gp->y = y;
	gp->r2 = r2;

	max_iter = 1000;

	fun.n = np;
	fun.f = cost_fun;
	fun.df = jac_cost_fun;
	fun.fdf = fdf_cost_fun;
	fun.params = gp;

	pv = gsl_vector_alloc(np);
	assert(pv);

	for (i = 0; i < np; i++) {
		gsl_vector_set(pv, i, p[i]);
	}

	T = gsl_multimin_fdfminimizer_conjugate_fr;
	s = gsl_multimin_fdfminimizer_alloc(T, np);

	step = 1E-3;
	tol = 1E-6;

	gsl_multimin_fdfminimizer_set(s, &fun, pv, step, tol);

	iter = 0;
	do {
		iter++;
		status = gsl_multimin_fdfminimizer_iterate(s);

		if (status)
			break;

		status = gsl_multimin_test_gradient(s->gradient, tol);

		for (i = 0; i < np; i++) {
			fprintf(stderr, "%5ld P[%ld] %+.15E DF[%ld] %+.15E\n", iter, i,
				gsl_vector_get(s->x, i), i, gsl_vector_get(s->gradient, i));
		}

	} while (status == GSL_CONTINUE && iter < max_iter);

	for (i = 0; i < np; i++) {
		p[i] = gsl_vector_get(s->x, i);
	}

	gsl_multimin_fdfminimizer_free(s);
	gsl_vector_free(pv);

	free(gp);
	free(r2);
}

void get_hyper_param_ard(double *p, int np, double *x, double *y, unsigned long ns, int dim)
{
	struct gpr_dat *gp;
	double *f, ret;
	unsigned long i, j, iter, max_iter;
	int k, status;
	double step, tol;

	const gsl_multimin_fdfminimizer_type *T;
	gsl_multimin_fdfminimizer *s;
	gsl_vector *pv;
	gsl_multimin_function_fdf fun;

	gp = malloc(1 * sizeof(struct gpr_dat));
	assert(gp);

	gp->ns = ns;
	gp->dim = dim;
	gp->x = x;
	gp->y = y;
	gp->r2 = NULL;

	max_iter = 10000;

	fun.n = np;
	fun.f = cost_fun_ard;
	fun.df = jac_cost_fun_ard;
	fun.fdf = fdf_cost_fun_ard;
	fun.params = gp;

	pv = gsl_vector_alloc(np);
	assert(pv);

	for (i = 0; i < np; i++) {
		gsl_vector_set(pv, i, p[i]);
	}

	T = gsl_multimin_fdfminimizer_conjugate_fr;
	s = gsl_multimin_fdfminimizer_alloc(T, np);

	step = 1E-2;
	tol = 1E-3;

	gsl_multimin_fdfminimizer_set(s, &fun, pv, step, tol);

	iter = 0;
	do {
		iter++;
		status = gsl_multimin_fdfminimizer_iterate(s);

		if (status)
			break;

		status = gsl_multimin_test_gradient(s->gradient, tol);

		for (i = 0; i < np; i++) {
			fprintf(stderr, "%5ld P[%ld] %+.15E DF[%ld] %+.15E\n", iter, i,
				gsl_vector_get(s->x, i), i, gsl_vector_get(s->gradient, i));
		}

	} while (status == GSL_CONTINUE && iter < max_iter);

	for (i = 0; i < np; i++) {
		p[i] = gsl_vector_get(s->x, i);
	}

	gsl_multimin_fdfminimizer_free(s);
	gsl_vector_free(pv);

	free(gp);
}

void gpr_interpolate(double *y, double *x, unsigned long ns, unsigned int dim, double *xp, unsigned long np,
		     double *yp, double *p, unsigned int npar, double *var_yp)
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

	get_hyper_param_ard(p, npar, x, y, 200, dim);

	get_krn_se_ard(krxx, x, x, ns, ns, dim, p, npar);

	get_gpr_weights(wt, lkrxx, krxx, ns, dim, y);

	get_krn_se_ard(krpx, xp, x, np, ns, dim, p, npar);

	gpr_predict(yp, wt, krpx, np, ns);

	get_krn_se_ard(krpp, xp, xp, np, np, dim, p, npar);

	get_var_mat_chd(var_yp, krpp, krpx, lkrxx, np, ns);

	free(wt);
	free(krpx);
	free(lkrxx);
	free(krxx);
	free(krpp);
}
