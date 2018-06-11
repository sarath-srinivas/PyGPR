#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "lib_gpr.h"
#include "../lib_levmar/levmar.h"

#define PI (3.14159265358979)

int get_krn_se(double *krn, const double *x, const double *xp, unsigned long nx, unsigned long nxp,
	       unsigned long dim, const double *p, int npar)
{
	double sig_y, l, l2, r2, x_xp;
	unsigned long i, j, k;

	assert(npar == 2);

	sig_y = p[0];
	l = p[1];
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

	return 0;
}

int get_gpr_weights(double *wt, double *krn_chd, const double *krn, unsigned long ns, unsigned long dim,
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

	return info;
}

int gpr_predict(double *yp, const double *wt, const double *krnp, unsigned long np, const unsigned long ns)
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

	return 0;
}

int get_var_mat(double *var, double *krnpp, double *krnp, double *krn, unsigned long np, unsigned long ns)
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

	return info;
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

int get_var_mat_chd(double *var, double *krnpp, double *krnp, double *krn_chd, unsigned long np,
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

	return 0;
}

void cost_fun(double *p, double *f, int np, int n, void *data)
{
	struct gpr_dat *gp;
	unsigned long ns;
	int i, info;
	double *wt, *krn, *krn_chd;

	gp = (struct gpr_dat *)data;
	ns = gp->ns;

	wt = malloc(ns * sizeof(double));
	assert(wt);

	krn = malloc(ns * ns * sizeof(double));
	assert(krn);

	krn_chd = malloc(ns * ns * sizeof(double));
	assert(krn_chd);

	get_krn_se(krn, gp->x, gp->x, ns, ns, gp->dim, p, np);

	info = get_gpr_weights(wt, krn_chd, krn, ns, gp->dim, gp->y);
	assert(info == 0);

	for (i = 0; i < n; ++i) {
		f[i] = -1.0 * get_log_likelihood(wt, gp->y, ns, krn_chd, NULL);
	}

	free(wt);
	free(krn);
	free(krn_chd);
}

void jac_cost_fun(double *p, double *jac, int np, int n, void *data)
{
	unsigned char tra, uplo;
	struct gpr_dat *gp;
	int i, j, ns, N, M, LDA, LDB, incx, incy, info;
	double *B, *kl, alph, bet, tr_wt[2], tr_krn[2], sig_f, l, l3, jac_sig_f, jac_l;
	double *wt, *krn, *krn_chd;

	sig_f = p[0];
	l = p[1];
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

	info = get_gpr_weights(wt, krn_chd, krn, ns, gp->dim, gp->y);
	assert(info == 0);

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

	dgemv_(&tra, &M, &N, &alph, krn, &LDA, wt, &incx, &bet, B, &incy);

	/* TR(WT*WT^T*K) */

	tr_wt[0] = 0;
	for (i = 0; i < N; ++i) {
		tr_wt[0] += wt[i] * B[i];
	}

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

	jac_sig_f = (1 / sig_f) * (tr_wt[0] - N);
	jac_l = (0.5 / l3) * (tr_wt[1] - tr_krn[1]);

	j = 0;
	for (i = 0; i < n; ++i) {
		jac[j++] = 0; /* jac_sig_f;*/
		jac[j++] = -jac_l;
	}

	free(B);
	free(kl);
	free(wt);
	free(krn);
	free(krn_chd);
}

int get_hyper_param(double *p, int np, double *x, double *y, unsigned long ns, int dim)
{
	struct gpr_dat *gp;
	double *r2, r2ij, x_xp, *f, ret, opts[LM_OPTS_SZ], info[LM_INFO_SZ];
	unsigned long i, j;
	int k, n;

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

	opts[0] = LM_INIT_MU;
	opts[1] = 1E-15;
	opts[2] = 1E-15;
	opts[3] = 1E-20;

	n = np;
	f = calloc(n, sizeof(double));
	assert(f);

	ret = dlevmar_der(cost_fun, jac_cost_fun, p, f, np, n, 1000, opts, info, NULL, NULL, gp);

	fprintf(stderr, "Levenberg-Marquardt returned %d in %g iter, reason %g\nSolution: ", ret, info[5],
		info[6]);
	fprintf(stderr, "\n\nMinimization info:\n");
	for (i = 0; i < LM_INFO_SZ; ++i) {
		fprintf(stderr, "%g ", info[i]);
	}

	printf("\n");

	free(gp);
	free(r2);
	free(f);

	return 0;
}
