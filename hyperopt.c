#include <assert.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <blas/lapack.h>
#include <blas/blas.h>
#include <lib_rng/lib_rng.h>
#include "lib_gpr.h"

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
	assert(np == dim + 1);

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
	unsigned long np, N2;
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
	assert(np == dim + 1);

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
	N2 = N * N;

	B = malloc(N * sizeof(double));
	assert(B);

	kl = malloc(N * N * sizeof(double));
	assert(kl);

	for (k = 0; k < np; k++) {

		get_dkrn_se_ard(kl, k, gp->x, krn, N, dim, p, np);

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
	unsigned long np, N2;
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
	assert(np == dim + 1);

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
	N2 = N * N;

	B = malloc(N * sizeof(double));
	assert(B);

	kl = malloc(N * N * sizeof(double));
	assert(kl);

	for (k = 0; k < np; k++) {

		get_dkrn_se_ard(kl, k, gp->x, krn, N, dim, p, np);

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

	T = gsl_multimin_fdfminimizer_conjugate_pr;
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

void get_hyper_param_ard_stoch(double *p, int np, double *x, double *y, unsigned long ns, int dim,
			       unsigned long nsub, double lrate, int seed)
{
	double tol, *pt, *jact, *xsub, *ysub;
	struct gpr_dat *gp;
	unsigned long i, m, n, j, iter, max_iter, *a;

	gsl_vector *pv, *jac;

	pt = malloc(np * sizeof(double));
	assert(pt);
	jact = malloc(np * sizeof(double));
	assert(jact);

	xsub = malloc(nsub * dim * sizeof(double));
	assert(xsub);
	ysub = malloc(nsub * sizeof(double));
	assert(ysub);
	a = malloc(ns * sizeof(unsigned long));
	assert(a);

	pv = gsl_vector_alloc(np);
	assert(pv);
	jac = gsl_vector_alloc(np);
	assert(jac);

	gp = malloc(1 * sizeof(struct gpr_dat));
	assert(gp);

	for (i = 0; i < np; i++) {
		gsl_vector_set(pv, i, p[i]);
	}

	for (i = 0; i < ns; i++) {
		a[i] = i;
	}

	max_iter = 1000;

	for (m = 0; m < max_iter; m++) {

		fisher_yates_shuffle(a, ns, seed);

		for (i = 0; i < nsub; i++) {

			n = a[i];

			for (j = 0; j < nsub; j++) {
				xsub[i * dim + j] = x[n * dim + j];
			}

			ysub[i] = y[n];
		}

		gp->ns = nsub;
		gp->dim = dim;
		gp->x = xsub;
		gp->y = ysub;
		gp->r2 = NULL;

		jac_cost_fun_ard(pv, gp, jac);

		for (i = 0; i < np; i++) {
			pt[i] = gsl_vector_get(pv, i);
			jact[i] = gsl_vector_get(jac, i);
		}

		for (i = 0; i < np; i++) {
			printf("P[%ld] %+.15E J[%ld] %+.15E\n", i, pt[i], i, jact[i]);
		}
		printf("\n");

		for (i = 0; i < np; i++) {
			pt[i] = pt[i] - lrate * jact[i];
		}

		for (i = 0; i < np; i++) {
			gsl_vector_set(pv, i, pt[i]);
			gsl_vector_set(jac, i, jact[i]);
		}
	}

	free(gp);
	free(pt);
	free(jact);
	free(xsub);
	free(ysub);
	free(a);
	gsl_vector_free(pv);
	gsl_vector_free(jac);
}

/* TESTS */
double test_jac_cost_fun_ard(int m, unsigned int dim, unsigned long nx, double eps, int seed)
{
	double *x, *kxx, *lkxx, *wt, *y, xd, llhd, llhd_eps, dllhd, dllhd_num, *p, xmax, err_norm;
	struct gpr_dat *gp;
	unsigned int np;
	unsigned long i, j, nx2;
	gsl_vector *pv, *jac;
	dsfmt_t drng;

	np = dim + 1;
	assert(m <= np);

	nx2 = nx * nx;

	x = malloc(dim * nx * sizeof(double));
	assert(x);
	y = malloc(nx * sizeof(double));
	assert(y);
	wt = malloc(nx * sizeof(double));
	assert(wt);
	p = malloc(np * sizeof(double));
	assert(p);

	kxx = malloc(nx * nx * sizeof(double));
	assert(kxx);
	lkxx = malloc(nx * nx * sizeof(double));
	assert(lkxx);

	dsfmt_init_gen_rand(&drng, seed);

	xmax = 2.0;

	for (i = 0; i < nx; i++) {
		for (j = 0; j < dim; j++) {
			x[i * dim + j] = xmax * dsfmt_genrand_close_open(&drng);
		}
	}

	for (i = 0; i < nx; i++) {
		xd = 0;
		for (j = 0; j < dim; j++) {
			xd += x[i * dim + j] * x[i * dim + j];
		}
		y[i] = sin(xd);
	}

	dsfmt_init_gen_rand(&drng, seed + 34);

	for (i = 0; i < np; i++) {
		p[i] = 0.5 + dsfmt_genrand_close_open(&drng);
	}

	get_krn_se_ard(kxx, x, x, nx, nx, dim, p, np);
	get_gpr_weights(wt, lkxx, kxx, nx, dim, y);

	llhd = get_log_likelihood(wt, y, nx, lkxx, NULL);

	gp = malloc(1 * sizeof(struct gpr_dat));
	assert(gp);

	gp->ns = nx;
	gp->dim = dim;
	gp->x = x;
	gp->y = y;
	gp->r2 = NULL;

	pv = gsl_vector_alloc(np);
	assert(pv);
	jac = gsl_vector_alloc(np);
	assert(jac);

	for (i = 0; i < np; i++) {
		gsl_vector_set(pv, i, p[i]);
	}

	jac_cost_fun_ard(pv, gp, jac);

	dllhd = gsl_vector_get(jac, m);

	p[m] += eps;

	get_krn_se_ard(kxx, x, x, nx, nx, dim, p, np);
	get_gpr_weights(wt, lkxx, kxx, nx, dim, y);

	llhd_eps = get_log_likelihood(wt, y, nx, lkxx, NULL);

	dllhd_num = (llhd - llhd_eps) / eps;

	gsl_vector_free(pv);
	gsl_vector_free(jac);
	free(x);
	free(p);
	free(kxx);
	free(lkxx);
	free(y);
	free(wt);

	if (DEBUG) {
		printf("%+.15E %+.15E\n", dllhd_num, dllhd);
	}

	return fabs(dllhd_num - dllhd);
}
