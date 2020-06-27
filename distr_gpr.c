#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <blas/lapack.h>
#include <blas/blas.h>
#include <lib_rng/lib_rng.h>
#include "lib_gpr.h"

static void get_tensor_product(double *AB, const double *A, const double *B, unsigned long n)
{

	int N, INCX, INCY, K, LDA;
	double BETA, ALPHA;
	unsigned char UPLO;

	N = n;
	K = 0;
	LDA = 1;
	INCX = 1;
	INCY = 1;
	UPLO = 'L';
	ALPHA = 1.0;
	BETA = 0.0;

	dsbmv_(&UPLO, &N, &K, &ALPHA, A, &LDA, B, &INCX, &BETA, AB, &INCY);
}

static void get_indicator(unsigned long *ind, const double *xp, unsigned long np, unsigned int dim,
			  const double *xc, unsigned long nc)
{
	unsigned long i;

	for (i = 0; i < np; i++) {
		ind[i] = get_near_idx(&xp[i * dim], dim, xc, nc);
	}
}

static void augment(double *x12, double *y12, const double *x1, const double *y1, unsigned long n1,
		    const double *x2, const double *y2, unsigned long n2, unsigned int dim)
{

	unsigned long i, k;

	for (i = 0; i < n1; i++) {
		for (k = 0; k < dim; k++) {
			x12[i * dim + k] = x1[i * dim + k];
		}

		y12[i] = y1[i];
	}

	for (i = 0; i < n2; i++) {
		for (k = 0; k < dim; k++) {
			x12[(i + n1) * dim + k] = x2[i * dim + k];
		}

		y12[i + n1] = y2[i];
	}
}

void gpr_interpolate_grbcm(
    double *yp, double *var_yp, const double *xp, unsigned long np, const double *xl,
    const double *y, unsigned long ns, const double *xg, const double *yg, unsigned long ng,
    unsigned long nc, unsigned int dim, double *hpl, unsigned long nhpl, double *hpg,
    unsigned long nhpg, int is_opt,
    void covar(double *krn, const double *x, const double *xp, unsigned long nx, unsigned long nxp,
	       unsigned int dim, const double *p, unsigned int npar, void *dat),
    void covar_jac(double *dK, unsigned int k, const double *x, const double *kxx, unsigned long nx,
		   unsigned int dim, const double *p, unsigned int np, void *dat),
    void *dat)

{

	double *ypl, *var_ypl, *ypg, *var_ypg, *xa, *ya;
	unsigned long i, nsc, nhp, *ind, na;

	ypg = malloc(np * sizeof(double));
	assert(ypg);
	var_ypg = malloc(np * np * sizeof(double));
	assert(var_ypg);

	ypl = malloc(np * nc * sizeof(double));
	assert(ypl);
	var_ypl = malloc(np * np * nc * sizeof(double));
	assert(var_ypl);

	nsc = ns / nc;
	nhp = nhpl / nc;
	na = ng + nsc;

	xa = malloc(na * dim * sizeof(double));
	assert(xa);
	ya = malloc(na * sizeof(double));
	assert(ya);

	gpr_interpolate_wrap(xp, ypg, np, xg, yg, ng, dim, hpg, nhpg, var_ypg, is_opt, covar,
			     covar_jac, dat);

	for (i = 0; i < nc; i++) {

		augment(xa, ya, xg, yg, ng, &xl[i * nsc * dim], &y[i * nsc], nsc, dim);

		gpr_interpolate_wrap(xp, &ypl[i * np], np, xa, ya, na, dim, &hpl[i * nhp], nhp,
				     &var_ypl[i * np * np], is_opt, covar, covar_jac, dat);
	}

	grbcm_experts(yp, var_yp, np, ypg, var_ypg, ypl, var_ypl, np * nc);

	free(ya);
	free(xa);
	free(var_ypl);
	free(ypl);
	free(var_ypg);
	free(ypg);
}

void grbcm_experts(double *yp, double *var_yp, unsigned long np, const double *ypg,
		   const double *var_ypg, const double *ypl, const double *var_ypl,
		   unsigned long nl)

{
	unsigned long i, k, nc, nhps;
	long Nl, Ml;
	int N, INCX, INCY;
	double *beta, *beta_sum, *prec_gl, *prec_g, tmp, *prec_sum, DA, *wt, *yp_sum, eps;

	nc = nl / np;

	prec_gl = malloc(np * nc * sizeof(double));
	assert(prec_gl);
	prec_g = malloc(np * sizeof(double));
	assert(prec_g);

	beta = malloc(np * nc * sizeof(double));
	assert(beta);
	wt = malloc(np * sizeof(double));
	assert(wt);

	yp_sum = calloc(np, sizeof(double));
	assert(yp_sum);
	prec_sum = calloc(np, sizeof(double));
	assert(prec_sum);
	beta_sum = calloc(np, sizeof(double));
	assert(beta_sum);

	eps = 1E-7;

	Nl = np;
	Ml = np * nc;
	N = (int)np;
	DA = 1.0;
	INCX = 1;
	INCY = 1;

	for (i = 0; i < np; i++) {

		prec_g[i] = 1 / (var_ypg[i * np + i] + eps);
		beta[i] = 1;
	}

	for (k = 0; k < nc; k++) {
		for (i = 0; i < np; i++) {

			prec_gl[k * np + i] = 1 / (var_ypl[np * np * k + i * np + i] + eps);
		}
	}

	for (k = 1; k < nc; k++) {
		for (i = 0; i < np; i++) {

			beta[k * np + i] = -0.5 * (log(prec_g[i]) - log(prec_gl[k * np + i]));
		}

		daxpy_(&Nl, &DA, &beta[k * np], &INCX, beta_sum, &INCY);
	}

	for (k = 0; k < nc; k++) {

		get_tensor_product(wt, &beta[k * np], &prec_gl[k * np], np);

		get_tensor_product(yp, wt, &ypl[k * np], np);

		daxpy_(&Nl, &DA, yp, &INCX, yp_sum, &INCY);

		daxpy_(&Nl, &DA, wt, &INCX, prec_sum, &INCY);
	}

	get_tensor_product(wt, beta_sum, prec_g, np);

	get_tensor_product(yp, wt, ypg, np);

	DA = -1.0;

	daxpy_(&Nl, &DA, yp, &INCX, yp_sum, &INCY);

	daxpy_(&Nl, &DA, wt, &INCX, prec_sum, &INCY);

	for (i = 0; i < np; i++) {
		prec_sum[i] = 1 / prec_sum[i];
	}

	get_tensor_product(yp, prec_sum, yp_sum, np);

	for (i = 0; i < np; i++) {
		var_yp[i * np + i] = prec_sum[i];
	}

	free(prec_sum);
	free(yp_sum);
	free(wt);
	free(beta);
	free(beta_sum);
	free(prec_g);
	free(prec_gl);
}

void gpr_interpolate_experts(
    double *yp, double *var_yp, const double *xp, unsigned long np, const double *x,
    const double *y, unsigned long ns, const double *xc, unsigned long nc, unsigned int dim,
    double *hp, unsigned long nhp, int is_opt,
    void covar(double *krn, const double *x, const double *xp, unsigned long nx, unsigned long nxp,
	       unsigned int dim, const double *p, unsigned int npar, void *dat),
    void covar_jac(double *dK, unsigned int k, const double *x, const double *kxx, unsigned long nx,
		   unsigned int dim, const double *p, unsigned int np, void *dat),
    void *dat, unsigned int gate)

{

	double *ypc, *var_ypc;
	unsigned long i, nsc, nhpc, *ind;

	ypc = malloc(np * nc * sizeof(double));
	assert(ypc);
	var_ypc = malloc(np * np * nc * sizeof(double));
	assert(var_ypc);

	nsc = ns / nc;
	nhpc = nhp / nc;

	for (i = 0; i < nc; i++) {

		gpr_interpolate_wrap(xp, &ypc[i * np], np, &x[i * nsc * dim], &y[i * nsc], nsc, dim,
				     &hp[i * nhpc], nhpc, &var_ypc[i * np * np], is_opt, covar,
				     covar_jac, dat);
	}

	if (gate == 0) {

		prod_experts(yp, var_yp, np, ypc, var_ypc, np * nc);

	} else if (gate == 1) {

		weighted_prod_experts(yp, var_yp, np, ypc, var_ypc, np * nc, covar, xp, dim, hp,
				      nhp, dat);

	} else if (gate == 2) {

		ind = malloc(np * sizeof(unsigned long));
		assert(ind);

		get_indicator(ind, xp, np, dim, xc, nc);

		bcm_experts(yp, var_yp, np, ypc, var_ypc, np * nc, covar, xp, dim, hp, nhp, ind,
			    dat);

		free(ind);

	} else if (gate == 3) {

		rbcm_experts(yp, var_yp, np, ypc, var_ypc, np * nc, covar, xp, dim, hp, nhp, dat);
	}

	free(ypc);
	free(var_ypc);
}

void prod_experts(double *yp, double *var_yp, unsigned long np, const double *ypc,
		  const double *var_ypc, unsigned long npc)
{
	unsigned long i, k, nc;
	long Nl;
	int N, INCX, INCY;
	double *prec, *tmp, *prec_sum, DA, eps;

	eps = 1E-7;

	nc = npc / np;

	prec = malloc(np * nc * sizeof(double));
	assert(prec);
	tmp = calloc(np, sizeof(double));
	assert(tmp);
	prec_sum = calloc(np, sizeof(double));
	assert(prec_sum);

	for (k = 0; k < nc; k++) {
		for (i = 0; i < np; i++) {

			prec[k * np + i] = 1 / (var_ypc[np * np * k + i * np + i] + eps);
		}
	}

	Nl = np;
	N = (int)np;
	DA = 1.0;
	INCX = 1;
	INCY = 1;

	for (k = 0; k < nc; k++) {
		daxpy_(&Nl, &DA, &prec[k * np], &INCX, prec_sum, &INCY);
	}

	for (i = 0; i < nc; i++) {

		get_tensor_product(yp, &prec[i * np], &ypc[i * np], np);

		daxpy_(&Nl, &DA, yp, &INCX, tmp, &INCY);
	}

	for (i = 0; i < np; i++) {
		prec_sum[i] = 1 / prec_sum[i];
	}

	get_tensor_product(yp, prec_sum, tmp, np);

	for (i = 0; i < np; i++) {
		var_yp[i * np + i] = prec_sum[i];
	}

	free(prec);
	free(prec_sum);
	free(tmp);
}

void weighted_prod_experts(double *yp, double *var_yp, unsigned long np, const double *ypc,
			   const double *var_ypc, unsigned long npc,
			   void covar(double *krn, const double *x, const double *xp,
				      unsigned long nx, unsigned long nxp, unsigned int dim,
				      const double *p, unsigned int npar, void *dat),
			   const double *xp, unsigned int dim, const double *hp, unsigned long nhp,
			   void *dat)

{
	unsigned long i, k, nc, nhps;
	long Nl;
	int N, INCX, INCY;
	double *beta, *prec, tmp, *prec_sum, DA, *wt, *yp_sum, eps, *prior_prec, *prior_prec_sum;

	nc = npc / np;
	nhps = nhp / nc;

	prec = malloc(np * nc * sizeof(double));
	assert(prec);
	beta = malloc(np * nc * sizeof(double));
	assert(beta);
	yp_sum = calloc(np, sizeof(double));
	assert(yp_sum);
	prior_prec = calloc(np * nc, sizeof(double));
	assert(prior_prec);
	prior_prec_sum = calloc(np, sizeof(double));
	assert(prior_prec_sum);
	wt = malloc(np * sizeof(double));
	assert(wt);
	prec_sum = calloc(np, sizeof(double));
	assert(prec_sum);

	eps = 1E-7;

	Nl = np;
	N = (int)np;
	DA = 1.0;
	INCX = 1;
	INCY = 1;

	for (k = 0; k < nc; k++) {
		for (i = 0; i < np; i++) {

			covar(&tmp, &xp[i * dim], &xp[i * dim], 1, 1, dim, &hp[k * nhps], nhps,
			      dat);

			prior_prec[k * np + i] = 1 / tmp;
		}
	}

	for (k = 0; k < nc; k++) {

		daxpy_(&Nl, &DA, &prior_prec[k * np], &INCX, prior_prec_sum, &INCY);
	}

	for (k = 0; k < nc; k++) {
		for (i = 0; i < np; i++) {

			prec[k * np + i] = 1 / (var_ypc[np * np * k + i * np + i] + eps);

			beta[k * np + i] = -0.5 * (log(prior_prec_sum[i]) - log(prec[k * np + i]));
		}
	}

	for (k = 0; k < nc; k++) {

		get_tensor_product(wt, &prec[k * np], &beta[k * np], np);

		daxpy_(&Nl, &DA, wt, &INCX, prec_sum, &INCY);

		get_tensor_product(yp, &ypc[k * np], wt, np);

		daxpy_(&Nl, &DA, yp, &INCX, yp_sum, &INCY);
	}

	for (i = 0; i < np; i++) {
		prec_sum[i] = 1 / prec_sum[i];
	}

	get_tensor_product(yp, prec_sum, yp_sum, np);

	for (i = 0; i < np; i++) {
		var_yp[i * np + i] = prec_sum[i];
	}

	free(prec_sum);
	free(wt);
	free(prior_prec_sum);
	free(prior_prec);
	free(yp_sum);
	free(beta);
	free(prec);
}

void bcm_experts(double *yp, double *var_yp, unsigned long np, const double *ypc,
		 const double *var_ypc, unsigned long npc,
		 void covar(double *krn, const double *x, const double *xp, unsigned long nx,
			    unsigned long nxp, unsigned int dim, const double *p, unsigned int npar,
			    void *dat),
		 const double *xp, unsigned int dim, const double *hp, unsigned long nhp,
		 const unsigned long *ind, void *dat)

{
	unsigned long i, k, nc, nhps;
	long Nl;
	int N, INCX, INCY;
	double *prec, *prec_sum, DA, *wt, tmp, *yp_sum, *prior_prec, eps;

	nc = npc / np;
	nhps = nhp / nc;

	prec = malloc(np * nc * sizeof(double));
	assert(prec);
	prior_prec = malloc(np * nc * sizeof(double));
	assert(prior_prec);
	yp_sum = calloc(np, sizeof(double));
	assert(yp_sum);
	prec_sum = calloc(np, sizeof(double));
	assert(prec_sum);

	eps = 1E-7;

	for (k = 0; k < nc; k++) {
		for (i = 0; i < np; i++) {

			prec[k * np + i] = 1 / (var_ypc[np * np * k + i * np + i] + eps);

			covar(&tmp, &xp[i * dim], &xp[i * dim], 1, 1, dim, &hp[k * nhps], nhps,
			      dat);

			prior_prec[k * np + i] = prec[k * np + i] - (1 / tmp);
		}
	}

	Nl = np;
	N = (int)np;
	DA = 1.0;
	INCX = 1;
	INCY = 1;

	for (k = 0; k < nc; k++) {
		get_tensor_product(yp, &ypc[k * np], &prec[k * np], np);

		daxpy_(&Nl, &DA, yp, &INCX, yp_sum, &INCY);

		daxpy_(&Nl, &DA, &prior_prec[k * np], &INCX, prec_sum, &INCY);
	}

	for (i = 0; i < np; i++) {
		prec_sum[i] = 1 / prec_sum[i];
	}

	get_tensor_product(yp, prec_sum, yp_sum, np);

	for (i = 0; i < np; i++) {
		var_yp[i * np + i] = prec_sum[i];
	}

	free(prior_prec);
	free(prec_sum);
	free(yp_sum);
	free(prec);
}

void rbcm_experts(double *yp, double *var_yp, unsigned long np, const double *ypc,
		  const double *var_ypc, unsigned long npc,
		  void covar(double *krn, const double *x, const double *xp, unsigned long nx,
			     unsigned long nxp, unsigned int dim, const double *p,
			     unsigned int npar, void *dat),
		  const double *xp, unsigned int dim, const double *hp, unsigned long nhp,
		  void *dat)

{
	unsigned long i, k, nc, nhps;
	long Nl, Ml;
	int N, INCX, INCY;
	double *beta, *prec, tmp, *prec_sum, DA, *wt, *yp_sum, eps, *prior_prec, *prior_prec_sum;

	nc = npc / np;
	nhps = nhp / nc;

	prec = malloc(np * nc * sizeof(double));
	assert(prec);
	beta = malloc(np * nc * sizeof(double));
	assert(beta);
	yp_sum = calloc(np, sizeof(double));
	assert(yp_sum);
	prior_prec = calloc(np * nc, sizeof(double));
	assert(prior_prec);
	prior_prec_sum = calloc(np, sizeof(double));
	assert(prior_prec_sum);
	wt = malloc(np * sizeof(double));
	assert(wt);
	prec_sum = calloc(np, sizeof(double));
	assert(prec_sum);

	eps = 1E-7;

	Nl = np;
	Ml = np * nc;
	N = (int)np;
	DA = 1.0;
	INCX = 1;
	INCY = 1;

	for (k = 0; k < nc; k++) {
		for (i = 0; i < np; i++) {

			covar(&tmp, &xp[i * dim], &xp[i * dim], 1, 1, dim, &hp[k * nhps], nhps,
			      dat);

			prior_prec[k * np + i] = 1 / tmp;
		}
	}

	for (k = 0; k < nc; k++) {

		daxpy_(&Nl, &DA, &prior_prec[k * np], &INCX, prior_prec_sum, &INCY);
	}

	for (k = 0; k < nc; k++) {
		for (i = 0; i < np; i++) {

			prec[k * np + i] = 1 / (var_ypc[np * np * k + i * np + i] + eps);

			beta[k * np + i] = -0.5 * (log(prior_prec_sum[i]) - log(prec[k * np + i]));

			prior_prec[k * np + i] = prec[k * np + i] - prior_prec[k * np + i];
		}
	}

	for (k = 0; k < nc; k++) {

		get_tensor_product(wt, &prec[k * np], &beta[k * np], np);

		get_tensor_product(yp, &ypc[k * np], wt, np);

		daxpy_(&Nl, &DA, yp, &INCX, yp_sum, &INCY);

		get_tensor_product(wt, &prior_prec[k * np], &beta[k * np], np);

		daxpy_(&Nl, &DA, wt, &INCX, prec_sum, &INCY);
	}

	for (i = 0; i < np; i++) {
		prec_sum[i] = 1 / prec_sum[i];
	}

	get_tensor_product(yp, prec_sum, yp_sum, np);

	for (i = 0; i < np; i++) {
		var_yp[i * np + i] = prec_sum[i];
	}

	free(prec_sum);
	free(wt);
	free(prior_prec_sum);
	free(prior_prec);
	free(yp_sum);
	free(beta);
	free(prec);
}
