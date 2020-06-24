#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <blas/lapack.h>
#include <blas/blas.h>
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

void gpr_interpolate_experts(double *yp, double *var_yp, const double *xp, unsigned long np,
			     const double *x, const double *y, unsigned long ns, unsigned long nc,
			     unsigned int dim, double *hp, unsigned long nhp, int is_opt,
			     void covar(double *krn, const double *x, const double *xp,
					unsigned long nx, unsigned long nxp, unsigned int dim,
					const double *p, unsigned int npar, void *dat),
			     void covar_jac(double *dK, unsigned int k, const double *x,
					    const double *kxx, unsigned long nx, unsigned int dim,
					    const double *p, unsigned int np, void *dat),
			     void *dat, unsigned int gate)

{

	double *ypc, *var_ypc;
	unsigned long i, nsc, nhpc;

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
	}

	else if (gate == 2) {

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
	double *beta, *prec, tmp, *prec_sum, DA, *wt, *yp_sum, eps;

	nc = npc / np;
	nhps = nhp / nc;

	prec = malloc(np * nc * sizeof(double));
	assert(prec);
	beta = malloc(np * nc * sizeof(double));
	assert(beta);
	yp_sum = calloc(np, sizeof(double));
	assert(yp_sum);
	wt = malloc(np * sizeof(double));
	assert(wt);
	prec_sum = calloc(np, sizeof(double));
	assert(prec_sum);

	eps = 1E-7;

	for (k = 0; k < nc; k++) {
		for (i = 0; i < np; i++) {

			prec[k * np + i] = 1 / (var_ypc[np * np * k + i * np + i] + eps);

			covar(&tmp, &xp[i], &xp[i], 1, 1, dim, &hp[k * nhps], nhps, dat);

			beta[k * np + i] = 0.5 * (log(tmp) + log(prec[k * np + i]));
		}
	}

	Nl = np;
	N = (int)np;
	DA = 1.0;
	INCX = 1;
	INCY = 1;

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
	free(yp_sum);
	free(beta);
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
	long Nl;
	int N, INCX, INCY;
	double *beta, *prec, *prec_sum, DA, *wt, tmp, *yp_sum, *prior_prec, eps;

	nc = npc / np;
	nhps = nhp / nc;

	prec = malloc(np * nc * sizeof(double));
	assert(prec);
	prior_prec = malloc(np * nc * sizeof(double));
	assert(prior_prec);
	beta = malloc(np * nc * sizeof(double));
	assert(beta);
	yp_sum = calloc(np, sizeof(double));
	assert(yp_sum);
	wt = malloc(np * sizeof(double));
	assert(wt);
	prec_sum = calloc(np, sizeof(double));
	assert(prec_sum);

	eps = 1E-7;

	for (k = 0; k < nc; k++) {
		for (i = 0; i < np; i++) {

			prec[k * np + i] = 1 / (var_ypc[np * np * k + i * np + i] + eps);

			covar(&tmp, &xp[i], &xp[i], 1, 1, dim, &hp[k * nhps], nhps, dat);

			prior_prec[k * np + i] = prec[k * np + i] - (1 / tmp);

			beta[k * np + i] = 0.5 * (log(tmp) + log(prec[k * np + i]));
		}
	}

	Nl = np;
	N = (int)np;
	DA = 1.0;
	INCX = 1;
	INCY = 1;

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
	free(yp_sum);
	free(beta);
	free(prec);
}
