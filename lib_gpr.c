#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "lib_gpr.h"

int get_krn_se(double *krn, const double *x, const double *xp, unsigned long nx, unsigned long nxp,
	       unsigned long dim, const double *p, int npar)
{
	double sig_y, sig_n, l, l2, r2, x_xp, noise;
	unsigned long i, j, k;

	assert(npar == 3);

	sig_y = p[0];
	sig_n = p[1];
	l = p[2];
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

	for (i = 0; i < nx; i++) {
		krn[i * nxp + i] += sig_n * sig_n;
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
