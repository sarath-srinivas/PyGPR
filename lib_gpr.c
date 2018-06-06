#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "lib_gpr.h"

extern void dposv_(const unsigned char *UPLO, const int *N, const int *NRHS, double *A, const int *LDA,
		   double *B, const int *LDB, int *info);

int get_krn_se(double *krn, const double *x, const double *xp, unsigned long ns, unsigned long dim,
	       const double *p, int np)
{
	double sig_y, sig_n, l, l2, r2, x_xp;
	unsigned long i, j, k;

	assert(np == 3);

	sig_y = p[0];
	sig_n = p[1];
	l = p[2];
	l2 = l * l;

	for (i = 0; i < ns; i++)
		for (j = 0; j < ns; j++) {

			r2 = 0;
			for (k = 0; k < dim; k++) {
				x_xp = x[dim * i + k] - xp[dim * j + k];
				r2 += x_xp * x_xp;
			}

			krn[i * ns + j] = sig_y * sig_y * exp(-0.5 * r2 / l2);
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
