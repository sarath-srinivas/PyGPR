#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "lib_gpr.h"

int get_krn_rat_quad(double *krn, const double *x, const double *xp, unsigned long nx, unsigned long nxp,
		     unsigned long dim, const double *par, int npar, const double *hpar, int nhpar)
{
	double sig_y, l, l2, r2, x_xp, alph;
	unsigned long i, j, k;

	assert(npar == 1);
	assert(nhpar == 1);

	sig_y = 1.0;
	l = hpar[0];
	alph = par[0];
	l2 = l * l;

	for (i = 0; i < nx; i++)
		for (j = 0; j < nxp; j++) {

			r2 = 0;
			for (k = 0; k < dim; k++) {
				x_xp = x[dim * i + k] - xp[dim * j + k];
				r2 += x_xp * x_xp;
			}

			krn[i * nxp + j] = sig_y * sig_y * pow((1 + r2 / (2 * alph * l2)), -alph);
		}

	return 0;
}
