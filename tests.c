#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "lib_gpr.h"

static void f_2d(const double *x, unsigned long ns, double *y, int fno)
{
	double x1, x2;
	unsigned long i;
	unsigned int dim;

	dim = 2;

	for (i = 0; i < ns; i++) {
		x1 = x[dim * i + 0];
		x2 = x[dim * i + 1];

		y[i] = x1 * sin(x2) + x2 * sin(x1);
	}
}

double test_gpr_interpolate(unsigned long ns, unsigned long np, int fno, int seed)
{
	double *x, *xp, *y, *yp, *yt, *p, *var_yp;
	unsigned int dim, npar;
	unsigned long i;

	dim = 2;
	npar = 2;

	x = malloc(dim * ns * sizeof(double));
	assert(x);

	xp = malloc(dim * np * sizeof(double));
	assert(xp);

	y = malloc(ns * sizeof(double));
	assert(y);

	yp = malloc(np * sizeof(double));
	assert(yp);

	yt = malloc(np * sizeof(double));
	assert(yt);

	p = malloc(npar * sizeof(double));
	assert(p);

	var_yp = malloc(np * np * sizeof(double));
	assert(p);

	srand(seed);
	for (i = 0; i < ns; i++) {
		x[dim * i + 0] = rand() / (RAND_MAX + 1.0);
		x[dim * i + 1] = rand() / (RAND_MAX + 1.0);
	}

	f_2d(x, ns, y, 1);

	srand(seed + 1232);
	for (i = 0; i < np; i++) {
		xp[dim * i + 0] = rand() / (RAND_MAX + 1.0);
		xp[dim * i + 1] = rand() / (RAND_MAX + 1.0);
	}

	f_2d(xp, np, yt, 1);

	for (i = 0; i < npar; i++) {
		p[i] = 1.0;
	}

	gpr_interploate(y, x, ns, dim, xp, np, yp, p, npar, var_yp);

	for (i = 0; i < np; i++) {
		printf("%+.15E %+.15E %+.15E %+.15E %+.15E %+.15E\n", xp[dim * i + 0], xp[dim * i + 1], yp[i],
		       yt[i], sqrt(var_yp[i * np + i]), fabs(yp[i] - yt[i]));
	}

	free(x);
	free(xp);
	free(y);
	free(yp);
	free(yt);
	free(p);
	free(var_yp);

	return 0;
}
