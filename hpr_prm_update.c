#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "lib_gpr.h"

void update_hpr_prm(double *hp, double *jac, unsigned long nhp, double h, const double *x,
		    const double *y, const double *r2, unsigned long nx, unsigned int dim)
{

	double f0, fhp, fhm, *hp_p, *hp_m, a, b, h2, alph;
	unsigned i;
	struct gpr_dat *dat;

	dat = malloc(1 * sizeof(struct gpr_dat));
	assert(dat);

	dat->x = x;
	dat->y = y;
	dat->ns = nx;
	dat->dim = dim;
	dat->r2 = r2;

	h2 = h * h;

	hp_p = malloc(nhp * sizeof(double));
	assert(hp_p);
	hp_m = malloc(nhp * sizeof(double));
	assert(hp_m);

	get_f_jac(&f0, jac, hp, nhp, dat);

	for (i = 0; i < nhp; i++) {
		hp_p[i] = hp[i] + h * jac[i];
		hp_m[i] = hp[i] - h * jac[i];
	}

	fhp = get_f(hp_p, nhp, dat);
	fhm = get_f(hp_m, nhp, dat);

	a = (fhp + fhm - 2 * f0) / (2 * h2);

	b = (fhp - fhm) / (2 * h);

	alph = -b / (2 * a);

	for (i = 0; i < nhp; i++) {
		hp[i] = hp[i] - alph * jac[i];
	}

	free(hp_m);
	free(hp_p);
	free(dat);
}
