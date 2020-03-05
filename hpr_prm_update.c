#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "lib_gpr.h"

void update_hpr_prm(double *hp, unsigned long nhp, double h, void *dat)
{

	double f0, fhp, fhm, *jac, *hp_p, *hp_m, a, b, h2, alph;
	unsigned i;

	h2 = h * h;

	jac = malloc(nhp * sizeof(double));
	assert(jac);
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
	free(jac);
}
