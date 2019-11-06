#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_ode/lib_ode.h>
#include "lib_gpr.h"

#define EXP_HALF (0.6065306597126334) /* exp(-0.5) */

void gpr_etd34rk_vec_step(double t0, unsigned long m, double *y0, double h, double *J,
			  double *exp_hj2, double *enf_jh2, double *enf_jh, double *alph,
			  double *bet, double *gam,
			  void fun(double *f, double t, double *u1, unsigned long mn, void *param),
			  void *param, double *y, double *eg, const double *x, unsigned int dim,
			  double *hparam, unsigned int nhparam, double *cv_step, unsigned long ntst,
			  unsigned long nbtch, enum estimator est, double *work,
			  unsigned long nwork)
{
	double *a, *b, *b3, *c, *f0, *fa, *fb, *fc, *fb3;
	unsigned long i;

	assert(work);
	assert(nwork == get_work_sz_etd34rk(m));
	a = &work[0 * m];
	b = &work[1 * m];
	c = &work[2 * m];
	f0 = &work[3 * m];
	fa = &work[4 * m];
	fb = &work[5 * m];
	fc = &work[6 * m];
	b3 = &work[7 * m];
	fb3 = &work[8 * m];

	if (cv_step) {
		cv_step[0] = get_gpr_cv_holdout(x, y0, m, dim, hparam, nhparam, ntst, nbtch, est);
	}

	fun(f0, t0, y0, m, param);

	for (i = 0; i < m; i++) {
		a[i] = y0[i] * exp_hj2[i] + 0.5 * h * enf_jh2[i] * f0[i];
	}

	if (cv_step) {
		cv_step[1] = get_gpr_cv_holdout(x, a, m, dim, hparam, nhparam, ntst, nbtch, est);
	}

	fun(fa, t0 + 0.5 * h, a, m, param);

	for (i = 0; i < m; i++) {
		b[i] = y0[i] * exp_hj2[i] + 0.5 * h * enf_jh2[i] * fa[i];
		b3[i] = y0[i] * exp_hj2[i] * exp_hj2[i] + h * enf_jh[i] * (2 * fa[i] - f0[i]);
	}

	if (cv_step) {
		cv_step[2] = get_gpr_cv_holdout(x, b, m, dim, hparam, nhparam, ntst, nbtch, est);
		cv_step[3] = get_gpr_cv_holdout(x, b3, m, dim, hparam, nhparam, ntst, nbtch, est);
	}

	fun(fb, t0 + 0.5 * h, b, m, param);
	fun(fb3, t0 + h, b3, m, param);

	for (i = 0; i < m; i++) {
		c[i] = a[i] * exp_hj2[i] + 0.5 * h * enf_jh2[i] * (2 * fb[i] - f0[i]);
	}

	if (cv_step) {
		cv_step[4] = get_gpr_cv_holdout(x, c, m, dim, hparam, nhparam, ntst, nbtch, est);
	}

	fun(fc, t0 + h, c, m, param);

	for (i = 0; i < m; i++) {
		y[i] = y0[i] * exp_hj2[i] * exp_hj2[i]
		       + h * (f0[i] * alph[i] + 2 * (fa[i] + fb[i]) * bet[i] + fc[i] * gam[i]);

		eg[i] += h * fabs(2 * bet[i] * (fa[i] - fb[i]) + gam[i] * (fc[i] - fb3[i]));
	}

	if (cv_step) {
		cv_step[5] = get_gpr_cv_holdout(x, y, m, dim, hparam, nhparam, ntst, nbtch, est);
	}
}

void gpr_etd34rk_vec(double t0, double tn, double h, double *y0, unsigned long n, double *J,
		     void fn(double *f, double t, double *u1, unsigned long mn, void *param),
		     void *param, double *eg, const double *x, unsigned int dim, double *hparam,
		     unsigned int nhparam, double *cv_step, double *cv, unsigned long ntst,
		     unsigned long nbtch, enum estimator est)
{
	double *dy, *work, *alp, *bet, *gam, *exp_jh2, *enf_jh2, *enf_jh, t;
	unsigned long nwork, i, j;
	unsigned int netd4rk;

	netd4rk = 6;

	nwork = get_work_sz_etd34rk(n);

	work = malloc(nwork * sizeof(double));
	assert(work);
	dy = malloc(n * sizeof(double));
	assert(dy);
	exp_jh2 = malloc(n * sizeof(double));
	assert(exp_jh2);
	enf_jh2 = malloc(n * sizeof(double));
	assert(enf_jh2);
	enf_jh = malloc(n * sizeof(double));
	assert(enf_jh);
	alp = malloc(n * sizeof(double));
	assert(alp);
	bet = malloc(n * sizeof(double));
	assert(bet);
	gam = malloc(n * sizeof(double));
	assert(gam);

	for (i = 0; i < n; i++) {
		eg[i] = 0;
	}

	get_expz(exp_jh2, J, 0.5 * h, n);
	get_enf(enf_jh2, J, 0.5 * h, n);
	get_enf(enf_jh, J, h, n);
	get_etd4rk_coeff(alp, bet, gam, J, h, n);

	j = 0;
	for (t = t0; t <= tn; t += h) {
		fprintf(stderr, "\r t = %+.15E  h = %.15E", t, h);

		cv[j++] = get_gpr_cv_holdout(x, y0, n, dim, hparam, nhparam, ntst, nbtch, est);

		gpr_etd34rk_vec_step(t, n, y0, h, J, exp_jh2, enf_jh2, enf_jh, alp, bet, gam, fn,
				     param, dy, eg, x, dim, hparam, nhparam,
				     &cv_step[netd4rk * (j++)], ntst, nbtch, est, work, nwork);

		for (i = 0; i < n; i++) {
			y0[i] = dy[i];
		}
	}

	fprintf(stderr, "\n");

	free(work);
	free(dy);
	free(alp);
	free(bet);
	free(gam);
	free(exp_jh2);
	free(enf_jh);
	free(enf_jh2);
}
