#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <lib_ode/lib_ode.h>
#include "lib_gpr.h"

void gpr_rk45vec_step(double t0, unsigned long m, double *y0, double h,
		      void fun(double *f, double t, double *u1, unsigned long mn, void *param),
		      void *param, double *y, double *e, const double *x, unsigned int dim,
		      double *hparam, unsigned int nhparam, double *cv_step, unsigned long ntst,
		      unsigned long nbtch, enum estimator est, double *work, unsigned long nwork)
{
	double *k0, *k1, *k2, *k3, *k4, *k5, *y1, *y2, *y3, *y4, *y5, t[6];
	double c[6] = {0, 0.25, 0.375, 0.923077, 1, 0.5};
	double a[6][6] = {{+0.000000, +0.000000, +0.000000, +0.000000, +0.000000, +0.000000},
			  {+0.250000, +0.000000, +0.000000, +0.000000, +0.000000, +0.000000},
			  {+0.093750, +0.281250, +0.000000, +0.000000, +0.000000, +0.000000},
			  {+0.879381, -3.277196, +3.320892, +0.000000, +0.000000, +0.000000},
			  {+2.032407, -8.000000, +7.173489, -0.205897, +0.000000, +0.000000},
			  {-0.296296, +2.000000, -1.381676, +0.452973, -0.275000, +0.000000}};
	double b5[6] = {+0.118519, +0.000000, +0.518986, +0.506131, -0.180000, +0.036364};
	/*double b4[6] = {+0.115741, +0.000000, +0.548928, +0.535331, -0.200000, +0.000000};*/
	double b45[6] = {+0.002778, +0.000000, -0.029942, -0.029200, +0.020000, +0.036364};
	unsigned long i;

	assert(work);
	assert(nwork == get_work_sz_rk45(m));
	y1 = &work[0 * m];
	y2 = &work[1 * m];
	y3 = &work[2 * m];
	y4 = &work[3 * m];
	y5 = &work[4 * m];
	k0 = &work[5 * m];
	k1 = &work[6 * m];
	k2 = &work[7 * m];
	k3 = &work[8 * m];
	k4 = &work[9 * m];
	k5 = &work[10 * m];

	for (i = 0; i < 6; i++) {
		t[i] = t0 + h * c[i];
	}

	fun(k0, t0, y0, m, param);

	for (i = 0; i < m; i++) {
		y1[i] = y0[i] + h * a[1][0] * k0[i];
	}

	fun(k1, t[1], y1, m, param);

	for (i = 0; i < m; i++) {
		y2[i] = y0[i] + h * (a[2][0] * k0[i] + a[2][1] * k1[i]);
	}

	fun(k2, t[2], y2, m, param);

	for (i = 0; i < m; i++) {
		y3[i] = y0[i] + h * (a[3][0] * k0[i] + a[3][1] * k1[i] + a[3][2] * k2[i]);
	}

	fun(k3, t[3], y3, m, param);

	for (i = 0; i < m; i++) {
		y4[i]
		    = y0[i]
		      + h * (a[4][0] * k0[i] + a[4][1] * k1[i] + a[4][2] * k2[i] + a[4][3] * k3[i]);
	}

	fun(k4, t[4], y4, m, param);

	for (i = 0; i < m; i++) {
		y5[i] = y0[i]
			+ h
			      * (a[5][0] * k0[i] + a[5][1] * k1[i] + a[5][2] * k2[i]
				 + a[5][3] * k3[i] + a[5][4] * k4[i]);
	}

	fun(k5, t[5], y5, m, param);

	for (i = 0; i < m; i++) {
		y[i] = y0[i]
		       + h
			     * (b5[0] * k0[i] + b5[1] * k1[i] + b5[2] * k2[i] + b5[3] * k3[i]
				+ b5[4] * k4[i] + b5[5] * k5[i]);
		e[i] += fabs(h
			     * (b45[0] * k0[i] + b45[1] * k1[i] + b45[2] * k2[i] + b45[3] * k3[i]
				+ b45[4] * k4[i] + b45[5] * k5[i]));
	}

	if (cv_step) {
		get_gpr_cv_holdout(&cv_step[0], x, y0, m, dim, hparam, nhparam, ntst, nbtch, est);
		get_gpr_cv_holdout(&cv_step[3], x, y1, m, dim, hparam, nhparam, ntst, nbtch, est);
		get_gpr_cv_holdout(&cv_step[6], x, y2, m, dim, hparam, nhparam, ntst, nbtch, est);
		get_gpr_cv_holdout(&cv_step[9], x, y3, m, dim, hparam, nhparam, ntst, nbtch, est);
		get_gpr_cv_holdout(&cv_step[12], x, y4, m, dim, hparam, nhparam, ntst, nbtch, est);
		get_gpr_cv_holdout(&cv_step[15], x, y5, m, dim, hparam, nhparam, ntst, nbtch, est);
		get_gpr_cv_holdout(&cv_step[18], x, y, m, dim, hparam, nhparam, ntst, nbtch, est);
	}
}

void gpr_rk45vec(double t0, double tn, double h, double *y0, unsigned long n,
		 void fun(double *f, double t, double *u1, unsigned long mn, void *param),
		 double tol, void *param, double *eg, const double *x, unsigned int dim,
		 double *hparam, unsigned int nhparam, double *cv_step, double *cv,
		 unsigned long ntst, unsigned long nbtch, enum estimator est)
{
	double *dy, t, *work;
	unsigned long i, j, nwork;
	unsigned int nrk45;

	nrk45 = 7;

	nwork = get_work_sz_rk45(n);

	work = malloc(nwork * sizeof(double));
	assert(work);
	dy = malloc(n * sizeof(double));
	assert(dy);

	for (i = 0; i < n; i++) {
		eg[i] = 0;
	}

	j = 0;
	for (t = t0; t <= tn; t += h) {

		fprintf(stderr, "\r t = %+.15E  h = %.15E", t, h);

		get_gpr_cv_holdout(&cv[3 * (j++)], x, y0, n, dim, hparam, nhparam, ntst, nbtch,
				   est);

		gpr_rk45vec_step(t, n, y0, h, fun, param, dy, eg, x, dim, hparam, nhparam, NULL,
				 ntst, nbtch, est, work, nwork);

		for (i = 0; i < n; i++) {
			y0[i] = dy[i];
		}
	}

	fprintf(stderr, "\n");

	free(dy);
	free(work);
}
