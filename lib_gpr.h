
#ifndef DEBUG
#define DEBUG (0)
#endif

/* COVARIANCE FUNCTIONS */

#include "covars.h"

/* GPR REGRESSION */

#include "gpr.h"

/* HYPERPARAMETER OPTIMISATION */

#include "hyperopt.h"

/* CROSS VALIDATION */

#include "cv.h"

/* DISTRIBUTED GPR */

#include "distr_gpr.h"

/* TESTS */

double test_gpr_interpolate(unsigned long ns, unsigned long np, unsigned int dim, int seed);
double test_gpr_interpolate_experts(unsigned long nsc, unsigned long nc, unsigned long np,
				    unsigned int dim, double min_dist, unsigned int gate, int seed);
void test_lib_gpr(void);

/* UPDATE HYPERPARAM */

void update_hpr_prm(double *hp, double *jac, unsigned long nhp, double h, const double *x,
		    const double *y, const double *r2, unsigned long nx, unsigned int dim);

/* GPR RK45 ODE */
void gpr_rk45vec_step(double t0, unsigned long m, double *y0, double h,
		      void fun(double *f, double t, double *u1, unsigned long mn, void *param),
		      void *param, double *y, double *e, const double *x, unsigned int dim,
		      double *hparam, unsigned int nhparam, double *cv_step, unsigned long ntst,
		      unsigned long nbtch, enum estimator est, double *work, unsigned long nwork);
void gpr_rk45vec(double t0, double h, unsigned long nstep, double *y0, unsigned long n,
		 void fun(double *f, double t, double *u1, unsigned long mn, void *param),
		 double tol, void *param, double *eg, const double *x, unsigned int dim,
		 double *hparam, unsigned int nhparam, double *cv_step, double *cv,
		 unsigned long ntst, unsigned long nbtch, enum estimator est);

/* GPR ETD34RK ODE */
void gpr_etd34rk_vec_step(double t0, unsigned long m, double *y0, double h, double *J,
			  double *exp_hj2, double *enf_jh2, double *enf_jh, double *alph,
			  double *bet, double *gam,
			  void fun(double *f, double t, double *u1, unsigned long mn, void *param),
			  void *param, double *y, double *eg, const double *x, unsigned int dim,
			  double *hparam, unsigned int nhparam, double *cv_step, unsigned long ntst,
			  unsigned long nbtch, enum estimator est, double *work,
			  unsigned long nwork);
void gpr_etd34rk_vec(double t0, double tn, double h, double *y0, unsigned long n, double *J,
		     void fn(double *f, double t, double *u1, unsigned long mn, void *param),
		     void *param, double *eg, const double *x, unsigned int dim, double *hparam,
		     unsigned int nhparam, double *cv_step, double *cv, unsigned long ntst,
		     unsigned long nbtch, enum estimator est);

/* TESTS */
void test_get_subsample_cv_holdout(unsigned long n, unsigned long ntst, unsigned long k,
				   unsigned int dim, int seed);
void test_get_gpr_cv_holdout(unsigned long n, unsigned int dim, unsigned long ntst,
			     unsigned long nbtch, enum estimator est, int seed);

/* LIB TEST */
