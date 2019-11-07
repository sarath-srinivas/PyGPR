#define NP_SE (1)
#define NP_SE_ARD (6)

struct gpr_dat {
	unsigned long int ns;
	int dim;
	double *x;
	double *y;
	double *r2;
};

enum estimator { CHI_SQ, MAHALANOBIS };

void get_gpr_weights(double *wt, double *krn_chd, const double *krn, unsigned long ns,
		     unsigned long dim, const double *y);
void gpr_predict(double *yp, const double *wt, const double *krnp, unsigned long np,
		 const unsigned long ns);
void get_var_mat(double *var, double *krnpp, double *krnp, double *krn_chd, unsigned long np,
		 unsigned long ns);
double get_log_likelihood(const double *wt, const double *y, unsigned long ns,
			  const double *krn_chd, double *ret);
void get_var_mat_chd(double *var, const double *krnpp, const double *krnp, const double *krn_chd,
		     unsigned long np, unsigned long ns);
void get_hyper_param_ard(double *p, int np, double *x, double *y, unsigned long ns, int dim);
void get_hyper_param_ard_stoch(double *p, int np, double *x, double *y, unsigned long ns, int dim,
			       unsigned long nsub, double lrate, int seed);
void gpr_interpolate(double *xp, double *yp, unsigned long np, double *x, double *y,
		     unsigned long ns, unsigned int dim, double *p, unsigned int npar,
		     double *var_yp, int is_opt);
void gpr_interpolate_mean(double *xp, double *yp, double *yp_mn, unsigned long np, double *x,
			  double *y, double *y_mn, unsigned long ns, unsigned int dim, double *p,
			  unsigned int npar, double *var_yp, int is_opt);

/* COVARIANCE FUNCTIONS */
void get_krn_se_ard(double *krn, const double *x, const double *xp, unsigned long nx,
		    unsigned long nxp, unsigned long dim, const double *p, int npar);
void get_dkrn_se_ard(double *dK, int k, const double *x, const double *kxx, unsigned long nx,
		     unsigned int dim, const double *p, int np);
void get_krn_rat_quad(double *krn, const double *x, const double *xp, unsigned long nx,
		      unsigned long nxp, unsigned long dim, const double *par, int npar,
		      const double *hpar, int nhpar);

/* TESTS */
double test_get_dkrn_se_ard(unsigned int m, unsigned int dim, unsigned long nx, double eps,
			    int seed);
double test_jac_cost_fun_ard(int m, unsigned int dim, unsigned long nx, double eps, int seed);
double test_gpr_interpolate(unsigned long ns, unsigned long np, int fno, int seed);

/* CROSS VALIDATION */
double get_chi_sq(const double *y, const double *y_pred, const double *covar, unsigned long n);
double get_mhlbs_dist(const double *y, const double *y_pred, const double *covar, unsigned long n);
void get_subsample_cv_holdout(double *ytst, double *xtst, unsigned long ntst, double *ytrn,
			      double *xtrn, unsigned long ntrn, const double *y, const double *x,
			      unsigned long n, unsigned int dim, unsigned long k);
double get_gpr_cv_holdout_batch(unsigned long k, unsigned long ntst, const double *x,
				const double *y, unsigned long n, unsigned int dim, double *hp,
				unsigned long nhp, const enum estimator est);
double get_gpr_cv_holdout(const double *x, const double *y, unsigned long n, unsigned int dim,
			  double *hp, unsigned long nhp, unsigned long ntst, unsigned long nbtch,
			  enum estimator est);
double get_gpr_mean_cv_holdout_batch(unsigned long k, unsigned long ntst, const double *x,
				     const double *y, const double *y_mn, unsigned long n,
				     unsigned int dim, double *hp, unsigned long nhp,
				     enum estimator est);
double get_gpr_mean_cv_holdout(const double *x, const double *y, const double *y_mn,
			       unsigned long n, unsigned int dim, double *hp, unsigned long nhp,
			       unsigned long ntst, unsigned long nbtch, enum estimator est);

/* GPR RK45 ODE */
void gpr_rk45vec_step(double t0, unsigned long m, double *y0, double h,
		      void fun(double *f, double t, double *u1, unsigned long mn, void *param),
		      void *param, double *y, double *e, const double *x, unsigned int dim,
		      double *hparam, unsigned int nhparam, double *cv_step, unsigned long ntst,
		      unsigned long nbtch, enum estimator est, double *work, unsigned long nwork);
void gpr_rk45vec(double t0, double tn, double h, double *y0, unsigned long n,
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
double test_get_gpr_cv_holdout(unsigned long n, unsigned int dim, unsigned long ntst,
			       unsigned long nbtch, enum estimator est, int seed);
