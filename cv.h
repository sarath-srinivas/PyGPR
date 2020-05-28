
enum estimator { CHI_SQ, MAHALANOBIS };

double get_rel_rmse(const double *y, const double *y_pred, unsigned long n);
double get_chi_sq(const double *y, const double *y_pred, const double *covar, unsigned long n);
double get_mhlbs_dist(const double *y, const double *y_pred, const double *covar, unsigned long n);
void get_subsample_cv_holdout(double *ytst, double *xtst, unsigned long ntst, double *ytrn,
			      double *xtrn, unsigned long ntrn, const double *y, const double *x,
			      unsigned long n, unsigned int dim, unsigned long k);
void get_gpr_cv_holdout_batch(double *cv, unsigned long k, unsigned long ntst, const double *x,
			      const double *y, unsigned long n, unsigned int dim, double *hp,
			      unsigned long nhp, const enum estimator est);
void get_gpr_cv_holdout(double *cv_btch, const double *x, const double *y, unsigned long n,
			unsigned int dim, double *hp, unsigned long nhp, unsigned long ntst,
			unsigned long nbtch, enum estimator est);
double get_gpr_mean_cv_holdout_batch(unsigned long k, unsigned long ntst, const double *x,
				     const double *y, const double *y_mn, unsigned long n,
				     unsigned int dim, double *hp, unsigned long nhp,
				     enum estimator est);
double get_gpr_mean_cv_holdout(const double *x, const double *y, const double *y_mn,
			       unsigned long n, unsigned int dim, double *hp, unsigned long nhp,
			       unsigned long ntst, unsigned long nbtch, enum estimator est);
