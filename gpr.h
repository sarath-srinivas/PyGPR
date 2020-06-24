
void get_gpr_weights(double *wt, double *krn_chd, const double *krn, unsigned long ns,
		     unsigned long dim, const double *y);
void gpr_predict(double *yp, const double *wt, const double *krnp, unsigned long np,
		 const unsigned long ns);
void get_var_mat(double *var, double *krnpp, double *krnp, double *krn, unsigned long np,
		 unsigned long ns);
void get_var_mat_chd(double *var, const double *krnpp, const double *krnp, const double *krn_chd,
		     unsigned long np, unsigned long ns);
double get_log_likelihood(const double *wt, const double *y, unsigned long ns,
			  const double *krn_chd, double *ret);
void sample_gp(double *y, const double *mn, const double *kxx, unsigned long ns, int seed);

/* WRAPPERS */

void gpr_interpolate(double *xp, double *yp, unsigned long np, double *x, double *y,
		     unsigned long ns, unsigned int dim, double *p, unsigned int npar,
		     double *var_yp, int is_opt);
void gpr_interpolate_symm(double *xp, double *axp, double *yp, unsigned long np, double *x,
			  double *ax, double *y, unsigned long ns, unsigned int dim, int sgn,
			  double *p, unsigned int npar, double *var_yp, int is_opt);
void gpr_interpolate_mean(double *xp, double *yp, double *yp_mn, unsigned long np, double *x,
			  double *y, double *y_mn, unsigned long ns, unsigned int dim, double *p,
			  unsigned int npar, double *var_yp, int is_opt);
void gpr_interpolate_wrap(
    const double *xp, double *yp, unsigned long np, const double *x, const double *y,
    unsigned long ns, unsigned int dim, double *p, unsigned int npar, double *var_yp, int is_opt,
    void covar(double *krn, const double *x, const double *xp, unsigned long nx, unsigned long nxp,
	       unsigned int dim, const double *p, unsigned int npar, void *dat),
    void covar_jac(double *dK, unsigned int k, const double *x, const double *kxx, unsigned long nx,
		   unsigned int dim, const double *p, unsigned int np, void *dat),
    void *dat);
