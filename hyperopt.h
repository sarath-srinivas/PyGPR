
typedef void (*fun_covar)(double *n, const double *x, const double *xp, unsigned long nx,
			  unsigned long nxp, unsigned int dim, const double *p, unsigned int npar,
			  void *dat);
typedef void (*fun_covar_jac)(double *dK, unsigned int k, const double *x, const double *kxx,
			      unsigned long nx, unsigned int dim, const double *p, unsigned int np,
			      void *dat);

struct gpr_dat {
	unsigned long int ns;
	unsigned int dim;
	int sgn;
	const double *x;
	const double *y;
	const double *r2;
	fun_covar covar;
	fun_covar_jac covar_jac;
	void *covar_args;
};

void get_hyper_param_ard(
    double *p, unsigned int np, double *x, double *y, unsigned long ns, unsigned int dim,
    void covar(double *krn, const double *x, const double *xp, unsigned long nx, unsigned long nxp,
	       unsigned int dim, const double *p, unsigned int npar, void *dat),
    void covar_jac(double *dK, unsigned int k, const double *x, const double *kxx, unsigned long nx,
		   unsigned int dim, const double *p, unsigned int np, void *dat),
    void *dat);

double get_f(const double *hp, unsigned long nhp, void *data);
void get_f_jac(double *f, double *jac, const double *hp, unsigned long nhp, void *data);

/* TESTS */

double test_jac_cost_fun_ard(unsigned int m, unsigned int dim, unsigned long nx, double eps,
			     int seed);
