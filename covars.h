
/* SQUARED EXPONENTIAL COVARIANCE */

void get_krn_se_ard(double *krn, const double *x, const double *xp, unsigned long nx,
		    unsigned long nxp, unsigned int dim, const double *p, unsigned int npar,
		    void *dat);
void get_dkrn_se_ard(double *dK, unsigned int k, const double *x, const double *kxx,
		     unsigned long nx, unsigned int dim, const double *p, unsigned int np,
		     void *dat);

double test_get_dkrn_se_ard(unsigned int m, unsigned int dim, unsigned long nx, double eps,
			    int seed);

/* EXPONENTIAL COVARIANCE */

void get_krn_modexp_ard(double *krn, const double *x, const double *xp, unsigned long nx,
			unsigned long nxp, unsigned int dim, const double *p, unsigned int npar,
			void *dat);
void get_dkrn_modexp_ard(double *dK, unsigned int k, const double *x, const double *kxx,
			 unsigned long nx, unsigned int dim, const double *p, unsigned int np,
			 void *dat);

/* SYMMERIC SQUARED EXPONENTIAL COVARIANCE */

struct symm_covar_dat {
	double *ax, *axp;
	int sgn;
};

void input_wrap(double *ax, const double *x, unsigned long nx, unsigned int dim);

void get_symm_covar(double *krn, const double *x, const double *xp, unsigned long nx,
		    unsigned long nxp, unsigned int dim, const double *p, unsigned int npar,
		    void *dat);
void get_symm_covar_jac(double *dK, unsigned int m, const double *kxx, const double *x,
			unsigned long nx, unsigned int dim, const double *p, unsigned int npar,
			void *dat);

double test_symm_covar(int sgn, unsigned int dim, unsigned long nx, unsigned long ns, int seed);
double test_symm_covar_jac(int sgn, unsigned int m, unsigned int dim, unsigned long nx, double eps,
			   int seed);
