
void gpr_interpolate_experts(double *yp, double *var_yp, const double *xp, unsigned long np,
			     const double *x, const double *y, unsigned long ns, unsigned long nc,
			     unsigned int dim, double *hp, unsigned long nhp, int is_opt,
			     void covar(double *krn, const double *x, const double *xp,
					unsigned long nx, unsigned long nxp, unsigned int dim,
					const double *p, unsigned int npar, void *dat),
			     void covar_jac(double *dK, unsigned int k, const double *x,
					    const double *kxx, unsigned long nx, unsigned int dim,
					    const double *p, unsigned int np, void *dat),
			     void *dat, unsigned int gate);

void prod_experts(double *yp, double *var_yp, unsigned long np, const double *ypc,
		  const double *var_ypc, unsigned long npc);

void weighted_prod_experts(double *yp, double *var_yp, unsigned long np, const double *ypc,
			   const double *var_ypc, unsigned long npc,
			   void covar(double *krn, const double *x, const double *xp,
				      unsigned long nx, unsigned long nxp, unsigned int dim,
				      const double *p, unsigned int npar, void *dat),
			   const double *xp, unsigned int dim, const double *hp, unsigned long nhp,
			   void *dat);

void rbcm_experts(double *yp, double *var_yp, unsigned long np, const double *ypc,
		  const double *var_ypc, unsigned long npc,
		  void covar(double *krn, const double *x, const double *xp, unsigned long nx,
			     unsigned long nxp, unsigned int dim, const double *p,
			     unsigned int npar, void *dat),
		  const double *xp, unsigned int dim, const double *hp, unsigned long nhp,
		  void *dat);
