
void gpr_interpolate_experts(
    double *yp, double *var_yp, const double *xp, unsigned long np, const double *x,
    const double *y, unsigned long ns, const double *xc, unsigned long nc, unsigned int dim,
    double *hp, unsigned long nhp, int is_opt,
    void covar(double *krn, const double *x, const double *xp, unsigned long nx, unsigned long nxp,
	       unsigned int dim, const double *p, unsigned int npar, void *dat),
    void covar_jac(double *dK, unsigned int k, const double *x, const double *kxx, unsigned long nx,
		   unsigned int dim, const double *p, unsigned int np, void *dat),
    void *dat, unsigned int gate);

void gpr_interpolate_grbcm(
    double *yp, double *var_yp, const double *xp, unsigned long np, const double *xl,
    const double *y, unsigned long ns, const double *xg, const double *yg, unsigned long ng,
    unsigned long nc, unsigned int dim, double *hpl, unsigned long nhpl, double *hpg,
    unsigned long nhg, int is_opt,
    void covar(double *krn, const double *x, const double *xp, unsigned long nx, unsigned long nxp,
	       unsigned int dim, const double *p, unsigned int npar, void *dat),
    void covar_jac(double *dK, unsigned int k, const double *x, const double *kxx, unsigned long nx,
		   unsigned int dim, const double *p, unsigned int np, void *dat),
    void *dat);

void grbcm_experts(double *yp, double *var_yp, unsigned long np, const double *ypg,
		   const double *var_ypg, const double *ypl, const double *var_ypl,
		   unsigned long nl);

void prod_experts(double *yp, double *var_yp, unsigned long np, const double *ypc,
		  const double *var_ypc, unsigned long npc);

void weighted_prod_experts(double *yp, double *var_yp, unsigned long np, const double *ypc,
			   const double *var_ypc, unsigned long npc,
			   void covar(double *krn, const double *x, const double *xp,
				      unsigned long nx, unsigned long nxp, unsigned int dim,
				      const double *p, unsigned int npar, void *dat),
			   const double *xp, unsigned int dim, const double *hp, unsigned long nhp,
			   void *dat);

void bcm_experts(double *yp, double *var_yp, unsigned long np, const double *ypc,
		 const double *var_ypc, unsigned long npc,
		 void covar(double *krn, const double *x, const double *xp, unsigned long nx,
			    unsigned long nxp, unsigned int dim, const double *p, unsigned int npar,
			    void *dat),
		 const double *xp, unsigned int dim, const double *hp, unsigned long nhp,
		 const unsigned long *ind, void *dat);

void rbcm_experts(double *yp, double *var_yp, unsigned long np, const double *ypc,
		  const double *var_ypc, unsigned long npc,
		  void covar(double *krn, const double *x, const double *xp, unsigned long nx,
			     unsigned long nxp, unsigned int dim, const double *p,
			     unsigned int npar, void *dat),
		  const double *xp, unsigned int dim, const double *hp, unsigned long nhp,
		  void *dat);
