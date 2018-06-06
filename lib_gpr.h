double fit_gpr(double *y, double *x, unsigned long ns, unsigned long dim, double *krn);
int get_krn_se(double *krn, const double *x, const double *xp, unsigned long ns, unsigned long dim, const double *p, int np);
int get_gpr_weights(double *wt, double *krn_chd, const double *krn, unsigned long ns, unsigned long dim, const double *y);
