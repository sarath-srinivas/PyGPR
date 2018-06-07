extern void dposv_(const unsigned char *UPLO, const int *N, const int *NRHS, double *A, const int *LDA, double *B, const int *LDB, int *info);
extern void dpotrs_(const unsigned char *UPLO, const int *N, const int *NRHS, double *A, const int *LDA, double *B, const int *LDB, int *info);
extern void dgemm_(const unsigned char *transa, const unsigned char *transb, const int *m, const int *n,
		   const int *k, const double *alpha, const double *a, const int *lda, const double *b,
		   const int *ldb, const double *beta, double *c, const int *ldc);
extern void dgemv_(const unsigned char *transa, const int *m, const int *n, const double *alpha,
		   const double *a, const int *lda, const double *x, const int *incx, const double *beta,
		   double *y, const int *incy);
double fit_gpr(double *y, double *x, unsigned long ns, unsigned long dim, double *krn);
int get_krn_se(double *krn, const double *x, const double *xp, unsigned long nx, unsigned long nxp, unsigned long dim, const double *p, int npar);
int get_gpr_weights(double *wt, double *krn_chd, const double *krn, unsigned long ns, unsigned long dim, const double *y);
int gpr_predict(double *yp, const double *wt, const double *krnp, unsigned long np, const unsigned long ns);
int get_var_mat(double *var, double *krnpp, double *krnp, double *krn_chd, unsigned long np, unsigned long ns);
