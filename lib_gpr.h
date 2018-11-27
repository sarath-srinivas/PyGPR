#define NP_SE (1)
#define NP_SE_ARD (6)

struct gpr_dat {
	unsigned long int ns;
	int dim;
	double *x;
	double *y;
	double *r2;
};

extern void dposv_(const unsigned char *UPLO, const int *N, const int *NRHS, double *A, const int *LDA, double *B, const int *LDB, int *info);
extern void dpotrs_(const unsigned char *UPLO, const int *N, const int *NRHS, double *A, const int *LDA, double *B, const int *LDB, int *info);
extern void dtrsm_(const unsigned char *SIDE, const unsigned char *UPLO, const unsigned char *TRA,
		   const unsigned char *DIAG, const int *M, const int *N, const double *ALPHA,
		   const double *A, const int *LDA, double *B, const int *LDB);
extern void dgemm_(const unsigned char *transa, const unsigned char *transb, const int *m, const int *n,
		   const int *k, const double *alpha, const double *a, const int *lda, const double *b,
		   const int *ldb, const double *beta, double *c, const int *ldc);
extern void dgemv_(const unsigned char *transa, const int *m, const int *n, const double *alpha,
		   const double *a, const int *lda, const double *x, const int *incx, const double *beta,
		   double *y, const int *incy);
extern double ddot_(const int *N, const double *X, const int *incx, const double *Y, const int *incy);
extern double dsyrk_(const unsigned char *UPLO, const unsigned char *TRA, const int *N, const int *K,
		     const double *ALPHA, const double *A, const int *LDA, const double *BETA, double *C,
		     const int *LDC);
extern void dsymv_(const unsigned char *uplo, const int *n, const double *alpha,
		   const double *a, const int *lda, const double *x, const int *incx, const double *beta,
		   double *y, const int *incy);

/*
double fit_gpr(double *y, double *x, unsigned long ns, unsigned long dim, double *krn);
*/
int get_krn_se(double *krn, const double *x, const double *xp, unsigned long nx, unsigned long nxp, unsigned long dim, const double *p, int npar);
int get_krn_se_ard(double *krn, const double *x, const double *xp, unsigned long nx, unsigned long nxp, unsigned long dim, const double *p, int npar);
int get_gpr_weights(double *wt, double *krn_chd, const double *krn, unsigned long ns, unsigned long dim, const double *y);
int gpr_predict(double *yp, const double *wt, const double *krnp, unsigned long np, const unsigned long ns);
int get_var_mat(double *var, double *krnpp, double *krnp, double *krn_chd, unsigned long np, unsigned long ns);
double get_log_likelihood(const double *wt, const double *y, unsigned long ns, const double *krn_chd, double *ret);
int get_var_mat_chd(double *var, double *krnpp, double *krnp, double *krn_chd, unsigned long np, unsigned long ns);
int get_hyper_param(double *p, int np, double *x, double *y, unsigned long ns, int dim);
int get_hyper_param_ard(double *p, int np, double *x, double *y, unsigned long ns, int dim);
void gpr_interpolate(double *y, double *x, unsigned long ns, unsigned int dim, double *xp, unsigned long np, double *yp, double *p, unsigned int npar, double *var_yp);
/* COVARIANCE FUNCTIONS */
int get_krn_rat_quad(double *krn, const double *x, const double *xp, unsigned long nx, unsigned long nxp, unsigned long dim, const double *par, int npar, const double *hpar, int nhpar);
/* TESTS */
double test_gpr_interpolate(unsigned long ns, unsigned long np, int fno, int seed);


