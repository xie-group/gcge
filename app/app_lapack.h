/**
 *    @file  app_lapack.h
 *   @brief  app of lapack 
 *
 *  lapack的操作接口, 针对稠密矩阵
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/8/13
 *      Revision:  none
 */
#ifndef  _APP_LAPACK_H_
#define  _APP_LAPACK_H_

#include	"ops.h"

typedef struct LAPACKMAT_ {
	double *data; int nrows; int ncols; int ldd;
} LAPACKMAT;
typedef LAPACKMAT LAPACKVEC;

void OPS_LAPACK_Set  (struct OPS_ *ops);

#define dasum FORTRAN_WRAPPER(dasum)
#define daxpy FORTRAN_WRAPPER(daxpy)
#define dcopy FORTRAN_WRAPPER(dcopy)
#define ddot FORTRAN_WRAPPER(ddot)
#define dgemm FORTRAN_WRAPPER(dgemm)
#define dgemv FORTRAN_WRAPPER(dgemv)
#define dlamch FORTRAN_WRAPPER(dlamch)
#define idamax FORTRAN_WRAPPER(idamax)
#define dscal FORTRAN_WRAPPER(dscal)
#define dsymm FORTRAN_WRAPPER(dsymm)
#define dsymv FORTRAN_WRAPPER(dsymv)
#define dgeqp3 FORTRAN_WRAPPER(dgeqp3)
#define dorgqr FORTRAN_WRAPPER(dorgqr)
#define dgerqf FORTRAN_WRAPPER(dgerqf)
#define dorgrq FORTRAN_WRAPPER(dorgrq)
#define dsyev FORTRAN_WRAPPER(dsyev)
#define dsyevx FORTRAN_WRAPPER(dsyevx)

#if !OPS_USE_INTEL_MKL
/* BLAS */
double dasum(int *n, double *dx, int *incx);
int daxpy(int *n, double *da, double *dx, int *incx, double *dy, int *incy);
int dcopy(int *n, double *dx, int *incx, double *dy, int *incy);
double ddot(int *n, double *dx, int *incx, double *dy, int *incy);
int dgemm(char *transa, char *transb, int *m, int *n, int *k,
		double *alpha, double *a, int *lda, 
		               double *b, int *ldb,
		double *beta , double *c, int *ldc);
int dgemv(char *trans, int *m, int *n,
		double *alpha, double *a, int *lda,
		               double *x, int *incx,
		double *beta , double *y, int *incy);
double dlamch(char *cmach);			
int idamax(int  *n, double *dx, int *incx);
int dscal(int *n, double *da, double *dx, int *incx);
int dsymm(char *side, char *uplo, int *m, int *n,
		double *alpha, double *a, int *lda, 
	 	               double *b, int *ldb, 
		double *beta , double *c, int *ldc);
int dsymv(char *uplo, int *n, 
		double *alpha, double *a, int *lda, 
		               double *x, int *incx, 
		double *beta , double *y, int *incy);
/* LAPACK */
/* DGEQP3 computes a QR factorization with column pivoting of 
 * a matrix A:  A*P = Q*R  using Level 3 BLAS 
 * LWORK >= 2*N+( N+1 )*NB, where NB is the optimal blocksize */
int dgeqp3(int *m, int *n, double *a, int *lda, int *jpvt,
	double *tau, double *work, int *lwork, int *info);
/* DORGQR generates an M-by-N real matrix Q with 
 * orthonormal columns 
 * K is the number of elementary reflectors whose product 
 * defines the matrix Q. N >= K >= 0.
 * LWORK >= N*NB, where NB is the optimal blocksize */
int dorgqr(int *m, int *n, int *k, double *a, int *lda,
	double *tau, double *work, int *lwork, int *info);
/* The length of the array WORK.  LWORK >= 1, when N <= 1;
 * otherwise 8*N.
 * For optimal efficiency, LWORK >= (NB+3)*N,
 * where NB is the max of the blocksize for DSYTRD and DORMTR
 * returned by ILAENV. */
/* RQ factorization */
int dgerqf(int *m, int *n, double *a, int *lda, 
	double *tau, double *work, int *lwork, int *info);
int dorgrq(int *m, int *n, int *k, double *a, int *lda, 
	double *tau, double *work, int *lwork, int *info);
int dsyev(char *jobz, char *uplo, int *n, 
	double *a, int *lda, double *w, 
	double *work, int *lwork, int *info);
int dsyevx(char *jobz, char *range, char *uplo, int *n, 
	double *a, int *lda, double *vl, double *vu, int *il, int *iu, 
	double *abstol, int *m, double *w, double *z, int *ldz, 
	double *work, int *lwork, int *iwork, int *ifail, int *info);
#endif
	
#endif  /* -- #ifndef _APP_LAPACK_H_ -- */
