/**
 *    @file  app_lapack.c
 *   @brief  app of lapack  
 *
 *  单向量与多向量结构是统一的
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *           ZJ Wang, for OpenMP
 *
 *       Created:  2020/8/13
 *      Revision:  none
 */

#include	<stdio.h>
#include	<stdlib.h>
#include	<assert.h>
#include	<math.h>
#include	<memory.h>
 
#include	"app_lapack.h"

/* matC = alpha*matQ^{\top}*matA*matP + beta*matC 
 * dbl_ws: nrowsA*ncolsC */
static void DenseMatQtAP(char ntluA, char nsdC,
		int nrowsA, int ncolsA, /* matA 的行列数 */
		int nrowsC, int ncolsC, /* matC 的行列数 */
		double alpha, double *matQ, int ldQ, 
		              double *matA, int ldA,
		              double *matP, int ldP,
		double beta , double *matC, int ldC,
		double *dbl_ws)
{
	if (nrowsC==0||ncolsC==0) {
		return; 
	}
	if (nrowsA==0&&ncolsA==0) {
		int col;
		if (nsdC == 'D') {
			assert(nrowsC == ncolsC);
			if (ldC == 1) {
				memset(matC,0,ncolsC*sizeof(double));
			}
			else {
				for (col = 0; col < ncolsC; ++col) {
					matC[ldC*col] = 0.0;
				}	
			}			
		}
		else {
			if (ldC==nrowsC) {
				memset(matC,0,nrowsC*ncolsC*sizeof(double));
			}
			else {
				for (col = 0; col < ncolsC; ++col) {
					memset(matC,0,nrowsC*sizeof(double));
					matC += ldC;
				}		
			}
		}
		return;	
	}	
	
	char charN = 'N', charT = 'T';
	if (matA == NULL) {
		int idx, inc;
		assert(nrowsA == ncolsA);
		if (nsdC == 'D') {
			assert(nrowsC == ncolsC);
			inc = 1;			
			if (beta==1.0) {
				if (alpha!=0.0) {
#if OPS_USE_OMP
					#pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS)
#endif
					for (idx = 0; idx < ncolsC; ++idx) {
						matC[ldC*idx] += alpha*ddot(&nrowsA,
							matQ+ldQ*idx,&inc,matP+ldP*idx,&inc);
					}
				}							
			}
			else if (beta==0.0) {
				if (alpha!=0.0) {
#if OPS_USE_OMP
					#pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS)
#endif
					for (idx = 0; idx < ncolsC; ++idx) {
						matC[ldC*idx] = alpha*ddot(&nrowsA,
							matQ+ldQ*idx,&inc,matP+ldP*idx,&inc);
					}
				}
				else {
					if (ldC == 1) {
						memset(matC,0,ncolsC*sizeof(double));
					}
					else {
#if OPS_USE_OMP
						#pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS)
#endif
						for (idx = 0; idx < ncolsC; ++idx) {
							matC[ldC*idx] = 0.0;
						}	
					}	
				}				
			}
			else {
#if OPS_USE_OMP
				#pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS)
#endif
				for (idx = 0; idx < ncolsC; ++idx) {
					matC[ldC*idx] = alpha*ddot(&nrowsA,
						matQ+ldQ*idx,&inc,matP+ldP*idx,&inc)
						+beta*matC[ldC*idx];
				}
			}
		}
		else if (nsdC == 'S') {
			assert(nrowsC == ncolsC);
			inc = 1;
#if OPS_USE_OMP
		        #pragma omp parallel for schedule(static) num_threads(OMP_NUM_THREADS)
#endif
			for (idx = 0; idx < ncolsC; ++idx) {
				int nrowsC;
				nrowsC = ncolsC - idx;
				dgemv(&charT,&nrowsA,&nrowsC,
						&alpha,matQ+ldQ*idx    ,&ldQ,
						       matP+ldP*idx    ,&inc, 
						&beta ,matC+ldC*idx+idx,&inc);
				--nrowsC;
				dcopy(&nrowsC,
						matC+ldC*idx+(idx+1),&inc,  /* copy x */
						matC+ldC*(idx+1)+idx,&ldC); /* to   y */
			}
		}
		else {
#if OPS_USE_OMP
#if 0
			#pragma omp parallel num_threads(OMP_NUM_THREADS)
			{
				int id, length, offset;
				id     = omp_get_thread_num();
				length = nrowsC/OMP_NUM_THREADS;
				offset = length*id;
				if (id < nrowsC%OMP_NUM_THREADS) {
					++length; offset += id;
				}
				else {
					offset += nrowsC%OMP_NUM_THREADS;
				} 
				dgemm(&charT,&charN,&length,&ncolsC,&nrowsA,
						&alpha,matQ+offset*ldQ,&ldQ,  /* A */
						       matP           ,&ldP,  /* B */
						&beta ,matC+offset    ,&ldC); /* C */
			}
#else
			#pragma omp parallel num_threads(OMP_NUM_THREADS)
			{
				int id, length, offset;
				id     = omp_get_thread_num();
				length = ncolsC/OMP_NUM_THREADS;
				offset = length*id;
				if (id < ncolsC%OMP_NUM_THREADS) {
					++length; offset += id;
				}
				else {
					offset += ncolsC%OMP_NUM_THREADS;
				} 
				dgemm(&charT,&charN,&nrowsC,&length,&nrowsA,
						&alpha,matQ           ,&ldQ,  /* A */
						       matP+offset*ldP,&ldP,  /* B */
						&beta ,matC+offset*ldC,&ldC); /* C */
			}

#endif

#else
			dgemm(&charT,&charN,&nrowsC,&ncolsC,&nrowsA,
					&alpha,matQ,&ldQ,  /* A */
					       matP,&ldP,  /* B */
					&beta ,matC,&ldC); /* C */
#endif
		}
	}
	else {
		char side; int nrowsW, ncolsW, ldW; double *matW, zero, one;
		nrowsW = nrowsA; ncolsW = ncolsC; ldW = nrowsW;
		matW   = dbl_ws; zero   = 0.0   ; one = 1.0   ;
		if (ntluA == 'L' || ntluA == 'U') {
			side = 'L'; /* left or right */
			/* matW = matA*matP */
			dsymm(&side,&ntluA,&nrowsW,&ncolsW,
					&one,matA,&ldA,matP,&ldP,&zero,matW,&ldW);
		}
		else {
			/* matW = matA*matP */
#if OPS_USE_OMP
			#pragma omp parallel num_threads(OMP_NUM_THREADS)
			{
				int id, length, offset;
				id     = omp_get_thread_num();
				length = ncolsW/OMP_NUM_THREADS;
				offset = length*id;
				if (id < ncolsW%OMP_NUM_THREADS) {
					++length; offset += id;
				}
				else {
					offset += ncolsW%OMP_NUM_THREADS;
				} 
				dgemm(&ntluA,&charN,&nrowsW,&length,&ncolsA,
						&one, matA           ,&ldA,
						      matP+offset*ldP,&ldP,
						&zero,matW+offset*ldW,&ldW);
			}
#else
			dgemm(&ntluA,&charN,&nrowsW,&ncolsW,&ncolsA,
					&one, matA,&ldA,
					      matP,&ldP,
					&zero,matW,&ldW);
#endif
		}
		/* matC = alpha*matQ^{\top}*matW + beta*matC */
		DenseMatQtAP(ntluA,nsdC,nrowsA,nrowsA,nrowsC,ncolsC,
				alpha,matQ,ldQ,NULL,ldA,matW,ldW,
				beta ,matC,ldC,dbl_ws);
	}
	return;
}

/* multi-vec */
static void MultiVecCreateByVec (LAPACKVEC **des_vec, int num_vec, LAPACKVEC *src_vec, struct OPS_ *ops)
{
	(*des_vec)        = malloc(sizeof(LAPACKVEC));
	(*des_vec)->nrows = src_vec->nrows;
	(*des_vec)->ncols = num_vec       ;
	(*des_vec)->ldd   = src_vec->ldd  ;
	(*des_vec)->data  = malloc(((*des_vec)->ldd)*((*des_vec)->ncols)*sizeof(double));
	memset((*des_vec)->data,0,((*des_vec)->ldd)*((*des_vec)->ncols)*sizeof(double));
	return;
}
static void MultiVecCreateByMat (LAPACKVEC **des_vec, int num_vec, LAPACKMAT *src_mat, struct OPS_ *ops)
{
	(*des_vec)        = malloc(sizeof(LAPACKVEC));
	(*des_vec)->nrows = src_mat->ncols   ; 
	(*des_vec)->ncols = num_vec          ;
	(*des_vec)->ldd   = (*des_vec)->nrows;
	(*des_vec)->data  = malloc(((*des_vec)->ldd)*((*des_vec)->ncols)*sizeof(double));
	memset((*des_vec)->data,0,((*des_vec)->ldd)*((*des_vec)->ncols)*sizeof(double));
	return;
}
static void MultiVecCreateByMultiVec (LAPACKVEC **des_vec, int num_vec, LAPACKVEC *src_vec, struct OPS_ *ops)
{
	(*des_vec)        = malloc(sizeof(LAPACKVEC));
	(*des_vec)->nrows = src_vec->nrows;
	(*des_vec)->ncols = num_vec       ;
	(*des_vec)->ldd   = src_vec->ldd  ;
	(*des_vec)->data  = malloc(((*des_vec)->ldd)*((*des_vec)->ncols)*sizeof(double));
	memset((*des_vec)->data,0,((*des_vec)->ldd)*((*des_vec)->ncols)*sizeof(double));
	return;
}
static void MultiVecDestroy (LAPACKVEC **des_vec, int num_vec, struct OPS_ *ops)
{
	(*des_vec)->nrows = 0; 
	(*des_vec)->ncols = 0;
	(*des_vec)->ldd   = 0;
	free((*des_vec)->data);
	free(*des_vec); *des_vec = NULL;
	return;
}
static void GetVecFromMultiVec (LAPACKVEC *multi_vec, int col, LAPACKVEC **vec, struct OPS_ *ops)
{
	(*vec)        = malloc(sizeof(LAPACKVEC));
	(*vec)->nrows = multi_vec->nrows;
	(*vec)->ncols = 1               ;
	(*vec)->ldd   = multi_vec->ldd  ;
	(*vec)->data  = multi_vec->data+multi_vec->ldd*col;
	return;
}
static void RestoreVecForMultiVec (LAPACKVEC *multi_vec, int col, LAPACKVEC **vec, struct OPS_ *ops)
{
	(*vec)->nrows = 0;
	(*vec)->ncols = 0;
	(*vec)->ldd   = 0;
	(*vec)->data  = NULL;
	free(*vec); *vec = NULL;
	return;
}
static void MultiVecView (LAPACKVEC *x, int start, int end, struct OPS_ *ops)
{
	int row, col; double *destin; 
	for (row = 0; row < x->nrows; ++row) {
		for (col = start; col < end; ++col) {
			destin = x->data+(x->ldd)*col+row;
			ops->Printf("%6.4e\t", *destin);
		}
		ops->Printf("\n");
	}
	return;
}
static void MultiVecLocalInnerProd (char nsdIP, 
		LAPACKVEC *x, LAPACKVEC *y, int is_vec, int *start, int *end, 
		double *inner_prod, int ldIP, struct OPS_ *ops)
{
	int nrows = end[0]-start[0], ncols = end[1]-start[1];
	if (nrows>0 && ncols>0) {
		DenseMatQtAP('S',nsdIP,x->nrows,y->nrows,nrows,ncols,
			1.0,x->data+x->ldd*start[0],(x->ldd), /* Q */
			    NULL                   ,0,        /* A */
			    y->data+y->ldd*start[1],(y->ldd), /* P */
			0.0,inner_prod             ,ldIP,
			NULL);		
	}	
	return;
}
static void MultiVecInnerProd (char nsdIP, 
		LAPACKVEC *x, LAPACKVEC *y, int is_vec, int *start, int *end, 
		double *inner_prod, int ldIP, struct OPS_ *ops)
{
	MultiVecLocalInnerProd(nsdIP,x,y,is_vec,
			start,end,inner_prod,ldIP, ops);
	return;
}
static void MultiVecSetRandomValue (LAPACKVEC *x, int start, int end, struct OPS_ *ops)
{
	int row, col; double *destin;
	for (col = start; col < end; ++col) {
		destin = x->data+x->ldd*col;
		for (row = 0; row < x->nrows; ++row) {
			*destin = ((double)rand())/((double)RAND_MAX+1);
			++destin;
		}
	}
	return;
}
static void MultiVecAxpby (double alpha, LAPACKVEC *x, 
		double beta, LAPACKVEC *y, int *start, int *end, struct OPS_ *ops)
{
	assert(end[0]-start[0]==end[1]-start[1]);
	int length, ncols = end[1]-start[1]; 
	double *source, *destin; int inc = 1, col;
	if (ncols==0) {
		return;
	}
	if (y->nrows == 0) {
		return;
	}	
	 
	if (y->nrows==y->ldd) {
		length = y->nrows*ncols;
		destin = y->data+y->ldd*(start[1]);
		if (beta==0.0) {
		   memset(destin,0,length*sizeof(double));
		}
		else {
		   if (beta!=1.0) {
		      dscal(&length,&beta,destin,&inc);
		   }
		}
		if ( x!=NULL ) {
			assert(x->nrows==y->nrows);
			if (x->nrows==x->ldd) {
				length = y->nrows*ncols;
				source = x->data+x->ldd*(start[0]);
				destin = y->data+y->ldd*(start[1]);
				daxpy(&length,&alpha,source,&inc,destin,&inc);
			}
			else {
				length = y->nrows;
				for (col = 0; col < ncols; ++col) {
					source = x->data+x->ldd*(start[0]+col);
					destin = y->data+y->ldd*(start[1]+col);
					daxpy(&length,&alpha,source,&inc,destin,&inc);
				}				
			}
		}
	}
	else {
		length = y->nrows;
		for (col = 0; col < ncols; ++col) {
			destin = y->data+y->ldd*(start[1]+col);
			if (beta==0.0) {
			   memset(destin,0,length*sizeof(double));
			}
			else {
			   if (beta!=1.0) {
			      dscal(&length,&beta,destin,&inc);
			   }
			}
			if ( x!=NULL ) {
				source = x->data+x->ldd*(start[0]+col);
				daxpy(&length,&alpha,source,&inc,destin,&inc);
			}		
		}	
	}	
	return;
}
static void MatDotMultiVec (LAPACKMAT *mat, LAPACKVEC *x, 
		LAPACKVEC *y, int *start, int *end, struct OPS_ *ops)
{
	assert(end[0]-start[0]==end[1]-start[1]);
	assert(y->nrows==y->ldd);
	assert(x->nrows==x->ldd);
	char charN = 'N'; double alpha = 1.0, beta = 0.0;
	int  ncols = end[1]-start[1];
	if (ncols==0) return;
	if (mat==NULL) {
		int incx = 1, incy = 1;
		ncols = y->nrows*(end[1]-start[1]);
		dcopy(&ncols,x->data+(x->ldd)*start[0],&incx,
				y->data+(y->ldd)*start[1],&incy);
	} 
	else {
#if OPS_USE_OMP
		#pragma omp parallel num_threads(OMP_NUM_THREADS)
		{
			int id, length, offset;
			id     = omp_get_thread_num();
			length = ncols/OMP_NUM_THREADS;
			offset = length*id;
			if (id < ncols%OMP_NUM_THREADS) {
				++length; offset += id;
			}
			else {
				offset += ncols%OMP_NUM_THREADS;
			} 
			dgemm(&charN,&charN,&y->nrows,&length,&x->nrows,
					&alpha,mat->data                       ,&mat->ldd,/* A */
					       x->data+x->ldd*(start[0]+offset),&x->ldd,  /* B */
					&beta ,y->data+y->ldd*(start[1]+offset),&y->ldd); /* C */
		}
#else
		dgemm(&charN,&charN,&y->nrows,&ncols,&x->nrows,
				&alpha,mat->data              ,&mat->ldd,/* A */
				       x->data+x->ldd*start[0],&x->ldd,  /* B */
				&beta ,y->data+y->ldd*start[1],&y->ldd); /* C */
#endif
	}
	return;
}
static void MatTransDotMultiVec (LAPACKMAT *mat, LAPACKVEC *x, 
		LAPACKVEC *y, int *start, int *end, struct OPS_ *ops)
{
	assert(end[0]-start[0]==end[1]-start[1]);
	assert(y->nrows==y->ldd);
	assert(x->nrows==x->ldd);
	char charN = 'N'; double alpha = 1.0, beta = 0.0;
	int  ncols = end[1]-start[1];

#if 1
	DenseMatQtAP(charN,charN,x->nrows,x->nrows,y->nrows,ncols,
			alpha,mat->data                ,mat->ldd,
			      NULL                     ,0,
			      x->data+(x->ldd)*start[0],x->ldd,
			beta ,y->data+(y->ldd)*start[1],y->ldd,
			NULL);
#else
	dgemm(&charT,&charN,&(y->nrows),&ncols,&x->nrows,
			&alpha,mat->data                ,&mat->ldd,/* A */
			       x->data+(x->ldd)*start[0],&x->ldd,  /* B */
			&beta ,y->data+(y->ldd)*start[1],&y->ldd); /* C */
#endif
	return;
}
static void MultiVecLinearComb (
		LAPACKVEC *x , LAPACKVEC *y, int is_vec, 
		int    *start, int  *end, 
		double *coef , int  ldc , 
		double *beta , int  incb, struct OPS_ *ops)
{
	int nrows, ncols, col, inc, length; char charN = 'N';
	double one = 1.0, gamma, *destin; 
	/* coef 的行数和列数 */
	nrows = end[0]-start[0]; ncols = end[1]-start[1];
	if (nrows==0||ncols==0) {
		return;		
	}
	if (y->nrows == 0) {
		return;
	}
	
	if (beta == NULL) {
		gamma = 0.0;
	}
	else {
		gamma = 1.0; inc = 1;
		if (incb==0) {
			if ((*beta)!=1.0) {
				if (y->ldd==y->nrows) {
					destin = y->data+y->ldd*(start[1]);
					length = (y->nrows)*ncols;
					dscal(&length,beta,destin,&inc);
				}
				else {
					for (col = 0; col < ncols; ++col) {
						destin = y->data+y->ldd*(start[1]+col);
						dscal(&(y->nrows),beta,destin,&inc);
					}
				}
			}
		}
		else {
			for (col = 0; col < ncols; ++col) {
				destin = y->data+y->ldd*(start[1]+col);
				dscal(&(y->nrows),beta+incb*col,destin,&inc);
			}		
		}
	}
	if (x!=NULL && coef!=NULL) {
#if OPS_USE_OMP
		#pragma omp parallel num_threads(OMP_NUM_THREADS)
		{
			int id, length, offset;
			id     = omp_get_thread_num();
			length = ncols/OMP_NUM_THREADS;
			offset = length*id;
			if (id < ncols%OMP_NUM_THREADS) {
				++length; offset += id;
			}
			else {
				offset += ncols%OMP_NUM_THREADS;
			} 
			dgemm(&charN,&charN,&y->nrows,&length,&nrows,
					&one  ,x->data+x->ldd*start[0]         ,&(x->ldd), /* A */
					       coef   +offset*ldc              ,&ldc,      /* B */
					&gamma,y->data+y->ldd*(start[1]+offset),&(y->ldd));/* C */
		}
#else
		dgemm(&charN,&charN,&y->nrows,&ncols,&nrows,
				&one  ,x->data+x->ldd*start[0],&(x->ldd), /* A */
				       coef                   ,&ldc,      /* B */
				&gamma,y->data+y->ldd*start[1],&(y->ldd));/* C */
#endif
	}
	return;
}
static void MultiVecQtAP (char ntsA, char ntsdQAP, 
		LAPACKVEC *mvQ, LAPACKMAT *matA, LAPACKVEC *mvP, int is_vec, 
		int *start, int *end, double *qAp, int ldQAP,
		LAPACKVEC *vec_ws, struct OPS_ *ops)
{
	double alpha = 1.0, beta = 0.0, *matA_data; int matA_ldd; 
	int nrows = end[0]-start[0], ncols = end[1]-start[1];
	if (nrows==0||ncols==0) return;
	if (ntsA == 'S') ntsA = 'L';
	if (matA == NULL) {		
		matA_data = NULL; matA_ldd = 0;
	}
	else {
		matA_data = matA->data; matA_ldd = matA->ldd; 
	}
	if (ntsdQAP=='T') {
		double *dbl_ws = malloc(nrows*ncols*sizeof(double));
		DenseMatQtAP(ntsA,'N',
				mvQ->nrows,mvP->nrows, /* matA 的行列数 */
				nrows     ,ncols,      /* matC 的行列数 */
				alpha,mvQ->data+mvQ->ldd*start[0],mvQ->ldd, 
				      matA_data                  ,matA_ldd,
				      mvP->data+mvP->ldd*start[1],mvP->ldd,
				beta ,dbl_ws                     ,nrows,
				vec_ws->data);
		double *source, *destin; int incx, incy, row;
		source = dbl_ws; incx  = nrows;   
		destin = qAp   ; incy  = 1;
		for (row = 0; row < nrows; ++row) {
			dcopy(&ncols,source,&incx,destin,&incy);
			source += 1; destin += ldQAP;
		}
		free(dbl_ws);
	}
	else {
		DenseMatQtAP(ntsA,ntsdQAP,
				mvQ->nrows,mvP->nrows, /* matA 的行列数 */
				nrows     ,ncols,      /* matC 的行列数 */
				alpha,mvQ->data+mvQ->ldd*start[0],mvQ->ldd, 
				      matA_data                  ,matA_ldd,
				      mvP->data+mvP->ldd*start[1],mvP->ldd,
				beta ,qAp                        ,ldQAP,
				vec_ws->data);
	}
	return;
}

/* vec */
static void VecCreateByVec (LAPACKVEC **des_vec, LAPACKVEC *src_vec, struct OPS_ *ops)
{
	MultiVecCreateByVec(des_vec,1,src_vec, ops);
	return;
}
static void VecCreateByMat (LAPACKVEC **des_vec, LAPACKMAT *src_mat, struct OPS_ *ops)
{
	MultiVecCreateByMat(des_vec,1,src_mat, ops);
	return;
}
static void VecDestroy (LAPACKVEC **des_vec, struct OPS_ *ops)
{
	MultiVecDestroy(des_vec,1, ops);
	return;
}
static void VecView (LAPACKVEC *x, struct OPS_ *ops)
{
	MultiVecView(x,0,1,ops);
	return;
}
static void VecInnerProd (LAPACKVEC *x, LAPACKVEC *y, double *inner_prod, struct OPS_ *ops)
{
	int start[2] = {0,0}, end[2] = {1,1};
	MultiVecInnerProd('S',x,y,0,start,end,inner_prod,1,ops);
	return;
}
static void VecLocalInnerProd (LAPACKVEC *x, LAPACKVEC *y, double *inner_prod, struct OPS_ *ops)
{
	int start[2] = {0,0}, end[2] = {1,1};
	MultiVecLocalInnerProd('S',x,y,0,start,end,inner_prod,1,ops);
	return;
}
static void VecSetRandomValue (LAPACKVEC *x, struct OPS_ *ops)
{
	MultiVecSetRandomValue(x,0,1,ops);
	return;
}
static void VecAxpby (double alpha, LAPACKVEC *x, double beta, LAPACKVEC *y, struct OPS_ *ops)
{
	int start[2] = {0,0}, end[2] = {1,1};
	MultiVecAxpby(alpha,x,beta,y,start,end, ops);
	return;
}
static void MatDotVec (LAPACKMAT *mat, LAPACKVEC *x, LAPACKVEC *y, struct OPS_ *ops)
{
	int start[2] = {0,0}, end[2] = {1,1};
	MatDotMultiVec(mat,x,y,start,end, ops);
	return;
}
static void MatTransDotVec (LAPACKMAT *mat, LAPACKVEC *x, LAPACKVEC *y, struct OPS_ *ops)
{
	int start[2] = {0,0}, end[2] = {1,1};
	MatTransDotMultiVec(mat,x,y,start,end, ops);
	return;
}
static void MatView (LAPACKMAT *mat, struct OPS_ *ops)
{
	int row, col; double *destin; 
	for (row = 0; row < mat->nrows; ++row) {
		for (col = 0; col < mat->ncols; ++col) {
			destin = mat->data+(mat->ldd)*col+row;
			ops->Printf("%6.4e\t", *destin);
		}
		ops->Printf("\n");
	}
	return;
}
/* length: length of dbl_ws >= 2*N+(N+1)*NB + min(M,N),
 * where NB is the optimal blocksize and N = end[0]-start
 * length of int_ws is N */
static void DenseMatOrth(double *mat, int nrows, int ldm,
		int start, int *end, double orth_zero_tol,
		double *dbl_ws, int length, int *int_ws)
{
	/* 去掉x1中的x0部分 */
	if (start > 0) {
		double *beta, *coef; int start_x01[2], end_x01[2], idx;
		int length, inc;
		LAPACKVEC x0, x1; 
		x0.nrows = nrows; x0.ncols = start        ; x0.ldd = ldm;
		x1.nrows = nrows; x1.ncols = *end-start   ; x1.ldd = ldm; 
		x0.data  = mat  ; x1.data  = mat+ldm*start;
		start_x01[0] = 0; end_x01[0] = x0.ncols   ; 
		start_x01[1] = 0; end_x01[1] = x1.ncols   ; 
		beta = dbl_ws   ; coef = dbl_ws+1;
		for (idx = 0; idx < 2; ++idx) {
			MultiVecInnerProd('N',&x0,&x1,0,start_x01,end_x01,
					coef,x0.ncols, NULL);
			*beta = -1.0; inc = 1;
			length = x0.ncols*x1.ncols;
			dscal(&length,beta,coef,&inc);
			*beta = 1.0;
			MultiVecLinearComb(&x0,&x1,0,start_x01,end_x01,
					coef,end_x01[0]-start_x01[0],beta,0, NULL);	
		}		
	}
	/* 列选主元的QR分解 */
	int m, n, k, lda, *jpvt, lwork, info, col; 
	double *a, *tau, *work;
	m    = nrows ; n     = *end-start; a   = mat+ldm*start;
	jpvt = int_ws; tau   = dbl_ws    ; lda = ldm;
	work = tau+n ; lwork = length-n  ;
	for (col = 0; col < n; ++col) {
		jpvt[col] = 0;
	}
	dgeqp3(&m,&n,a,&lda,jpvt,tau,work,&lwork,&info);
	/* 得到a的秩, 并更新n, a的对角线存储了r_ii */
	for (col = n-1; col >= 0; --col) {
		if (fabs(a[lda*col+col]) > orth_zero_tol) break;
	}
	n = col+1; k = m<n?m:n;
	/* 生成Q, 即完成对x1的正交化 */
	dorgqr(&m,&n,&k,a,&lda,tau,work,&lwork,&info);
	/* 标记得到的正交向量组的末尾 */
	*end = start+n;
	return;
}

/* Encapsulation */
static void LAPACK_MatView (void *mat, struct OPS_ *ops)
{
	MatView ((LAPACKMAT *)mat, ops);
	return;
}
/* vec */
static void LAPACK_VecCreateByVec (void **des_vec, void *src_vec, struct OPS_ *ops)
{
	VecCreateByVec ((LAPACKVEC **)des_vec, (LAPACKVEC *)src_vec, ops);
	return;
}
static void LAPACK_VecCreateByMat (void **des_vec, void *src_mat, struct OPS_ *ops)
{
	VecCreateByMat ((LAPACKVEC **)des_vec, (LAPACKMAT *)src_mat, ops);
	return;
}
static void LAPACK_VecDestroy (void **des_vec, struct OPS_ *ops)
{
	VecDestroy ((LAPACKVEC **)des_vec, ops);
	return;
}
static void LAPACK_VecView (void *x, struct OPS_ *ops)
{
	VecView ((LAPACKVEC *)x, ops);
	return;
}
static void LAPACK_VecInnerProd (void *x, void *y, double *inner_prod, struct OPS_ *ops)
{
	VecInnerProd ((LAPACKVEC *)x, (LAPACKVEC *)y, inner_prod, ops);
	return;
}
static void LAPACK_VecLocalInnerProd (void *x, void *y, double *inner_prod, struct OPS_ *ops)
{
	VecLocalInnerProd ((LAPACKVEC *)x, (LAPACKVEC *)y, inner_prod, ops);
	return;
}
static void LAPACK_VecSetRandomValue (void *x, struct OPS_ *ops)
{
	VecSetRandomValue ((LAPACKVEC *)x, ops);
	return;
}
static void LAPACK_VecAxpby (double alpha, void *x, double beta, void *y, struct OPS_ *ops)
{
	VecAxpby (alpha, (LAPACKVEC *)x, beta, (LAPACKVEC *)y, ops);
	return;
}
static void LAPACK_MatDotVec (void *mat, void *x, void *y, struct OPS_ *ops)
{
	MatDotVec ((LAPACKMAT *)mat, (LAPACKVEC *)x, (LAPACKVEC *)y, ops);
	return;
}
static void LAPACK_MatTransDotVec (void *mat, void *x, void *y, struct OPS_ *ops)
{
	MatTransDotVec ((LAPACKMAT *)mat, (LAPACKVEC *)x, (LAPACKVEC *)y, ops);
	return;
}
/* multi-vec */
static void LAPACK_MultiVecCreateByVec (void ***des_vec, int num_vec, void *src_vec, struct OPS_ *ops)
{
	MultiVecCreateByVec ((LAPACKVEC **)des_vec, num_vec, (LAPACKVEC *)src_vec, ops);
	return;
}
static void LAPACK_MultiVecCreateByMat (void ***des_vec, int num_vec, void *src_mat, struct OPS_ *ops)
{
	MultiVecCreateByMat ((LAPACKVEC **)des_vec, num_vec, (LAPACKMAT *)src_mat, ops);		
	return;
}
static void LAPACK_MultiVecCreateByMultiVec (void ***des_vec, int num_vec, void **src_vec, struct OPS_ *ops)
{
	MultiVecCreateByMultiVec ((LAPACKVEC **)des_vec, num_vec, (LAPACKVEC *)src_vec, ops);
	return;
}
static void LAPACK_MultiVecDestroy (void ***des_vec, int num_vec, struct OPS_ *ops)
{
	MultiVecDestroy ((LAPACKVEC **)des_vec, num_vec, ops);
	return;
}
static void LAPACK_GetVecFromMultiVec (void **multi_vec, int col, void **vec, struct OPS_ *ops)
{
	GetVecFromMultiVec ((LAPACKVEC *)multi_vec, col, (LAPACKVEC **)vec, ops);
	return;
}
static void LAPACK_RestoreVecForMultiVec (void **multi_vec, int col, void **vec, struct OPS_ *ops)
{
	RestoreVecForMultiVec ((LAPACKVEC *)multi_vec, col, (LAPACKVEC **)vec, ops);
	return;
}
static void LAPACK_MultiVecView (void **x, int start, int end, struct OPS_ *ops)
{
	MultiVecView ((LAPACKVEC *)x, start, end, ops);
	return;
}
static void LAPACK_MultiVecLocalInnerProd (char nsdIP, 
		void **x, void **y, int is_vec, int *start, int *end, 
		double *inner_prod, int ldIP, struct OPS_ *ops)
{
	MultiVecLocalInnerProd (nsdIP, 
			(LAPACKVEC *)x, (LAPACKVEC *)y, is_vec, start, end, 
			inner_prod, ldIP, ops);
	return;
}
static void LAPACK_MultiVecInnerProd (char nsdIP, 
		void **x, void **y, int is_vec, int *start, int *end, 
		double *inner_prod, int ldIP, struct OPS_ *ops)
{
	MultiVecInnerProd (nsdIP, 
			(LAPACKVEC *)x, (LAPACKVEC *)y, is_vec, start, end, 
			inner_prod, ldIP, ops);
	return;
}
static void LAPACK_MultiVecSetRandomValue (void **x, int start, int end, struct OPS_ *ops)
{
	MultiVecSetRandomValue ((LAPACKVEC *)x, start, end, ops);
	return;
}
static void LAPACK_MultiVecAxpby (double alpha, void **x, 
		double beta, void **y, int *start, int *end, struct OPS_ *ops)
{
	MultiVecAxpby (alpha, (LAPACKVEC *)x, 
			beta, (LAPACKVEC *)y, start, end, ops);
	return;
}
static void LAPACK_MatDotMultiVec (void *mat, void **x, 
		void **y, int *start, int *end, struct OPS_ *ops)
{
	MatDotMultiVec ((LAPACKMAT *)mat, (LAPACKVEC *)x, 
			(LAPACKVEC *)y, start, end, ops);
	return;
}
static void LAPACK_MatTransDotMultiVec (void *mat, void **x, 
		void **y, int *start, int *end, struct OPS_ *ops)
{
	MatTransDotMultiVec ((LAPACKMAT *)mat, (LAPACKVEC *)x, 
			(LAPACKVEC *)y, start, end, ops);
	return;
}
static void LAPACK_MultiVecLinearComb (
		void **x , void **y, int is_vec, 
		int    *start, int  *end, 
		double *coef , int  ldc , 
		double *beta , int  incb, struct OPS_ *ops)
{
	MultiVecLinearComb (
			(LAPACKVEC *)x , (LAPACKVEC *)y, is_vec, 
			start, end, 
			coef , ldc, 
			beta , incb, ops);
	return;
}
static void LAPACK_MultiVecQtAP (char ntsA, char nsdQAP, 
		void **mvQ , void *matA, void   **mvP, int is_vec, 
		int  *start, int  *end , double *qAp , int ldQAP ,
		void **mv_ws, struct OPS_ *ops)
{
	MultiVecQtAP (ntsA, nsdQAP, 
			(LAPACKVEC *)mvQ, (LAPACKMAT *)matA, (LAPACKVEC *)mvP, is_vec, 
			start, end, qAp, ldQAP,
			(LAPACKVEC *)mv_ws, ops);
	return;
}

void MultiGridCreate (LAPACKMAT ***A_array, LAPACKMAT ***B_array, LAPACKMAT ***P_array,
		int *num_levels, LAPACKMAT *A, LAPACKMAT *B, struct OPS_ *ops)
{
	ops->Printf("Just a test, P is fixed\n");
	int nrows, ncols, row, col, level; double *dbl_ws;
	 
	(*A_array) = malloc((*num_levels)  *sizeof(LAPACKMAT*));
	(*P_array) = malloc((*num_levels-1)*sizeof(LAPACKMAT*));
	
	(*A_array)[0] = A;
	for (level = 1; level < *num_levels; ++level) {
		(*A_array)[level]   = malloc(sizeof(LAPACKMAT));	
		(*P_array)[level-1] = malloc(sizeof(LAPACKMAT));	
	}
	nrows = A->nrows; ncols = (nrows-1)/2;

	dbl_ws = malloc(nrows*ncols*sizeof(double));
	memset(dbl_ws,0,nrows*ncols*sizeof(double));
	for (level = 1; level < *num_levels; ++level) {
		ops->Printf("nrows = %d, ncols = %d\n",nrows,ncols);	
		(*P_array)[level-1]->nrows = nrows; 
		(*P_array)[level-1]->ncols = ncols;
		(*P_array)[level-1]->ldd   = nrows;
		(*P_array)[level-1]->data  = malloc(nrows*ncols*sizeof(double));
		memset((*P_array)[level-1]->data,0,nrows*ncols*sizeof(double));
		for (col = 0; col < ncols; ++col) {
			row = col*2+1;
			(*P_array)[level-1]->data[nrows*col+row]   = 1.0;
			(*P_array)[level-1]->data[nrows*col+row-1] = 0.5;
			(*P_array)[level-1]->data[nrows*col+row+1] = 0.5;
		}
		(*A_array)[level]->nrows = ncols;
		(*A_array)[level]->ncols = ncols;
		(*A_array)[level]->ldd   = ncols;
		(*A_array)[level]->data  = malloc(ncols*ncols*sizeof(double));
		memset((*A_array)[level]->data,0,ncols*ncols*sizeof(double));
		DenseMatQtAP('L','S',nrows,nrows,ncols,ncols,
			1.0,(*P_array)[level-1]->data,nrows,
			    (*A_array)[level-1]->data,nrows,
			    (*P_array)[level-1]->data,nrows,
			0.0,(*A_array)[level  ]->data,ncols,dbl_ws);

		nrows = ncols; ncols = (nrows-1)/2;
	}
	if (B!=NULL) {
		(*B_array) = malloc((*num_levels)*sizeof(LAPACKMAT*));
		(*B_array)[0] = B;
		for (level = 1; level < *num_levels; ++level) {
			(*B_array)[level] = malloc(sizeof(LAPACKMAT));
		}
		nrows = B->nrows; ncols = (nrows-1)/2;
		for (level = 1; level < *num_levels; ++level) {
			(*B_array)[level]->nrows = ncols;
			(*B_array)[level]->ncols = ncols;
			(*B_array)[level]->ldd   = ncols;
			(*B_array)[level]->data  = malloc(ncols*ncols*sizeof(double));
			memset((*B_array)[level]->data,0,ncols*ncols*sizeof(double));
			DenseMatQtAP('L','S',nrows,nrows,ncols,ncols,
				1.0,(*P_array)[level-1]->data,nrows,
				    (*B_array)[level-1]->data,nrows,
				    (*P_array)[level-1]->data,nrows,
				0.0,(*B_array)[level  ]->data,ncols,dbl_ws);
			nrows = ncols; ncols = (nrows-1)/2;
		}
	}
	free(dbl_ws);
	return;
}
void MultiGridDestroy (LAPACKMAT ***A_array, LAPACKMAT ***B_array, LAPACKMAT ***P_array,
		int *num_levels, struct OPS_ *ops)
{	
	int level;
	(*A_array)[0] = NULL;
	for (level = 1; level < *num_levels; ++level) {
		free((*A_array)[level]->data)  ; (*A_array)[level]->data   = NULL;
		free((*A_array)[level])        ; (*A_array)[level]         = NULL;
		free((*P_array)[level-1]->data); (*P_array)[level-1]->data = NULL;
		free((*P_array)[level-1])      ; (*P_array)[level-1]       = NULL;
	}
	free(*A_array); *A_array = NULL;
	free(*P_array); *P_array = NULL;
	
	if (B_array!=NULL) {
		(*B_array)[0] = NULL;
		for (level = 1; level < *num_levels; ++level) {
			free((*B_array)[level]->data); (*B_array)[level]->data = NULL;
			free((*B_array)[level])      ; (*B_array)[level]       = NULL;
		}
		free(*B_array); *B_array = NULL;
	}
	return;
}

static void LAPACK_MultiGridCreate (void ***A_array, void ***B_array, void ***P_array,
		int *num_levels, void *A, void *B, struct OPS_ *ops)
{
	MultiGridCreate ((LAPACKMAT***)A_array,(LAPACKMAT***)B_array,(LAPACKMAT***)P_array,
		num_levels,(LAPACKMAT*)A,(LAPACKMAT*)B, ops);
	return;			
}
static void LAPACK_MultiGridDestroy (void ***A_array , void ***B_array, void ***P_array,
		int *num_levels, struct OPS_ *ops)
{
	MultiGridDestroy ((LAPACKMAT***)A_array,(LAPACKMAT***)B_array,(LAPACKMAT***)P_array,num_levels, ops);
	return;
}
		
void OPS_LAPACK_Set (struct OPS_ *ops)
{
	ops->Printf                   = DefaultPrintf  ;
	ops->GetWtime                 = DefaultGetWtime;
	ops->GetOptionFromCommandLine = DefaultGetOptionFromCommandLine;
	ops->MatView                  = LAPACK_MatView;
	/* vec */
	ops->VecCreateByMat           = LAPACK_VecCreateByMat   ;
	ops->VecCreateByVec           = LAPACK_VecCreateByVec   ;
	ops->VecDestroy               = LAPACK_VecDestroy       ;
	ops->VecView                  = LAPACK_VecView          ;
	ops->VecInnerProd             = LAPACK_VecInnerProd     ;
	ops->VecLocalInnerProd        = LAPACK_VecLocalInnerProd;
	ops->VecSetRandomValue        = LAPACK_VecSetRandomValue;
	ops->VecAxpby                 = LAPACK_VecAxpby         ;
	ops->MatDotVec                = LAPACK_MatDotVec        ;
	ops->MatTransDotVec           = LAPACK_MatTransDotVec   ;
	/* multi-vec */
	ops->MultiVecCreateByMat      = LAPACK_MultiVecCreateByMat     ;
	ops->MultiVecCreateByVec      = LAPACK_MultiVecCreateByVec     ;
	ops->MultiVecCreateByMultiVec = LAPACK_MultiVecCreateByMultiVec;
	ops->MultiVecDestroy          = LAPACK_MultiVecDestroy         ;
	ops->GetVecFromMultiVec       = LAPACK_GetVecFromMultiVec      ;
	ops->RestoreVecForMultiVec    = LAPACK_RestoreVecForMultiVec   ;
	ops->MultiVecView             = LAPACK_MultiVecView            ;
	ops->MultiVecLocalInnerProd   = LAPACK_MultiVecLocalInnerProd  ;
	ops->MultiVecInnerProd        = LAPACK_MultiVecInnerProd       ;
	ops->MultiVecSetRandomValue   = LAPACK_MultiVecSetRandomValue  ;
	ops->MultiVecAxpby            = LAPACK_MultiVecAxpby           ;
	ops->MultiVecLinearComb       = LAPACK_MultiVecLinearComb      ;
	ops->MatDotMultiVec           = LAPACK_MatDotMultiVec          ;
	ops->MatTransDotMultiVec      = LAPACK_MatTransDotMultiVec     ;
	if (0)
		ops->MultiVecQtAP     = LAPACK_MultiVecQtAP            ;
	else
		ops->MultiVecQtAP     = DefaultMultiVecQtAP            ;
	/* dense mat */
	ops->lapack_ops               = NULL        ;
	ops->DenseMatQtAP             = DenseMatQtAP;
	ops->DenseMatOrth             = DenseMatOrth;
	/* multi grid */
	ops->MultiGridCreate          = LAPACK_MultiGridCreate ;
	ops->MultiGridDestroy         = LAPACK_MultiGridDestroy;
	return;
}

