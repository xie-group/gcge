/**
 *    @file  ops_orth.c
 *   @brief  正交化操作 
 *
 *  正交化操作
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/8/17
 *      Revision:  none
 */

#include	<stdio.h>
#include	<stdlib.h>
#include    <assert.h> 
#include	<float.h>
#include    <math.h>
#include    <time.h>

#include    "ops_orth.h"

#define  DEBUG 0
#define  TIME_MGS 0
#define  TIME_BGS 0

typedef struct TimeMGS_ {
	double axpby_time;
	double line_comb_time;
	double orth_self_time;
    double qAp_time;
	double time_total;
} TimeMGS;

typedef struct TimeBGS_ {
	double axpby_time;
	double line_comb_time;
	double orth_self_time;
    double qAp_time;
	double time_total;
} TimeBGS;

struct TimeMGS_ time_mgs = {0.0,0.0,0.0,0.0,0.0};
struct TimeBGS_ time_bgs = {0.0,0.0,0.0,0.0,0.0};

static void OrthSelf(void **x,int start_x,int *end_x, void *B, int max_reorth,
		double orth_zero_tol,double reorth_tol,void **mv_ws,double *dbl_ws,struct OPS_ *ops)
{
	if (*end_x<=start_x) return;
	
	int    k, start[2], end[2], length, inc, idx, idx_abs_max; 
	double *r_k, *beta, *coef;
	r_k  = dbl_ws; beta = dbl_ws; coef = dbl_ws+1; 
	for (k = start_x; k < (*end_x); ++k) {
		/* compute r_k */
		start[0] = k; end[0] = *end_x;
		start[1] = k; end[1] = k+1;
		ops->MultiVecQtAP('S','N',x,B,x,0,start,end,
				r_k,end[0]-start[0],mv_ws,ops);
#if DEBUG
		ops->Printf("k = %d, start = %d,%d, end = %d,%d\n", k,start[0],start[1],end[0],end[1]);
		ops->Printf("r_k = %e\n", *r_k);
#endif
		*r_k = sqrt(*r_k);
		if (*r_k < orth_zero_tol) {
			ops->Printf("r_[%d] = %6.4e\n",k,*r_k);
			/* 保证 k 不指向 最后一个 向量, 若是最后一个, 直接舍去 */
			if (k < *end_x-1) {
				start[0] = *end_x-1; end[0] = *end_x;
				start[1] = k       ; end[1] = k+1   ;
				ops->MultiVecAxpby(1.0,x,0.0,x,start,end,ops);
			}
			--k; --(*end_x);
			continue;
		}
		else {
			start[0] = k; end[0] = k+1;
			start[1] = k; end[1] = k+1;
			/* compute q_k */
			*r_k = 1.0/(*r_k);
			ops->MultiVecAxpby(0.0,NULL,*r_k,x,start,end,ops);
		}
		/* compute x_k+1 ... x_n */
		if (k<*end_x-1) {
			length = *end_x-(k+1);
			*r_k *= -1.0; inc = 1;
			dscal(&length,r_k,coef,&inc);			
			*beta = 1.0;
			start[0] = k  ; end[0] = k+1   ;
			start[1] = k+1; end[1] = *end_x;
			ops->MultiVecLinearComb(x,x,0,start,end,
				coef,end[0]-start[0],beta,0,ops);			

			for (idx = 1; idx < -1+max_reorth; ++idx) {
				start[0] = k+1; end[0] = *end_x;
				start[1] = k  ; end[1] = k+1   ;
				ops->MultiVecQtAP('S','N',x,B,x,0,start,end,
						coef,end[0]-start[0],mv_ws,ops);				
				length = (*end_x-k-1);
				*beta = -1.0; inc = 1;
				dscal(&length,beta,coef,&inc);				
				*beta = 1.0;
				start[0] = k  ; end[0] = k+1   ;
				start[1] = k+1; end[1] = *end_x;
				ops->MultiVecLinearComb(x,x,0,start,end,
						coef,end[0]-start[0],beta,0,ops);			

				idx_abs_max = idamax(&length,coef,&inc);
				if (fabs(coef[idx_abs_max-1]) < reorth_tol) {
#if DEBUG
				   ops->Printf("OrthSelf: The number of reorth = %d\n", idx);
#endif
				   break;
				}
			}			
		}
	}
	return;
}


/* the size of mv_ws is end[0]-start[0] */
static void OrthSelfEVP(void **x,int start_x,int *end_x, void *B, int max_reorth,
		double orth_zero_tol,double reorth_tol,void **mv_ws,double *dbl_ws,struct OPS_ *ops)
{
	if (*end_x<=start_x) return;
#if DEBUG 
	ops->Printf("start_x = %d, end_x = %d\n", start_x, *end_x);
#endif
	int    k, start[2], end[2], idx, inc = 1, lin_dep; 
	char   JOBZ = 'V', UPLO = 'L';
	int    N, LDA, LWORK, INFO;
	double *A, *W, *WORK;

	for (idx = 0; idx < 1+max_reorth; ++idx) {
		if (*end_x==start_x) return;
		assert(*end_x-start_x>=1);
		lin_dep = 0;
		N = *end_x-start_x; LDA = N  ; LWORK = 3*N*N-N;
		A = dbl_ws        ; W = A+N*N; WORK  = W+N;
		start[0] = start_x; end[0] = *end_x;
		start[1] = start_x; end[1] = *end_x;
		ops->MultiVecQtAP('S','S',x,B,x,0,start,end,
				A,end[0]-start[0],mv_ws,ops);
		dsyev(&JOBZ,&UPLO,&N,A,&LDA,W,WORK,&LWORK,&INFO);
#if DEBUG
		//ops->MultiVecView(x, start[0], end[0], ops);
		int i;
		for (i = 0; i < N*N; ++i) {
		   if (i%LDA==0) printf("\n");
		   ops->Printf("%e\t", A[i]);
		}
		ops->Printf("\n");
		ops->Printf("INFO = %d, start = %d-%d, end = %d-%d\n", 
		      INFO, start[0], start[1], end[0], end[1]);
#endif
		assert(INFO==0);
		/* W is in ascending order */
#if DEBUG
		ops->Printf("\n%d:\n",idx);
#endif
		for (k = 0; k < N; ++k) {
#if DEBUG
			ops->Printf("%e\n",W[k]);
#endif
			/* 理论上 A 是半正定矩阵, 特征值>=0 */
			assert(W[k] > -orth_zero_tol);
			/* 这里 与 OrthSelf 不同的是, orth_zero_tol 判断的是 模长平方
			 * 若判断 sqrt(W[k]) 容易得到错误的 向量组的秩
			 * 原因是 特征值有精度 */
			if (fabs(W[k]) > orth_zero_tol) {
				W[k] = 1.0/sqrt(W[k]);
			}
			else {
				++lin_dep;
			}
		}
		if (lin_dep > 0) {
			ops->Printf("There has %d linear dependent vec\n",lin_dep);
		}
		/* mv_ws = x A */
		start[0] = start_x  ; end[0] = *end_x;
		start[1] = 0        ; end[1] = N-lin_dep;
		ops->MultiVecLinearComb(x,mv_ws,0,start,end,
				A+LDA*lin_dep,end[0]-start[0],NULL,0,ops);			
		/* mv_ws = mv_ws * 1/sqrt(lambda) */
		ops->MultiVecLinearComb(NULL,mv_ws,0,start,end,
				NULL,0,W+lin_dep,1,ops);
		*end_x = *end_x - lin_dep;
		start[0] = 0        ; end[0] = N-lin_dep;
		start[1] = start_x  ; end[1] = *end_x;
		ops->MultiVecAxpby(1.0,mv_ws,0.0,x,start,end,ops);
		if (lin_dep==0&&fabs(dasum(&N,W,&inc)-N)<reorth_tol) {
#if DEBUG
			ops->Printf("lin_dep = %d, dasum(&N,W,&inc) = %e, reorth_tol = %e\n", 
				lin_dep, dasum(&N,W,&inc), reorth_tol);
#endif
			break;
		}	
	}
	return;
}

static void ModifiedGramSchmidt(void **x, int start_x, int *end_x, 
		void *B, struct OPS_ *ops)
{
#if DEBUG
        ops->Printf("ModifiedGramSchmidtOrth (%d, %d)\n", start_x, *end_x);
#endif
	if (*end_x <= start_x) return;
	
#if TIME_MGS
	time_mgs.axpby_time     = 0.0;
	time_mgs.line_comb_time = 0.0;
	time_mgs.orth_self_time = 0.0;
	time_mgs.qAp_time       = 0.0;
#endif
	
	ModifiedGramSchmidtOrth *mgs_orth = 
			(ModifiedGramSchmidtOrth*)ops->orth_workspace;
	int    start[2], end[2], length, block_size, idx, idx_abs_max;
//	int    nrows   , ncols , incx  , incy, row; 
	int incx;
//	double *source , *destin;
	double *coef   , *beta , orth_zero_tol, reorth_tol;
	void   **mv_ws;
	orth_zero_tol = mgs_orth->orth_zero_tol;
	reorth_tol    = mgs_orth->reorth_tol;
	block_size    = mgs_orth->block_size;
	mv_ws         = mgs_orth->mv_ws ;
	beta          = mgs_orth->dbl_ws;
	start[0] = 0     ; end[0] = start_x;
	start[1] = end[0]; end[1] = *end_x ;
	coef     = beta+1;
	/* 去掉 X1 中 X0 的部分 */
	if (start_x > 0) {
		for (idx = 0; idx < 1+mgs_orth->max_reorth; ++idx) {	
#if TIME_MGS
			time_mgs.qAp_time -= ops->GetWtime();
#endif
			ops->MultiVecQtAP('S','N',x,B,x,0,start,end,
					coef,end[0]-start[0],mv_ws,ops);
#if TIME_MGS
			time_mgs.qAp_time += ops->GetWtime();
#endif 

			length = (end[1] - start[1])*(end[0] - start[0]);
			*beta = -1.0; incx = 1;
			dscal(&length,beta,coef,&incx);
			*beta = 1.0;

#if TIME_MGS
			time_mgs.line_comb_time -= ops->GetWtime();
#endif
			ops->MultiVecLinearComb(x,x,0,start,end,
					coef,end[0]-start[0],beta,0,ops);
#if TIME_MGS
			time_mgs.line_comb_time += ops->GetWtime();
#endif 


			idx_abs_max = idamax(&length,coef,&incx);
			if (fabs(coef[idx_abs_max-1]) < reorth_tol) {
#if DEBUG 
			   ops->Printf("X1 - X0: The number of reorth = %d\n", idx);
#endif
			   break;
			}
		}
	}
	
	/* 块正交化 */
	int init_start = start_x, init_end;
	/* 默认为要正交的向量长度的一半 */
	if (block_size <= 0) {
	   block_size = (*end_x-init_start)/2 > 2 ? (*end_x-init_start)/2 : 2;
	}
	block_size = (block_size<*end_x-init_start)?block_size:(*end_x-init_start);
	//ops->Printf("block_size = %d\n", block_size);
	while (block_size > 0) {
#if DEBUG
		ops->Printf("start block_size %d\n", block_size);
#endif
		start[1] = init_start; end[1] = start[1]+block_size;
#if TIME_MGS
		time_mgs.orth_self_time -= ops->GetWtime();
#endif
		OrthSelf(x,start[1],&(end[1]),B,
		      mgs_orth->max_reorth,orth_zero_tol,reorth_tol,
		      mv_ws,mgs_orth->dbl_ws,ops);	      
#if TIME_MGS
		time_mgs.orth_self_time += ops->GetWtime();
#endif 
		init_end = end[1];
		/* 将后面部分复制到线性相关部分 */
		length = block_size - (end[1]-start[1]);								
		length = (length<*end_x-end[1]-length)?length:(*end_x-end[1]-length);
		if (length > 0) {
			end[0]   = *end_x; start[0] = end[0]-length;
			start[1] = init_end; end[1] = start[1]+length;
#if TIME_MGS
			time_mgs.axpby_time -= ops->GetWtime();
#endif 
			ops->MultiVecAxpby(1.0,x,0.0,x,start,end,ops);			
#if TIME_MGS
			time_mgs.axpby_time += ops->GetWtime();
#endif 
		}
		*end_x = *end_x - (block_size-(init_end-init_start));
		/* 将 去掉后面部分 中的 前面正交化部分 */
		if ( init_end < (*end_x) && init_start < init_end ) {	
			for (idx = 0; idx < 1+mgs_orth->max_reorth; ++idx) {
#if TIME_MGS
				time_mgs.qAp_time -= ops->GetWtime();
#endif
				/* 重正交化时, 无需再算 Bx, 用 mv_ws 即可, 另外, QtAP 中 coef 设为 'T', 则不再需要转置 */
				if (B!=NULL && idx > 0) {
					start[0] = init_end  ; end[0] = *end_x;
					start[1] = 0         ; end[1] = init_end - init_start;		
#if DEBUG
					ops->Printf("start = %d,%d, end = %d,%d, idx = %d\n",start[0],start[1],end[0],end[1],idx);
#endif
					ops->MultiVecQtAP('S','T',x,NULL,mv_ws,0,start,end,
							coef,end[1]-start[1],mv_ws,ops);
				}
				else {
					start[0] = init_end  ; end[0] = *end_x  ;
					start[1] = init_start; end[1] = init_end;		
#if DEBUG
					ops->Printf("start = %d,%d, end = %d,%d, idx = %d\n",start[0],start[1],end[0],end[1],idx);
#endif
					ops->MultiVecQtAP('S','T',x,B,x,0,start,end,
							coef,end[1]-start[1],mv_ws,ops);

				}
#if TIME_MGS
				time_mgs.qAp_time += ops->GetWtime();
#endif
				length = (end[1] - start[1])*(end[0] - start[0]); 
				*beta  = -1.0; incx = 1;
				dscal(&length,beta,coef,&incx);
				*beta  = 1.0;

				start[0] = init_start; end[0] = init_end;
				start[1] = init_end  ; end[1] = *end_x  ;
#if TIME_MGS
				time_mgs.line_comb_time -= ops->GetWtime();
#endif 
				ops->MultiVecLinearComb(x,x,0,start,end,
						coef,end[0]-start[0],beta,0,ops);
#if TIME_MGS
				time_mgs.line_comb_time += ops->GetWtime();
#endif 
				incx = 1;
				idx_abs_max = idamax(&length,coef,&incx);
				if (fabs(coef[idx_abs_max-1]) < reorth_tol) {
#if DEBUG
				   ops->Printf("X1 - block_size: The number of reorth = %d\n", idx);
#endif
				   break;
				}
			}
		}
		init_start = init_end;
		block_size = (block_size<*end_x-init_start)?block_size:(*end_x-init_start);		
#if DEBUG
		ops->Printf("end block_size %d\n", block_size);
#endif
	}
	
#if TIME_MGS
	ops->Printf("|--MGS----------------------------\n");
	time_mgs.time_total = time_mgs.axpby_time
		+time_mgs.line_comb_time
		+time_mgs.orth_self_time
		+time_mgs.qAp_time;
	ops->Printf("|axpby  line_comb  orth_self  qAp\n");
	ops->Printf("|%.2f\t%.2f\t%.2f\t%.2f\n",
		time_mgs.axpby_time,		
		time_mgs.line_comb_time,		
		time_mgs.orth_self_time,		
		time_mgs.qAp_time);
	ops->Printf("|%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\n",
		time_mgs.axpby_time    /time_mgs.time_total*100,
		time_mgs.line_comb_time/time_mgs.time_total*100,
		time_mgs.orth_self_time/time_mgs.time_total*100,
		time_mgs.qAp_time      /time_mgs.time_total*100);
	ops->Printf("|--MGS----------------------------\n");
	time_mgs.axpby_time     = 0.0;
	time_mgs.line_comb_time = 0.0;
	time_mgs.orth_self_time = 0.0;
	time_mgs.qAp_time       = 0.0;	
#endif
	return;
}
void MultiVecOrthSetup_ModifiedGramSchmidt(
		int block_size, int max_reorth, double orth_zero_tol, 
		void **mv_ws, double *dbl_ws, struct OPS_ *ops)
{
#if DEBUG
        ops->Printf("MultiVecOrthSetup_ModifiedGramSchmidt (%p, %p)\n", mv_ws, dbl_ws);
#endif
	static ModifiedGramSchmidtOrth mgs_orth_static = {
		.block_size = -1  , .orth_zero_tol = 20*DBL_EPSILON,
		.max_reorth = 3   , .reorth_tol    = 50*DBL_EPSILON,
		.mv_ws      = NULL, .dbl_ws        = NULL};
	mgs_orth_static.block_size    = block_size   ;
	mgs_orth_static.orth_zero_tol = orth_zero_tol;
	mgs_orth_static.max_reorth    = max_reorth;
	mgs_orth_static.mv_ws         = mv_ws ;
	mgs_orth_static.dbl_ws        = dbl_ws;
	ops->orth_workspace = (void *)&mgs_orth_static;
	ops->MultiVecOrth = ModifiedGramSchmidt;
	return;
}

static void OrthBinary(void **x,int start_x, int *end_x, void *B, char orth_self_method,
	int block_size, int max_reorth, double orth_zero_tol, double reorth_tol,
	void **mv_ws, double *dbl_ws, struct OPS_ *ops)
{
	if (*end_x<=start_x) return;
#if DEBUG
	ops->Printf("%d,%d,%d\n",start_x,*end_x,block_size);
#endif
		
	int ncols = *end_x-start_x, length, start[2], end[2], idx, inc, idx_abs_max;
	double *beta = dbl_ws, *coef = beta+1;
	if (ncols<=block_size) {
#if TIME_BGS
		time_bgs.orth_self_time -= ops->GetWtime();
#endif
		if (orth_self_method=='E') {
			OrthSelfEVP(x,start_x,end_x,B,
					max_reorth,orth_zero_tol,reorth_tol,mv_ws,coef,ops);
		}
		else {
			OrthSelf(x,start_x,end_x,B,
					max_reorth,orth_zero_tol,reorth_tol,mv_ws,coef,ops);
		}
#if TIME_BGS
		time_bgs.orth_self_time += ops->GetWtime();
#endif
	}
	else {
		start[0] = start_x; end[0] = start_x+ncols/2;
		start[1] = end[0] ; end[1] = *end_x;
		/* 正交化 X0: end[0] 可能会改变 */
		OrthBinary(x,start[0],&end[0],B,orth_self_method,
		      block_size,max_reorth,orth_zero_tol,reorth_tol,
		      mv_ws,dbl_ws,ops);		
		//ops->MultiVecView(x, 80, 81, ops);
		//ops->Printf("start = %d-%d, end = %d-%d\n", 
		//      start[0], start[1], end[0], end[1]);
		/* 去掉 X1 中 X0 的部分 */
		for (idx = 0; idx < 1+max_reorth; ++idx) {
#if TIME_BGS
			time_bgs.qAp_time -= ops->GetWtime();
#endif
			int start_QtAP[2], end_QtAP[2];
			if (B!=NULL && idx > 0) {
				start_QtAP[0] = start[1]; end_QtAP[0] = end[1];
				start_QtAP[1] = 0       ; end_QtAP[1] = end[0]-start[0];
				ops->MultiVecQtAP('S','T',x,NULL,mv_ws,0,start_QtAP,end_QtAP,
						coef,end_QtAP[1]-start_QtAP[1],mv_ws,ops);
			}
			else {
				start_QtAP[0] = start[1]; end_QtAP[0] = end[1];
				start_QtAP[1] = start[0]; end_QtAP[1] = end[0];
				ops->MultiVecQtAP('S','T',x,B,x,0,start_QtAP,end_QtAP,
						coef,end_QtAP[1]-start_QtAP[1],mv_ws,ops);
			}
#if TIME_BGS
			time_bgs.qAp_time += ops->GetWtime();
#endif

			length = (end[1] - start[1])*(end[0] - start[0]);
			*beta  = -1.0; inc = 1;
			dscal(&length,beta,coef,&inc);		
			*beta  = 1.0;
#if TIME_BGS
			time_bgs.line_comb_time -= ops->GetWtime();
#endif			
			ops->MultiVecLinearComb(x,x,0,start,end,
					coef,end[0]-start[0],beta,0,ops);
#if TIME_BGS
			time_bgs.line_comb_time += ops->GetWtime();
#endif			

			idx_abs_max = idamax(&length,coef,&inc);
			if (fabs(coef[idx_abs_max-1]) < reorth_tol) {
#if DEBUG 
			   ops->Printf("X1 - X0: The number of reorth = %d\n", idx);
#endif
			   break;
			}
		}			
		/* 正交化 X1 */ 
		OrthBinary(x,start[1],&end[1],B,orth_self_method,
		      block_size,max_reorth,orth_zero_tol,reorth_tol,
		      mv_ws,dbl_ws,ops);
		/* 将 X0 中线性相关部分用 X1 中的正交化部分填充 */
		length = start[0]+ncols/2-end[0]; /* 线性相关部分的长度 */
		*end_x = end[1]-length;
		/* X1 正交部分可以为 X0 线性部分提供的长度*/
		length = (length<end[1]-start[1])?length:(end[1]-start[1]);
		start[1] = end[0]; /* X0 线性相关的起始位置 */
		end[0]	 = end[1]; /* X1 正交部分的结束位置 */
		start[0] = end[0]   - length;
		end[1]   = start[1] + length;
#if TIME_BGS
		time_bgs.axpby_time -= ops->GetWtime();
#endif 
		ops->MultiVecAxpby(1.0,x,0.0,x,start,end,ops);
#if TIME_BGS
		time_bgs.axpby_time += ops->GetWtime();
#endif				
	}
	return;
}
static void BinaryGramSchmidt(void **x, int start_x, int *end_x, 
		void *B, struct OPS_ *ops)
{
	if (*end_x<=start_x) return;
#if TIME_BGS
	time_bgs.axpby_time     = 0.0;
	time_bgs.line_comb_time = 0.0;
	time_bgs.orth_self_time = 0.0;
	time_bgs.qAp_time       = 0.0;
#endif
	
	BinaryGramSchmidtOrth *bgs_orth = 
		(BinaryGramSchmidtOrth*)ops->orth_workspace;
	int    start[2], end[2], block_size, idx, length, inc, idx_abs_max;
	double *coef   , *beta , orth_zero_tol, reorth_tol;
	void   **mv_ws;
	orth_zero_tol = bgs_orth->orth_zero_tol;
	reorth_tol    = bgs_orth->reorth_tol;
	block_size    = bgs_orth->block_size;
	mv_ws         = bgs_orth->mv_ws;
	beta          = bgs_orth->dbl_ws;
	/* 去掉 X1 中 X0 的部分 */
	if (start_x > 0) {
		start[0] = 0     ; end[0] = start_x;
		start[1] = end[0]; end[1] = *end_x ;
		coef     = beta+1; 
		for (idx = 0; idx < 1+bgs_orth->max_reorth; ++idx) {
#if TIME_BGS
			time_bgs.qAp_time -= ops->GetWtime();
#endif
			ops->MultiVecQtAP('S','N',x,B,x,0,start,end,
					coef,end[0]-start[0],mv_ws,ops);
#if TIME_BGS
			time_bgs.qAp_time += ops->GetWtime();
#endif
			length = (end[1] - start[1])*(end[0] - start[0]); 
			*beta  = -1.0; inc = 1;
			dscal(&length,beta,coef,&inc);		
			*beta  = 1.0;
#if TIME_BGS
			time_bgs.line_comb_time -= ops->GetWtime();
#endif
			ops->MultiVecLinearComb(x,x,0,start,end,
					coef,end[0]-start[0],beta,0,ops);
#if TIME_BGS
			time_bgs.line_comb_time += ops->GetWtime();
#endif		
			idx_abs_max = idamax(&length,coef,&inc);
			if (fabs(coef[idx_abs_max-1]) < reorth_tol) {
#if DEBUG
			   ops->Printf("X1 - X0: The number of reorth = %d\n", idx);
#endif
			   break;
			}
		}
	}
	/* 二分块正交化 */
	/* 4<= block_size <= (*end_x-start_x)/4 */
	char orth_self_method;
	if ((*end_x-start_x)<16) {
		if (block_size<=0) {
			block_size = 4;
		}
		/* 使用 OrthSelf */
		orth_self_method = 'M';		
	}
	else {		
		if (block_size<=0 || block_size>(*end_x-start_x)/4) {
			block_size = (*end_x-start_x)/4;
		}
		/* 使用 OrthSelfEVP */
		orth_self_method = 'E';
	}
	//ops->Printf("start_x = %d, end_x = %d, block_size = %d\n", start_x, *end_x, block_size);

	OrthBinary(x,start_x,end_x,B,orth_self_method,
	      block_size,bgs_orth->max_reorth,orth_zero_tol,reorth_tol,
	      mv_ws,bgs_orth->dbl_ws,ops);
	      
	      
#if TIME_BGS
	ops->Printf("|--BGS----------------------------\n");
	time_bgs.time_total = time_bgs.axpby_time
		+time_bgs.line_comb_time
		+time_bgs.orth_self_time
		+time_bgs.qAp_time;
	ops->Printf("|axpby  line_comb  orth_self  qAp\n");
	ops->Printf("|%.2f\t%.2f\t%.2f\t%.2f\n",
		time_bgs.axpby_time,		
		time_bgs.line_comb_time,		
		time_bgs.orth_self_time,		
		time_bgs.qAp_time);
	ops->Printf("|%.2f%%\t%.2f%%\t%.2f%%\t%.2f%%\n",
		time_bgs.axpby_time    /time_bgs.time_total*100,
		time_bgs.line_comb_time/time_bgs.time_total*100,
		time_bgs.orth_self_time/time_bgs.time_total*100,
		time_bgs.qAp_time      /time_bgs.time_total*100);
	ops->Printf("|--BGS----------------------------\n");
	time_bgs.axpby_time     = 0.0;
	time_bgs.line_comb_time = 0.0;
	time_bgs.orth_self_time = 0.0;
	time_bgs.qAp_time       = 0.0;	
#endif
	return;
}

void MultiVecOrthSetup_BinaryGramSchmidt(
		int block_size, int max_reorth, double orth_zero_tol, 
		void **mv_ws, double *dbl_ws, struct OPS_ *ops)
{
	static BinaryGramSchmidtOrth bgs_orth_static = {
		.block_size = 16  , .orth_zero_tol = 20*DBL_EPSILON, 
		.max_reorth = 4   , .reorth_tol    = 50*DBL_EPSILON,
		.mv_ws      = NULL, .dbl_ws        = NULL};
	bgs_orth_static.block_size    = block_size   ;
	bgs_orth_static.orth_zero_tol = orth_zero_tol;
	bgs_orth_static.max_reorth    = max_reorth;
	bgs_orth_static.mv_ws         = mv_ws ;
	bgs_orth_static.dbl_ws        = dbl_ws;
	ops->orth_workspace = (void *)&bgs_orth_static;
	ops->MultiVecOrth = BinaryGramSchmidt;
	return;
}
