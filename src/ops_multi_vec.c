/**
 *    @file  ops_multi_vec.c
 *   @brief  默认多向量操作 
 *
 *  默认多向量操作可以被APP调用
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/8/23
 *      Revision:  none
 */

#include	<stdio.h>
#include	<stdlib.h>
#include	<string.h>
#include    <assert.h>
#include    <stdarg.h>
#include    <time.h>

#include    "ops.h"
#include    "app_lapack.h"

#define DEBUG 0


void DefaultPrintf(const char *fmt, ...)
{
#if OPS_USE_MPI
    int rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(PRINT_RANK == rank) {
        va_list vp;
        va_start(vp, fmt);
        vprintf(fmt, vp);
        va_end(vp);
    }
#else
    va_list vp;
    va_start(vp, fmt);
    vprintf(fmt, vp);
    va_end(vp);
#endif
    return;
}
double DefaultGetWtime(void)
{
   double time;
#if   OPS_USE_MPI
    time = MPI_Wtime();
#elif OPS_USE_OMP || OPS_USE_INTEL_MKL
    time = omp_get_wtime();
#else
    time = (double)clock()/CLOCKS_PER_SEC;
#endif
    return time;
}

int DefaultGetOptionFromCommandLine(
		const char *name, char type, void *value,
		int argc, char* argv[], struct OPS_ *ops)
{
    int arg_idx = 0, set = 0;
    int *int_value; double *dbl_value; char *str_value;
    while(arg_idx < argc) 
    {
       	if(0 == strcmp(argv[arg_idx], name)) {
       		ops->Printf("argv[%d] = \"%s\", name = \"%s\"\n", arg_idx, argv[arg_idx], name);
			if (arg_idx+1 < argc) {
			 	set = 1;
			}
			else {
			 	break;
			}
			switch (type) {
				case 'i':
					int_value  = (int*)value; 
					*int_value = atoi(argv[++arg_idx]);
					break;
				case 'f':
					dbl_value  = (double*)value;
					*dbl_value = atof(argv[++arg_idx]);
					break;
				case 's':
					str_value  = (char*)value;
					strcpy(str_value, argv[++arg_idx]);
					break;
				default:
					break;
			}
	  		break;
       }
       ++arg_idx;
    }
    return set;
}

void DefaultMultiVecCreateByVec      (void ***multi_vec, int num_vec, void *src_vec, struct OPS_ *ops)
{
	int col;
	(*multi_vec) = malloc(num_vec*sizeof(void*));
	for (col = 0; col < num_vec; ++col) {
		ops->VecCreateByVec((*multi_vec)+col,src_vec,ops);
	}
	return;
}
void DefaultMultiVecCreateByMat      (void ***multi_vec, int num_vec, void *src_mat, struct OPS_ *ops)
{
	int col;
	(*multi_vec) = malloc(num_vec*sizeof(void*));
	for (col = 0; col < num_vec; ++col) {
		ops->VecCreateByMat((*multi_vec)+col,src_mat,ops);
	}
	return;
}
void DefaultMultiVecCreateByMultiVec (void ***multi_vec, int num_vec, void **src_mv, struct OPS_ *ops)
{
	int col;
	(*multi_vec) = malloc(num_vec*sizeof(void*));
	for (col = 0; col < num_vec; ++col) {
		ops->VecCreateByVec((*multi_vec)+col,*src_mv,ops);
	}
	return;
}
void DefaultMultiVecDestroy          (void ***multi_vec, int num_vec, struct OPS_ *ops)
{
	int col;
	for (col = 0; col < num_vec; ++col) {
		ops->VecDestroy((*multi_vec)+col,ops);
	}
	free((*multi_vec)); *multi_vec = NULL;
	return;
}
void DefaultGetVecFromMultiVec    (void **multi_vec, int col, void **vec, struct OPS_ *ops)
{
	*vec = multi_vec[col];
	return;
}
void DefaultRestoreVecForMultiVec (void **multi_vec, int col, void **vec, struct OPS_ *ops)
{
	*vec = NULL;
	return;
}
void DefaultMultiVecView           (void **x, int start, int end, struct OPS_ *ops)
{
	int col;
	for (col = start; col < end; ++col) {
		ops->VecView(x[col],ops);
	}
	return;
}
void DefaultMultiVecLocalInnerProd (char nsdIP, void **x, void **y, int is_vec, int *start, int *end, 
	double *inner_prod, int ldIP, struct OPS_ *ops)
{
	int row, col, nrows, ncols, length, incx, incy; 
	double *source, *destin; void *vec_x, *vec_y; 
	nrows = end[0]-start[0]; ncols = end[1]-start[1];
	if (nsdIP=='S') {
		assert(nrows == ncols);
		for (col = 0; col < ncols; ++col) {
			ops->GetVecFromMultiVec(y,start[1]+col,&vec_y,ops);
			destin = inner_prod+ldIP*col+col;
			for (row = col; row < nrows; ++row) {
				ops->GetVecFromMultiVec(x,start[0]+row,&vec_x,ops);
				ops->VecInnerProd(vec_x,vec_y,destin,ops);
				++destin;
				ops->RestoreVecForMultiVec(x,start[0]+row,&vec_x,ops);
			}
			ops->RestoreVecForMultiVec(y,start[1]+col,&vec_y,ops);
		}
		for (col = 0; col < ncols; ++col) {
			length = ncols-col-1;
			source = inner_prod+ldIP*col+(col+1); incx = 1;
			destin = inner_prod+ldIP*(col+1)+col; incy = ldIP;
			dcopy(&length,source,&incx,destin,&incy);
		}
	} 
	else if (nsdIP=='D') {
		assert(nrows == ncols);
		for (col = 0; col < ncols; ++col) {
			ops->GetVecFromMultiVec(y,start[1]+col,&vec_y,ops);
			ops->GetVecFromMultiVec(x,start[0]+col,&vec_x,ops);
			ops->VecInnerProd(vec_x,vec_y,inner_prod+ldIP*col,ops);
			ops->RestoreVecForMultiVec(x,start[0]+col,&vec_x,ops);
			ops->RestoreVecForMultiVec(y,start[1]+col,&vec_y,ops);
		}
	} 
	else {
		for (col = 0; col < ncols; ++col) {
			ops->GetVecFromMultiVec(y,start[1]+col,&vec_y,ops);
			destin = inner_prod+ldIP*col;
			for (row = 0; row < nrows; ++row) {
				ops->GetVecFromMultiVec(x,start[0]+row,&vec_x,ops);
				ops->VecInnerProd(vec_x,vec_y,destin,ops);
				++destin;
				ops->RestoreVecForMultiVec(x,start[0]+row,&vec_x,ops);
			}
			ops->RestoreVecForMultiVec(y,start[1]+col,&vec_y,ops);
		}
	}
	return;
}
void DefaultMultiVecInnerProd      (char nsdIP, void **x, void **y, int is_vec, int *start, int *end, 
	double *inner_prod, int ldIP, struct OPS_ *ops)
{
	ops->MultiVecLocalInnerProd (nsdIP,x,y,is_vec,start,end,inner_prod,ldIP,ops);
#if OPS_USE_MPI

	int nrows = end[0]-start[0], ncols = end[1]-start[1];
	if (nsdIP == 'D') {
		assert(nrows == ncols);
		nrows = 1;
	}
	if (nrows==ldIP) {
	   MPI_Allreduce(MPI_IN_PLACE,inner_prod,
		 nrows*ncols,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	}
	else {
	   MPI_Datatype data_type; MPI_Op op;
	   CreateMPIDataTypeSubMat(&data_type,nrows,ncols,ldIP);	
	   CreateMPIOpSubMatSum(&op);/* 对第一个创建的submat */
	   /* 求和归约, 1 个 SUBMAT_TYPE */
	   MPI_Allreduce(MPI_IN_PLACE,inner_prod,
		 1,data_type,op,MPI_COMM_WORLD);
	   DestroyMPIOpSubMatSum(&op);
	   DestroyMPIDataTypeSubMat(&data_type);
	}

#endif
	return;
}
void DefaultMultiVecSetRandomValue (void **x, int start, int end, struct OPS_ *ops)
{
	int col;
	for (col = start; col < end; ++col) {
		ops->VecSetRandomValue(x[col],ops);
	}
	return;
}
void DefaultMultiVecAxpby          (
	double alpha , void **x , double beta, void **y, 
	int    *start, int  *end, struct OPS_ *ops)
{
	int col, ncols = end[1]-start[1]; void *vec_x, *vec_y;
	if (alpha == 0.0 || x == NULL) {
		for (col = 0; col < ncols; ++col) {
			ops->GetVecFromMultiVec(y,start[1]+col,&vec_y,ops);
			ops->VecAxpby(alpha,NULL,beta,vec_y,ops);
			ops->RestoreVecForMultiVec(y,start[1]+col,&vec_y,ops);
		}
		
	} 
	else {
		for (col = 0; col < ncols; ++col) {
			ops->GetVecFromMultiVec(y,start[1]+col,&vec_y,ops);
			ops->GetVecFromMultiVec(x,start[0]+col,&vec_x,ops);
			ops->VecAxpby(alpha,vec_x,beta,vec_y,ops);
			ops->RestoreVecForMultiVec(x,start[0]+col,&vec_x,ops);
			ops->RestoreVecForMultiVec(y,start[1]+col,&vec_y,ops);
		}
	}
	return;
}
void DefaultMultiVecLinearComb     (
	void   **x   , void **y , int is_vec, 
	int    *start_xy, int  *end_xy, 
	double *coef , int  ldc , 
	double *beta , int  incb, struct OPS_ *ops)
{
	int i, k, nrows, ncols, start[2], end[2]; 
	double gamma; void *vec_x, *vec_y;
	nrows = end_xy[0]-start_xy[0]; ncols = end_xy[1]-start_xy[1];

	if (x == NULL || coef == NULL) {
	   	for (k = 0; k < ncols; ++k) {
			if (beta == NULL) gamma = 0.0;
			else gamma = *(beta+k*incb);
			if (is_vec == 0) {
				start[0] = start_xy[1]+k; end[0] = start[0]+1;
				start[1] = start_xy[1]+k; end[1] = start[1]+1;
				ops->MultiVecAxpby(0,NULL,gamma,y,start,end,ops);
			}
			else {
				ops->VecAxpby(0,NULL,gamma,*y,ops);
			}
	   	}
	} 
	else {
		if (is_vec == 0) {			
			for (k = 0; k < ncols; ++k) {
				if (beta == NULL) gamma = 0.0;
				else gamma = *(beta+k*incb);
				for (i = 0; i < nrows; ++i) {
					start[0] = start_xy[0]+i; end[0] = start[0]+1;
					start[1] = start_xy[1]+k; end[1] = start[1]+1;
					if (i == 0) {						
						ops->MultiVecAxpby(*(coef+i+k*ldc),x,gamma,y,start,end,ops);
					}
					else {
						ops->MultiVecAxpby(*(coef+i+k*ldc),x,1.0,y,start,end,ops);
					}
				}
			}			
		} 
		else {
			for (k = 0; k < ncols; ++k) {
				if (beta == NULL) gamma = 0.0;
				else gamma = *(beta+k*incb);			
				vec_y = *y;			
				for (i = 0; i < nrows; ++i) {
					ops->GetVecFromMultiVec(x,start_xy[0]+i,&vec_x,ops);
					if (i == 0) {
						ops->VecAxpby(*(coef+i+k*ldc),vec_x,gamma,vec_y,ops);
					}
					else {
						ops->VecAxpby(*(coef+i+k*ldc),vec_x,1.0,vec_y,ops);
					}
					ops->RestoreVecForMultiVec(x,start_xy[0]+i,&vec_x,ops);
				}
				vec_y = NULL;
			}			
		}		
	}
	return;
}
void DefaultMatDotMultiVec      (void *mat, void **x, void **y, 
	int  *start, int *end, struct OPS_ *ops)
{
	int col, ncols = end[1]-start[1]; void *vec_x, *vec_y;
	for (col = 0; col < ncols; ++col) {
		ops->GetVecFromMultiVec(y,start[1]+col,&vec_y,ops);
		ops->GetVecFromMultiVec(x,start[0]+col,&vec_x,ops);
		ops->MatDotVec(mat,vec_x,vec_y,ops);
		ops->RestoreVecForMultiVec(x,start[0]+col,&vec_x,ops);
		ops->RestoreVecForMultiVec(y,start[1]+col,&vec_y,ops);
	}
	return;
}
void DefaultMatTransDotMultiVec (void *mat, void **x, void **y, 
	int  *start, int *end, struct OPS_ *ops)
{
	int col, ncols = end[1]-start[1]; void *vec_x, *vec_y;
	for (col = 0; col < ncols; ++col) {
		ops->GetVecFromMultiVec(y,start[1]+col,&vec_y,ops);
		ops->GetVecFromMultiVec(x,start[0]+col,&vec_x,ops);
		ops->MatTransDotVec(mat,vec_x,vec_y,ops);
		ops->RestoreVecForMultiVec(x,start[0]+col,&vec_x,ops);
		ops->RestoreVecForMultiVec(y,start[1]+col,&vec_y,ops);
	}
	return;
}
void DefaultMultiVecQtAP        (char ntsA, char ntsdQAP, 
	void **mvQ   , void   *matA , void   **mvP, int is_vec, 
	int  *startQP, int    *endQP, double *qAp , int ldQAP , 
	void **mv_ws , struct OPS_ *ops)
{
	int start[2], end[2], nrows, ncols;
	nrows = endQP[0]-startQP[0]; ncols = endQP[1]-startQP[1];
	if (nrows<=0||ncols<=0) return;
	if (matA == NULL) {
		if (ntsdQAP=='T') {
			start[0] = startQP[1]; end[0] = endQP[1];
			start[1] = startQP[0]; end[1] = endQP[0];
			ops->MultiVecInnerProd('N',mvP,mvQ,is_vec,start,end,qAp,ldQAP,ops);
		}
		else {
			ops->MultiVecInnerProd(ntsdQAP,mvQ,mvP,is_vec,startQP,endQP,qAp,ldQAP,ops);
		}
		return;
	}
	start[0] = startQP[1]; end[0] = endQP[1];
	start[1] = 0         ; end[1] = endQP[1]-startQP[1];
	if (is_vec == 0) {
		if (ntsA == 'N' || ntsA == 'S') {
			ops->MatDotMultiVec(matA,mvP,mv_ws,start,end,ops);
		} 
		else if (ntsA == 'T') {
			ops->MatTransDotMultiVec(matA,mvP,mv_ws,start,end,ops);
		}
		//ops->MultiVecView(mv_ws,start[1],end[1],ops);
	} 
	else {
		if (ntsA == 'N' || ntsA == 'S') {
			ops->MatDotVec(matA,mvP,mv_ws,ops);
		} 
		else if (ntsA == 'T') {
			ops->MatTransDotMultiVec(matA,mvP,mv_ws,start,end,ops);
		}
	}
#if DEBUG
	ops->Printf("A*P\n");
	ops->MultiVecView(mv_ws,start[1],end[1],ops);
#endif

	if (ntsdQAP=='T') {
		start[0] = 0         ; end[0] = endQP[1]-startQP[1];
		start[1] = startQP[0]; end[1] = endQP[0]           ;
		ops->MultiVecInnerProd('N',mv_ws,mvQ,is_vec,start,end,qAp,ldQAP,ops);
	}
	else {
		start[0] = startQP[0]; end[0] = endQP[0]           ;
		start[1] = 0         ; end[1] = endQP[1]-startQP[1];
		ops->MultiVecInnerProd(ntsdQAP,mvQ,mv_ws,is_vec,start,end,qAp,ldQAP,ops);
	}
#if DEBUG
	ops->Printf("qAp = %e\n", *qAp);
	ops->Printf("Q\n");
	ops->MultiVecView(mvQ,start[0],end[0],ops);
	ops->Printf("Q*A*P\n");
#endif
	return;
}
