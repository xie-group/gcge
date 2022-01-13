#include <slepc/private/epsimpl.h> 
#include "gcge.h"
#include <assert.h>
#include "ops_eig_sol_gcg.h"
#include "ops.h"
#include <slepceps.h>

typedef struct {
  OPS                  *gcgeops;   
  PetscInt             block_size;
  PetscInt             nevConv;
  PetscInt             nevInit; 
  PetscInt             nevMax; 
  PetscInt             nevGiven;
  PetscInt             max_iter_gcg;   
  PetscReal            tol_gcg[2];
  PetscReal            gapMin;
} EPS_GCGE;

/* multi-vec */
static void MultiVecCreateByMat (BV *des_bv, int num_vec, Mat src_mat, struct OPS_ *ops)
{
	Vec vector;
    MatCreateVecs(src_mat,NULL,&vector);
    BVCreate(PETSC_COMM_WORLD, des_bv);
    BVSetType(*des_bv,BVMAT);
    BVSetSizesFromVec(*des_bv,vector,num_vec);
    VecDestroy(&vector);
	BVSetActiveColumns(*des_bv,0,num_vec);
	BVSetRandom(*des_bv);
	return;
}
static void MultiVecDestroy (BV *des_bv, int num_vec, struct OPS_ *ops)
{ 
	BVDestroy(des_bv);
	return;
}
static void MultiVecView (BV x, int start, int end, struct OPS_ *ops)
{
	BVSetActiveColumns(x,start,end);
	BVView(x,PETSC_VIEWER_STDOUT_WORLD);
	return;
}
static void MultiVecLocalInnerProd (char nsdIP, 
		BV x, BV y, int is_vec, int *start, int *end, 
		double *inner_prod, int ldIP, struct OPS_ *ops)
{
	assert(is_vec==0);
	
    const PetscScalar *x_array, *y_array;
    int x_nrows, x_ncols, y_nrows, y_ncols;
    BVGetArrayRead(x,&x_array);
    BVGetSizes(x,&x_nrows,NULL,&x_ncols);
    if(is_vec == 0) {
		BVGetArrayRead(y,&y_array);
		BVGetSizes(y,&y_nrows,NULL,&y_ncols);
		LAPACKVEC x_vec, y_vec;
		x_vec.nrows = x_nrows; y_vec.nrows = y_nrows;
		x_vec.ncols = x_ncols; y_vec.ncols = y_ncols;
		x_vec.ldd   = x_nrows; y_vec.ldd   = y_nrows;
		x_vec.data  = (double *)x_array; 
		y_vec.data  = (double *)y_array;
		ops->lapack_ops->MultiVecLocalInnerProd(nsdIP,
				(void**)&x_vec,(void**)&y_vec,is_vec,
				start,end,inner_prod,ldIP,ops->lapack_ops);
		BVRestoreArrayRead(y, &y_array);
    }
    BVRestoreArrayRead(x,&x_array);
	return;
}
static void MultiVecSetRandomValue (BV x, int start, int end, struct OPS_ *ops)
{
	BVSetActiveColumns(x,start,end);
	BVSetRandom(x);
	return;
}
static void MultiVecAxpby (double alpha, BV x, 
		double beta, BV y, int *start, int *end, struct OPS_ *ops)
{
	assert(end[0]-start[0]==end[1]-start[1]);
    PetscScalar *y_array;
    int x_nrows, x_ncols, y_nrows, y_ncols;

    BVGetArray(y,&y_array);
    BVGetSizes(y,&y_nrows,NULL,&y_ncols);
    LAPACKVEC y_vec;
    y_vec.nrows = y_nrows; y_vec.ncols = y_ncols;
    y_vec.ldd   = y_nrows; y_vec.data  = y_array;
    if (x==NULL) {
#if 0
       ops->lapack_ops->MultiVecAxpby(alpha,
	     NULL,beta,(void**)&y_vec,start,end,ops->lapack_ops);
#else
       BVSetActiveColumns(y,start[1],end[1]);
       BVScale(y,beta);
#endif
    }
	else {
		if (x!=y) {
			BVSetActiveColumns(x,start[0],end[0]);
			BVSetActiveColumns(y,start[1],end[1]);
			BVMult(y,alpha,beta,x,NULL);
		}
		else if (start[0]==start[1]) {
			BVSetActiveColumns(y,start[1],end[1]);
			BVScale(y,(alpha+beta));
		}
		else {
			assert(end[0]<=start[1]||end[1]<=start[0]);
			const PetscScalar *x_array;
			LAPACKVEC x_vec;
			BVGetArrayRead(x,&x_array);
			BVGetSizes(x,&x_nrows,NULL,&x_ncols);
			x_vec.nrows = x_nrows;
			x_vec.ncols = x_ncols;
			x_vec.ldd   = x_nrows;
			x_vec.data  = (double *)x_array; 
			ops->lapack_ops->MultiVecAxpby(alpha,
					(void**)&x_vec,beta,(void**)&y_vec,start,end,ops->lapack_ops);
			BVRestoreArrayRead(x, &x_array);
		}
	}
    BVRestoreArray(y, &y_array);

    return;
}
static void MatDotMultiVec (Mat mat, BV x, 
		BV y, int *start, int *end, struct OPS_ *ops)
{
      
	assert(end[0]-start[0]==end[1]-start[1]);
	int nrows_x, nrows_y;
	BVGetSizes(x, &nrows_x, NULL, NULL);
	BVGetSizes(y, &nrows_y, NULL, NULL);

	if (nrows_x==nrows_y) {
		if (mat==NULL) {
			MultiVecAxpby(1.0, x, 0.0, y, start, end, ops);
		}
		else {
			/* sometimes Active does not work */
			assert(x!=y);
			if (end[0]-start[0] < 5) {
				int ncols = end[1]-start[1], col;
				Vec vec_x, vec_y;      
				for (col = 0; col < ncols; ++col) {
					BVGetColumn(x, start[0]+col, &vec_x);
					BVGetColumn(y, start[1]+col, &vec_y);
					MatMult(mat, vec_x, vec_y); 
					BVRestoreColumn(x, start[0]+col, &vec_x);
					BVRestoreColumn(y, start[1]+col, &vec_y);
				}
			}
			else {
				BVSetActiveColumns(x,start[0],end[0]);
				BVSetActiveColumns(y,start[1],end[1]);
				BVMatMult(x,mat,y);
			}
		}
	}
	else {
		assert(mat!=NULL);
		Vec vec_x, vec_y;      
		int ncols = end[1]-start[1], col;
		for (col = 0; col < ncols; ++col) {
			BVGetColumn(x, start[0]+col, &vec_x);
			BVGetColumn(y, start[1]+col, &vec_y);
			MatMult(mat, vec_x, vec_y); 
			BVRestoreColumn(x, start[0]+col, &vec_x);
			BVRestoreColumn(y, start[1]+col, &vec_y);
		}
	}
	return;
}
static void MatTransDotMultiVec (Mat mat, BV x, 
		BV y, int *start, int *end, struct OPS_ *ops)
{
	assert(end[0]-start[0]==end[1]-start[1]);
	int nrows_x, nrows_y;
	BVGetSizes(x, &nrows_x, NULL, NULL);
	BVGetSizes(y, &nrows_y, NULL, NULL);
	if (nrows_x==nrows_y) {
		if (mat==NULL) {
			MultiVecAxpby(1.0, x, 0.0, y, start, end, ops);
		}
		else {
			BVSetActiveColumns(x,start[0],end[0]);
			BVSetActiveColumns(y,start[1],end[1]);
			BVMatMultTranspose(x,mat,y);
		}
	}
	else {
		Vec vec_x, vec_y;
		assert(end[0]-start[0]==end[1]-start[1]);
		int ncols = end[1]-start[1], col;
		for (col = 0; col < ncols; ++col) {
			BVGetColumn(x, start[0]+col, &vec_x);
			BVGetColumn(y, start[1]+col, &vec_y);
			MatMultTranspose(mat, vec_x, vec_y); 
			BVRestoreColumn(x, start[0]+col, &vec_x);
			BVRestoreColumn(y, start[1]+col, &vec_y);
		}
	}
	return;
}
static void MultiVecLinearComb (BV x, BV y, int is_vec, 
		int    *start, int *end, 
		double *coef , int ldc , 
		double *beta , int incb, struct OPS_ *ops)
{
    assert(is_vec==0);
    PetscScalar *y_array;
    int x_nrows, x_ncols, y_nrows, y_ncols;

    BVGetArray(y,&y_array);
    BVGetSizes(y,&y_nrows,NULL,&y_ncols);
    LAPACKVEC y_vec;
    y_vec.nrows = y_nrows;
    y_vec.ncols = y_ncols;
    y_vec.ldd   = y_nrows;
    y_vec.data  = (double *)y_array;
    if (x==NULL) {
		ops->lapack_ops->MultiVecLinearComb(
			NULL,(void**)&y_vec,is_vec,
			start,end,coef,ldc,beta,incb,ops->lapack_ops);
    }
    else {
       //assert(end[0]<=start[1]||end[1]<=start[0]);
		const PetscScalar *x_array;
		LAPACKVEC x_vec;
		BVGetArrayRead(x,&x_array);
		BVGetSizes(x,&x_nrows,NULL,&x_ncols);
		x_vec.nrows = x_nrows; 
		x_vec.ncols = x_ncols; 
		x_vec.ldd   = x_nrows; 
		x_vec.data  = (double*)x_array; 
		ops->lapack_ops->MultiVecLinearComb(
			(void**)&x_vec,(void**)&y_vec,is_vec,
			start,end,coef,ldc,beta,incb,ops->lapack_ops);
		BVRestoreArrayRead(x, &x_array);
    }
    BVRestoreArray(y, &y_array);
    return;
}
/* Encapsulation */
static void MatView_GCGE (void *mat, struct OPS_ *ops)
{
	MatView((Mat)mat,PETSC_VIEWER_STDOUT_WORLD);
	return;
}
static void MatAxpby_GCGE (double alpha, void *matX, 
		double beta, void *matY, struct OPS_ *ops)
{
	/* y = alpha x + beta y */
	if (beta == 1.0) {
		/* SAME_NONZERO_PATTERN, DIFFERENT_NONZERO_PATTERN or SUBSET_NONZERO_PATTERN */
		/* y = alpha x + y */
		MatAXPY((Mat)matY,alpha,(Mat)matX,SUBSET_NONZERO_PATTERN);
	}
	else if (alpha == 1.0) {
		/* y = x + beta y */
		MatAYPX((Mat)matY,beta,(Mat)matX,SUBSET_NONZERO_PATTERN);
	}
	else {
		if (beta == 0.0) {
			MatCopy((Mat)matX,(Mat)matY,DIFFERENT_NONZERO_PATTERN);
			MatScale((Mat)matY,alpha);
		}
		else {
			MatAXPY((Mat)matY,(alpha-1.0)/beta,(Mat)matX,SUBSET_NONZERO_PATTERN);
			MatAYPX((Mat)matY,beta,(Mat)matX,SUBSET_NONZERO_PATTERN);
		}
	}
	return;
}
/* multi-vec */
static void MultiVecCreateByMat_GCGE (void ***des_vec, int num_vec, void *src_mat, struct OPS_ *ops)
{
	MultiVecCreateByMat ((BV*)des_vec,num_vec,(Mat)src_mat,ops);		
	return;
}
static void MultiVecDestroy_GCGE (void ***des_vec, int num_vec, struct OPS_ *ops)
{
	MultiVecDestroy ((BV*)des_vec,num_vec,ops);
	return;
}
static void MultiVecView_GCGE (void **x, int start, int end, struct OPS_ *ops)
{
	MultiVecView ((BV)x,start,end,ops);
	return;
}
static void MultiVecLocalInnerProd_GCGE (char nsdIP, 
		void **x, void **y, int is_vec, int *start, int *end, 
		double *inner_prod, int ldIP, struct OPS_ *ops)
{
	MultiVecLocalInnerProd (nsdIP, 
			(BV)x,(BV)y,is_vec,start,end, 
			inner_prod,ldIP,ops);
	return;
}
static void MultiVecSetRandomValue_GCGE (void **x, int start, int end, struct OPS_ *ops)
{
	MultiVecSetRandomValue ((BV)x,start,end,ops);
	return;
}
static void MultiVecAxpby_GCGE (double alpha, void **x, 
		double beta, void **y, int *start, int *end, struct OPS_ *ops)
{
	MultiVecAxpby (alpha,(BV)x,beta,(BV)y,start,end,ops);
	return;
}
static void MatDotMultiVec_GCGE (void *mat, void **x, 
		void **y, int *start, int *end, struct OPS_ *ops)
{
	MatDotMultiVec ((Mat)mat,(BV)x,(BV)y,start,end,ops);
	return;
}
static void MatTransDotMultiVec_GCGE (void *mat, void **x, 
		void **y, int *start, int *end, struct OPS_ *ops)
{
	MatTransDotMultiVec ((Mat)mat,(BV)x,(BV)y,start,end,ops);
	return;
}
static void MultiVecLinearComb_GCGE (
		void **x , void **y, int is_vec, 
		int    *start, int  *end, 
		double *coef , int  ldc , 
		double *beta , int  incb, struct OPS_ *ops)
{
        //assert(x!=y);
	MultiVecLinearComb (
			(BV)x, (BV)y, is_vec, 
			start, end , 
			coef , ldc , 
			beta , incb, ops);
	return;
}

static int GetOptionFromCommandLine_GCGE (
		const char *name, char type, void *value,
		int argc, char* argv[], struct OPS_ *ops)
{
	PetscBool set;
	int *int_value; double *dbl_value; char *str_value; 
	switch (type) {
		case 'i':
			int_value = (int*)value; 
			PetscOptionsGetInt(NULL, NULL, name, int_value, &set);
			break;
		case 'f':
			dbl_value = (double*)value; 
			PetscOptionsGetReal(NULL, NULL, name, dbl_value, &set);
			break;
		case 's':
			str_value = (char*) value;
			PetscOptionsGetString(NULL, NULL, name, str_value, 8, &set);
			//set = DefaultGetOptionFromCommandLine(name, type, value, argc, argv, ops);
			break;
			default:
		break;
	}	        
	return set;
}


PetscErrorCode EPSSetUp_GCGE(EPS eps)
{
    PetscErrorCode ierr;
    PetscBool      isshift;
    EPS_GCGE     	*gcge = (EPS_GCGE*)eps->data;
    PetscFunctionBegin;
    EPSCheckHermitianDefinite(eps);
    ierr = PetscObjectTypeCompare((PetscObject)eps->st,STSHIFT,&isshift);CHKERRQ(ierr);
    if (!isshift) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver does not support spectral transformations");
    if (!eps->which) eps->which = EPS_SMALLEST_REAL;
    EPSCheckUnsupported(eps,EPS_FEATURE_BALANCE | EPS_FEATURE_ARBITRARY | EPS_FEATURE_REGION | EPS_FEATURE_STOPPING);
    EPSCheckIgnored(eps,EPS_FEATURE_EXTRACTION | EPS_FEATURE_CONVERGENCE);

    eps->ncv = gcge->nevMax;
    if (!eps->V) { ierr = EPSGetBV(eps,&eps->V);CHKERRQ(ierr); }
    ierr = EPSAllocateSolution(eps,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);

}

PetscErrorCode EPSSolve_GCGE(EPS eps)
{   
    PetscErrorCode 	ierr;
    EPS_GCGE     	*gcge = (EPS_GCGE*)eps->data;
    PetscInt 		nevMax, nevInit, nevConv, block_size, nevGiven, max_iter_gcg, flag, M ,N;
    PetscReal       gapMin, tol_gcg[2];
    PetscMPIInt     size,rank;
    PetscScalar	    *a, *b;

    tol_gcg[0] = gcge->tol_gcg[0];
    tol_gcg[1] = gcge->tol_gcg[1];
    nevInit 	= gcge->nevInit;
    nevConv 	= gcge->nevConv;
    nevMax  	= gcge->nevMax;
    nevGiven 	= gcge->nevGiven;
    max_iter_gcg= gcge->max_iter_gcg;
    block_size  = gcge->block_size;
    flag		= 0;
    gapMin 		= gcge->gapMin;
    OPS *slepc_ops;
    slepc_ops = NULL;
    PetscFunctionBegin;
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)eps),&size);CHKERRMPI(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)eps),&rank);CHKERRMPI(ierr);
    OPS_Create (&slepc_ops);
    slepc_ops->GetOptionFromCommandLine = GetOptionFromCommandLine_GCGE;
    slepc_ops->MatAxpby               = MatAxpby_GCGE;
    slepc_ops->MatView                = MatView_GCGE;
    slepc_ops->MultiVecCreateByMat    = MultiVecCreateByMat_GCGE   ;
    slepc_ops->MultiVecDestroy        = MultiVecDestroy_GCGE       ;
    slepc_ops->MultiVecView           = MultiVecView_GCGE          ;
    slepc_ops->MultiVecLocalInnerProd = MultiVecLocalInnerProd_GCGE;
    slepc_ops->MultiVecSetRandomValue = MultiVecSetRandomValue_GCGE;
    slepc_ops->MultiVecAxpby          = MultiVecAxpby_GCGE         ;
    slepc_ops->MatDotMultiVec         = MatDotMultiVec_GCGE        ;
    slepc_ops->MatTransDotMultiVec    = MatTransDotMultiVec_GCGE   ;
    slepc_ops->MultiVecLinearComb     = MultiVecLinearComb_GCGE    ;
    OPS_Setup (slepc_ops);
    void *A, *B; OPS *ops;
    Mat      slepc_matA, slepc_matB;
    ierr = STGetMatrix(eps->st,0,&slepc_matA);CHKERRQ(ierr);
    ierr = STGetMatrix(eps->st,1,&slepc_matB);CHKERRQ(ierr);
    //ierr = EPSGetOperators(eps,&slepc_matA,&slepc_matB);CHKERRQ(ierr);
    ierr = MatGetSize(slepc_matA,&M,&N);
    ops = slepc_ops; A = (void*)(slepc_matA); B = (void*)(slepc_matB);
    double *eval; 
    void **evec;
    eval = malloc(nevMax*sizeof(double));
    memset(eval,0,nevMax*sizeof(double));
    ops->MultiVecCreateByMat(&evec,nevMax,A,ops);
    ops->MultiVecSetRandomValue(evec,0,nevMax,ops);
    void **gcg_mv_ws[4]; double *dbl_ws = NULL; int *int_ws = NULL;
    GCGE_Create(A, nevMax, block_size, nevInit, gcg_mv_ws, dbl_ws, int_ws, ops);
    int sizeV = nevInit + 2*block_size;
    int length_dbl_ws = 2*sizeV*sizeV+10*sizeV
        +(nevMax+2*block_size)+(nevMax)*block_size;
    int length_int_ws = 6*sizeV+2*(block_size+3);
    dbl_ws = malloc(length_dbl_ws*sizeof(double));
    memset(dbl_ws,0,length_dbl_ws*sizeof(double));
    int_ws = malloc(length_int_ws*sizeof(int));
    memset(int_ws,0,length_int_ws*sizeof(int));
    srand(0);
    double time_start, time_interval;
    time_start = ops->GetWtime();
    ops->Printf("===============================================\n");
    ops->Printf("GCG Eigen Solver\n");
    int i;
    ierr = BVGetArray(eps->V,&a);CHKERRQ(ierr);
    EigenSolverSetup_GCG(gapMin,nevInit,nevMax,block_size,
            tol_gcg,max_iter_gcg,flag,gcg_mv_ws,dbl_ws,int_ws,ops);
    GCGE_Setparameters(gapMin,ops);
    ops->EigenSolver(A,B,eval,evec,nevGiven,&nevConv,ops);
    ierr = BVGetArray((BV)evec,&b);CHKERRQ(ierr);
    PetscInt pnev = (PetscInt)nevConv/size+1;
    for (i=0;i<pnev*N;++i) {
        a[i] = b[i];
    }
    ierr = BVRestoreArray(eps->V,&a);CHKERRQ(ierr);
    ierr = BVRestoreArray((BV)evec,&b);CHKERRQ(ierr);
    ops->Printf("numIter = %d, nevConv = %d\n",
            ((GCGSolver*)ops->eigen_solver_workspace)->numIter, nevConv);
    //eps->nev = nevConv;
    eps->nconv = ((GCGSolver*)ops->eigen_solver_workspace)->nevConv;
    eps->reason = EPS_CONVERGED_TOL;
    for (i=0;i<eps->nconv;i++) eps->eigr[i] = eval[i];
    eps->its = ((GCGSolver*)ops->eigen_solver_workspace)->numIter;
    ops->Printf("++++++++++++++++++++++++++++++++++++++++++++++\n");
    time_interval = ops->GetWtime() - time_start;
    ops->Printf("Time is %.3f\n", time_interval);
    GCGE_Destroymvws(gcg_mv_ws, dbl_ws, int_ws, nevMax, block_size,ops);
    ops->Printf("eigenvalues\n");
    int idx;
    for (idx = 0; idx < nevConv; ++idx) {
        ops->Printf("%d: %6.14e\n",idx+1,eval[idx]);
    }
    OPS_Destroy (&slepc_ops);
    PetscFunctionReturn(0);
}

PetscErrorCode EPSDestroy_GCGE(EPS eps)
{
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscFree(eps->data);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode EPSSetFromOptions_GCGE(PetscOptionItems *PetscOptionsObject,EPS eps)
{
    PetscErrorCode  ierr;

    PetscFunctionBegin;

    ierr = PetscOptionsTail();CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

PetscErrorCode EPSView_GCGE(EPS eps,PetscViewer viewer)
{
    PetscFunctionBegin;
    PetscFunctionReturn(0);
}

PetscErrorCode EPSReset_GCGE(EPS eps)
{
    PetscFunctionBegin;
    PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode EPSCreate_GCGE(EPS eps)
{
    EPS_GCGE       *ctx;
    PetscErrorCode ierr;
    PetscFunctionBegin;
    ierr = PetscNewLog(eps,&ctx);CHKERRQ(ierr);
    eps->data = (void*)ctx;
    eps->nev = 2;//Default
    ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);
    ctx->nevConv = eps->nev;
    ctx->block_size = (ctx->nevConv)>20?((PetscInt)((ctx->nevConv)/3)):((PetscInt)((ctx->nevConv)/2));
    ctx->nevInit = 3*(ctx->block_size);
    ctx->nevMax = (ctx->nevInit)+(ctx->nevConv);
    ctx->gapMin = 1e-5;
    ctx->tol_gcg[0] = 1e-1;
    ctx->tol_gcg[1] = 1e-8;
    ctx->max_iter_gcg = 500;

    eps->categ = EPS_CATEGORY_OTHER;

    eps->ops->solve          = EPSSolve_GCGE;
    eps->ops->setup          = EPSSetUp_GCGE;
    eps->ops->setupsort      = EPSSetUpSort_Basic; 
    eps->ops->setfromoptions = EPSSetFromOptions_GCGE;
    eps->ops->destroy        = EPSDestroy_GCGE;
    eps->ops->reset          = EPSReset_GCGE;
    eps->ops->view           = EPSView_GCGE;
    eps->ops->backtransform  = EPSBackTransform_Default;
    eps->ops->setdefaultst   = EPSSetDefaultST_NoFactor;
    PetscFunctionReturn(0);

}


