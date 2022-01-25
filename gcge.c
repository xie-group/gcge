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
    PetscBool            autoshift;
    PetscBool            print;
    PetscBool            printtime;
    char                 *orthmethod;
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
    gcge->tol_gcg[0] = 1e-1;
    gcge->tol_gcg[1] = 1e-8;
    if (eps->converged == EPSConvergedRelative) {
        if (eps->tol > 0) {
            gcge->tol_gcg[1] = eps->tol;
        }
    }
    else if (eps->converged == EPSConvergedAbsolute) {
        if (eps->tol > 0) {
            gcge->tol_gcg[0] = eps->tol;
        }
    }
    eps->ncv = 2*eps->nev;
    eps->mpd = eps->ncv;
    if (eps->max_it==PETSC_DEFAULT) eps->max_it = PETSC_MAX_INT;
    gcge->nevConv = eps->nev;
    gcge->block_size = (gcge->nevConv)>20?((PetscInt)((gcge->nevConv)/3)):((PetscInt)((gcge->nevConv)/2));
    gcge->nevInit = 3*(gcge->block_size);
    gcge->nevMax = (gcge->nevInit)+(gcge->nevConv);
    if (eps->nev<6) {
        gcge->block_size = gcge->nevConv;
        gcge->nevInit = 2*gcge->nevConv;
        gcge->nevMax = gcge->nevInit;
    }
    gcge->gapMin = 1e-5;
    gcge->max_iter_gcg = eps->max_it;

    if (!eps->V) { ierr = EPSGetBV(eps,&eps->V);CHKERRQ(ierr); }
    ierr = EPSAllocateSolution(eps,0);CHKERRQ(ierr);
    PetscFunctionReturn(0);

}

PetscErrorCode EPSSolve_GCGE(EPS eps)
{   
    PetscErrorCode 	ierr;
    EPS_GCGE     	*gcge = (EPS_GCGE*)eps->data;
    PetscInt 		nevMax, nevInit, nevConv, block_size, nevGiven, max_iter_gcg, flag, M ,N;
    PetscBool       print,printtime;
    PetscReal       gapMin, tol_gcg[2];
    PetscMPIInt     size,rank;
    PetscScalar	    *a, *b;
    PetscBool       shift;

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
    shift       = gcge->autoshift;
    print       = gcge->print;
    printtime   = gcge->printtime;
    OPS *slepc_ops;
    slepc_ops = NULL;
    PetscFunctionBegin;
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)eps),&size);CHKERRMPI(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)eps),&rank);CHKERRMPI(ierr);
    OPS_Create (&slepc_ops);
    slepc_ops->GetOptionFromCommandLine = GetOptionFromCommandLine_GCGE;
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
    ierr = EPSGetOperators(eps,&slepc_matA,&slepc_matB);CHKERRQ(ierr);
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
    int i;
    ierr = BVGetArray(eps->V,&a);CHKERRQ(ierr);
    if (eps->numbermonitors>0) {
        print = 1;
    }
    EigenSolverSetup_GCG(gapMin,nevInit,nevMax,block_size,
            tol_gcg,max_iter_gcg,print,printtime,flag,gcg_mv_ws,dbl_ws,int_ws,ops);
    GCGE_Setparameters(gapMin,shift,ops);
    ops->EigenSolver(A,B,eval,evec,nevGiven,&nevConv,ops);
    ierr = BVGetArray((BV)evec,&b);CHKERRQ(ierr);
    PetscInt pnev = N/size;
    for (i=0;i<(pnev+1)*nevConv;++i) {
        a[i] = b[i];
    }
    ierr = BVRestoreArray(eps->V,&a);CHKERRQ(ierr);
    ierr = BVRestoreArray((BV)evec,&b);CHKERRQ(ierr);
    eps->nconv = nevConv;
    eps->reason = eps->nconv >= eps->nev ? EPS_CONVERGED_TOL : EPS_DIVERGED_ITS;
    for (i=0;i<eps->nconv;i++) eps->eigr[i] = eval[i];
    eps->its = ((GCGSolver*)ops->eigen_solver_workspace)->numIter+1;
    if (eps->nconv < eps->nev) {
        eps->its = ((GCGSolver*)ops->eigen_solver_workspace)->numIter;
    }
    GCGE_Destroymvws(gcg_mv_ws, dbl_ws, int_ws, nevMax, block_size,ops);
    OPS_Destroy (&slepc_ops);
    PetscFunctionReturn(0);
}

PetscErrorCode EPSDestroy_GCGE(EPS eps)
{
    PetscErrorCode ierr;

    PetscFunctionBegin;
    ierr = PetscFree(eps->data);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGCGESetShift_C",NULL);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGCGEGetShift_C",NULL);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGCGESetPrint_C",NULL);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGCGEGetPrint_C",NULL);CHKERRQ(ierr);    
    PetscFunctionReturn(0);
}

PetscErrorCode EPSSetFromOptions_GCGE(PetscOptionItems *PetscOptionsObject,EPS eps)
{
    PetscErrorCode  ierr;
    EPS_GCGE        *ctx = (EPS_GCGE*)eps->data;
    PetscBool        shift, printtime;
    PetscBool       flg;
    PetscFunctionBegin;
    ierr = PetscOptionsHead(PetscOptionsObject,"EPS GCGE Options");CHKERRQ(ierr);
    ierr = PetscOptionsBool("-eps_gcge_autoshift","autoshift for bcg","EPSGCGESetShift",ctx->autoshift,&shift,&flg);CHKERRQ(ierr);
    if (flg) { ierr = EPSGCGESetShift(eps,shift);CHKERRQ(ierr); }
    ierr = PetscOptionsBool("-eps_gcge_print_parttime","print time of each step of gcge","EPSGCGESetPrint",ctx->printtime,&printtime,&flg);CHKERRQ(ierr);
    if (flg) { ierr = EPSGCGESetPrint(eps,printtime);CHKERRQ(ierr); }
    ierr = PetscOptionsTail();CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
/*
   set shift
 */
static PetscErrorCode EPSGCGESetShift_GCGE(EPS eps,PetscBool shift)
{
    EPS_GCGE *gcge = (EPS_GCGE*)eps->data;
    PetscFunctionBegin;
    gcge->autoshift = shift;
    PetscFunctionReturn(0);
}

PetscErrorCode EPSGCGESetShift(EPS eps,PetscBool shift)
{
    PetscErrorCode ierr;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
    PetscValidLogicalCollectiveInt(eps,shift,2);
    ierr = PetscTryMethod(eps,"EPSGCGESetShift_C",(EPS,PetscBool),(eps,shift));CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
static PetscErrorCode EPSGCGEGetShift_GCGE(EPS eps,PetscBool *shift)
{
    EPS_GCGE *gcge = (EPS_GCGE*)eps->data;

    PetscFunctionBegin;
    *shift = gcge->autoshift;
    PetscFunctionReturn(0);
}
PetscErrorCode EPSGCGEGetShift(EPS eps,PetscBool *shift)
{
    PetscErrorCode ierr;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
    PetscValidIntPointer(shift,2);
    ierr = PetscUseMethod(eps,"EPSGCGEGetShift_C",(EPS,PetscBool*),(eps,shift));CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

/* 
   set print
 */
static PetscErrorCode EPSGCGESetPrint_GCGE(EPS eps,PetscBool printtime)
{
    EPS_GCGE *gcge = (EPS_GCGE*)eps->data;

    PetscFunctionBegin;
    gcge->printtime = printtime;
    PetscFunctionReturn(0);
}

PetscErrorCode EPSGCGESetPrint(EPS eps,PetscBool printtime)
{
    PetscErrorCode ierr;

    PetscFunctionBegin;
    PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
    PetscValidLogicalCollectiveInt(eps,printtime,2);
    ierr = PetscTryMethod(eps,"EPSGCGESetPrint_C",(EPS,PetscBool),(eps,printtime));CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
static PetscErrorCode EPSGCGEGetPrint_GCGE(EPS eps,PetscBool *printtime)
{
    EPS_GCGE *gcge = (EPS_GCGE*)eps->data;

    PetscFunctionBegin;
    *printtime = gcge->printtime;
    PetscFunctionReturn(0);
}
PetscErrorCode EPSGCGEGetPrint(EPS eps,PetscBool *printtime)
{
    PetscErrorCode ierr;
    PetscFunctionBegin;
    PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
    PetscValidIntPointer(printtime,2);
    ierr = PetscUseMethod(eps,"EPSGCGEGetPrint_C",(EPS,PetscBool*),(eps,printtime));CHKERRQ(ierr);
    PetscFunctionReturn(0);
}
PetscErrorCode EPSView_GCGE(EPS eps,PetscViewer viewer)
{
    PetscErrorCode ierr;
    PetscBool      isascii;
    EPS_GCGE       *ctx = (EPS_GCGE*)eps->data;
    PetscFunctionBegin;
    ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
    if (isascii) {
        ierr = PetscViewerASCIIPrintf(viewer,"  orthogonalization = b%s (Block Modified Gram-Schmidt)\n",ctx->orthmethod);CHKERRQ(ierr);
        if (ctx->autoshift==1)
        {
            ierr = PetscViewerASCIIPrintf(viewer,"  shift = gcge auto shift\n");CHKERRQ(ierr);
        }
    }
    PetscFunctionReturn(0);
}

PetscErrorCode EPSReset_GCGE(EPS eps)
{
    PetscFunctionBegin;
    PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode EPSCreate_GCGE(EPS eps)
{
    EPS_GCGE       *gcge;
    PetscErrorCode ierr;
    PetscFunctionBegin;
    ierr = PetscNewLog(eps,&gcge);CHKERRQ(ierr);

    eps->data = (void*)gcge;
    eps->categ = EPS_CATEGORY_OTHER;
    gcge->autoshift = 1;
    gcge->print = 0;
    gcge->printtime = 0;
    gcge->orthmethod = "mgs";
    eps->ops->solve          = EPSSolve_GCGE;
    eps->ops->setup          = EPSSetUp_GCGE;
    eps->ops->setupsort      = EPSSetUpSort_Basic; 
    eps->ops->setfromoptions = EPSSetFromOptions_GCGE;
    eps->ops->destroy        = EPSDestroy_GCGE;
    eps->ops->reset          = EPSReset_GCGE;
    eps->ops->view           = EPSView_GCGE;
    eps->ops->backtransform  = EPSBackTransform_Default;
    eps->ops->setdefaultst   = EPSSetDefaultST_NoFactor;

    ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGCGESetShift_C",EPSGCGESetShift_GCGE);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGCGEGetShift_C",EPSGCGEGetShift_GCGE);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGCGESetPrint_C",EPSGCGESetPrint_GCGE);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSGCGEGetPrint_C",EPSGCGEGetPrint_GCGE);CHKERRQ(ierr);

    PetscFunctionReturn(0);

}


