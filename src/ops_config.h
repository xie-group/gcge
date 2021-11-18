/**
 *
 *    @file  ops_config.h
 *   @brief  配置文件
 *
 *  配置文件
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/9/13
 *      Revision:  none
 */
#ifndef  _OPS_CONFIG_H_
#define  _OPS_CONFIG_H_

/* 当OPS_USE_SLEPC或PHG或HYPRE为1时, 需要让OPS_USE_MPI为1 */
/* TODO 当 OPS_INTEL_MKL 为 1 时, CCS 接口需要加速 矩阵乘以向量 */
#define  OPS_USE_INTEL_MKL 0
#define  OPS_USE_MPI       1
#define  OPS_USE_MUMPS     0
#define  OPS_USE_OMP       0
#define  OPS_USE_PETSC     1
#define  OPS_USE_SLEPC     1
/* 表示只打印0进程的输出信息 */
#define  PRINT_RANK    0

/* MATLAB 和 intel mkl 中 blas lapack 库的函数名不加 _ */
#if OPS_USE_MATLAB || OPS_USE_INTEL_MKL
#define FORTRAN_WRAPPER(x) x
#else
#define FORTRAN_WRAPPER(x) x ## _
#endif

#if OPS_USE_OMP
#define OMP_NUM_THREADS 2
#endif

//#if OPS_USE_INTEL_MKL
//#define MKL_NUM_THREADS 16
//#endif


#endif  /* -- #ifndef _OPS_CONFIG_H_ -- */
