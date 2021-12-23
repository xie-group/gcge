/**
 * i
 *    @file  ops_config.h
 *   @brief  �����ļ�
 *
 *  �����ļ�
 *
 *  @author  Yu Li, liyu@tjufe.edu.cn
 *
 *       Created:  2020/9/13
 *      Revision:  none
 */
#ifndef  _OPS_CONFIG_H_
#define  _OPS_CONFIG_H_

#define  OPS_USE_INTEL_MKL 0
#define  OPS_USE_MPI       1
#define  OPS_USE_OMP       0
/* ��ʾֻ��ӡ0���̵������Ϣ */
#define  PRINT_RANK    0

/* MATLAB �� intel mkl �� blas lapack ��ĺ��������� _ */
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
