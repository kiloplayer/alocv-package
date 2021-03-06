#ifndef WENDA_BLAS_CONFIGURATION_H_INCLUDED
#define WENDA_BLAS_CONFIGURATION_H_INCLUDED

#include "stddef.h"

#ifdef USE_MKL

#define MKL_DIRECT_CALL

#include "mkl.h"
#include "mkl_blas.h"
#include "mkl_lapack.h"

#elif USE_R
#include "R_ext/blas.h"
#include "R_ext/lapack.h"

// R may use F77 convention with additionall underscore at the end.
// Redefine the functions so that we pick up the right names.
#define drot F77_CALL(drot)
#define drotg F77_CALL(drotg)
#define daxpy F77_CALL(daxpy)
#define dscal F77_CALL(dscal)
#define dlacpy F77_CALL(dlacpy)
#define dcopy F77_CALL(dcopy)
#define dgemm F77_CALL(dgemm)
#define dtrsm F77_CALL(dtrsm)
#define ddot F77_CALL(ddot)
#define dgemv F77_CALL(dgemv)
#define dpotrf F77_CALL(dpotrf)

#else
#include "blas.h"
#include "lapack.h"
#endif

#ifdef USE_MKL
inline void* blas_malloc(size_t alignment, size_t size) {
    return mkl_malloc(size, alignment);
}

inline void blas_free(void* ptr) {
    mkl_free(ptr);
}
#elif MATLAB_MEX_FILE
#include "mex.h"

inline void* blas_malloc(size_t alignment, size_t size) {
    return mxMalloc(size);
}

inline void blas_free(void* ptr) {
    mxFree(ptr);
}
#else

#if defined(_WIN32) || defined(_WIN64)

// on windows use platform-specific _aligned_malloc
#include "malloc.h"
inline void* blas_malloc(size_t alignment, size_t size) {
    return _aligned_malloc(size, alignment);
}

inline void blas_free(void* ptr) {
    return _aligned_free(ptr);
}

#else // _WIN32 || _WIN64
#include "stdlib.h"

inline void* blas_malloc(size_t alignment, size_t size) {
    return aligned_alloc(alignment, alignment * (size + alignment - 1) / alignment);
}

inline void blas_free(void* ptr) {
    free(ptr);
}
#endif // _WIN32 || _WIN64

#endif

#endif
