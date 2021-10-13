/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_READER_C_H
#define GRB_FUSION_READER_C_H

#include <mio/mio.h>
#include <stdlib.h>
#include <GraphBLAS.h>

#define DEFINE_READ_MATRIX(NAME, TYPE)                                          \
GrB_Info read_Matrix_##NAME(GrB_Matrix *A, const char *path, bool shuffle) {    \
    int nrows, ncols, nz;                                                       \
    int *I, *J;                                                                 \
    TYPE *X;                                                                    \
                                                                                \
    GrB_Index nrows_grb, ncols_grb, nz_grb;                                     \
    GrB_Index *I_grb, *J_grb;                                                   \
    TYPE *X_grb;                                                                \
                                                                                \
    mio_read_tuples_##NAME(path, &nrows, &ncols, &nz, &I, &J, &X, shuffle);     \
                                                                                \
    /* Convert read values to GraphBLAS types */                                \
    nrows_grb = nrows;                                                          \
    ncols_grb = ncols;                                                          \
    nz_grb = nz;                                                                \
    I_grb = (GrB_Index *) malloc(nz * sizeof(GrB_Index));                       \
    J_grb = (GrB_Index *) malloc(nz * sizeof(GrB_Index));                       \
    X_grb = (TYPE*) malloc(nz * sizeof(TYPE));                                  \
                                                                                \
    for (int i = 0;i < nz; i++) {                                               \
        I_grb[i] = I[i];                                                        \
        J_grb[i] = J[i];                                                        \
        X_grb[i] = X[i];                                                        \
    }                                                                           \
                                                                                \
    /* Create the matrix */                                                     \
    GrB_Info info;                                                              \
    GrB_Type xtype = GrB_##NAME;                                                \
    GrB_BinaryOp xop = GrB_PLUS_##NAME; /* copied from read_matrix.c */         \
    GrB_Matrix C;                                                               \
    info = GrB_Matrix_new(&C, xtype, nrows_grb, ncols_grb);                     \
    if (info != GrB_SUCCESS && info != GrB_NO_VALUE) {                          \
        return info;                                                            \
    }                                                                           \
                                                                                \
    info = GrB_Matrix_build_##NAME(C, I_grb, J_grb, X_grb, nz_grb, xop);        \
    if (info != GrB_SUCCESS && info != GrB_NO_VALUE) {                          \
        return info;                                                            \
    }                                                                           \
                                                                                \
    *A = C;                                                                     \
                                                                                \
    mio_free_tuples(I, J, X);                                                   \
                                                                                \
    free(I_grb);                                                                \
    free(J_grb);                                                                \
    free(X_grb);                                                                \
                                                                                \
    return GrB_SUCCESS;                                                         \
}

/* @formatter:off */
DEFINE_READ_MATRIX(BOOL,   bool)
DEFINE_READ_MATRIX(UINT8,  uint8_t)
DEFINE_READ_MATRIX(INT8,   int8_t)
DEFINE_READ_MATRIX(UINT16, uint16_t)
DEFINE_READ_MATRIX(INT16,  int16_t)
DEFINE_READ_MATRIX(UINT32, uint32_t)
DEFINE_READ_MATRIX(INT32,  int32_t)
DEFINE_READ_MATRIX(UINT64, uint64_t)
DEFINE_READ_MATRIX(INT64,  int64_t)
DEFINE_READ_MATRIX(FP32,   float)
DEFINE_READ_MATRIX(FP64,   double)
/* @formatter:on */


#endif //GRB_FUSION_READER_C_H
