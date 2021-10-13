#ifndef GRB_FUSION_BLOCK_MATRIX_H
#define GRB_FUSION_BLOCK_MATRIX_H

#ifdef __cplusplus
extern "C" {
#endif

#include <GraphBLAS.h>

#ifdef __cplusplus
}
#endif

#ifndef OK
#error OK undefined
#define OK(A) A;
#endif

#ifndef TYPE_GrB
#error TYPE_GrB undefined
#define TYPE_GrB GrB_BOOL
#endif

#include "../util/get_feature.h"
#include "../util/timer.h"

#define CREATE_BLOCKS_DEF(_name, _type, _grb_type)                                                              \
GrB_Info create_blocks_##_name(GrB_Matrix **bA_out, GrB_Index *nrows_blocked_out, GrB_Index *ncols_blocked_out, \
            GrB_Matrix A, size_t nrows_block, size_t ncols_block, bool full_idxes, double* features) {          \
    /* Input matrix tuples */                                                                                   \
    GrB_Index *I, *J;                                                                                           \
    _type *X;                                                                                                   \
                                                                                                                \
    /* Blocked tuples and matrix blocks */                                                                      \
    GrB_Index **bI, **bJ;                                                                                       \
    _type **bX;                                                                                                 \
    GrB_Matrix *bA;                                                                                             \
                                                                                                                \
    GrB_Index nrows, ncols, nvals;                                                                              \
    size_t nblocks, nrows_blocked, ncols_blocked;                                                               \
    size_t *nzb_cnt;                                                                                            \
                                                                                                                \
    /* Get input matrix info and calculate the number of blocks */                                              \
    OK(GrB_Matrix_nvals(&nvals, A));                                                                            \
    OK(GrB_Matrix_nrows(&nrows, A));                                                                            \
    OK(GrB_Matrix_ncols(&ncols, A));                                                                            \
                                                                                                                \
    if (nrows_block == 0) { GrB_Matrix_nrows(&nrows_block, A); }                                                \
    if (ncols_block == 0) { GrB_Matrix_ncols(&ncols_block, A); }                                                \
                                                                                                                \
    nrows_blocked = ((nrows + nrows_block - 1) / nrows_block);                                                  \
    ncols_blocked = ((ncols + ncols_block - 1) / ncols_block);                                                  \
    nblocks = nrows_blocked * ncols_blocked;                                                                    \
                                                                                                                \
    /* Allocate the output arrays */                                                                            \
    nzb_cnt = (size_t *) calloc(nblocks, sizeof(size_t));                                                       \
    bI = (GrB_Index **) malloc(nblocks * sizeof(GrB_Index *));                                                  \
    bJ = (GrB_Index **) malloc(nblocks * sizeof(GrB_Index *));                                                  \
    bX = (_type **) malloc(nblocks * sizeof(_type *));                                                          \
    if (nzb_cnt == NULL || bI == NULL || bJ == NULL) { exit(1); }                                               \
                                                                                                                \
    /* Get tuples form the input matrix */                                                                      \
    I = (GrB_Index *) malloc((nvals + 1) * sizeof(GrB_Index));                                                  \
    J = (GrB_Index *) malloc((nvals + 1) * sizeof(GrB_Index));                                                  \
    X = (_type *) malloc((nvals + 1) * sizeof(_type));                                                          \
    if (I == NULL || J == NULL || X == NULL) { exit(1); }                                                       \
    OK(GrB_Matrix_extractTuples_##_grb_type(I, J, X, &nvals, A));                                               \
                                                                                                                \
    /* Calculate the number of elements per block */                                                            \
    for (GrB_Index i = 0; i < nvals; i++) {                                                                     \
        GrB_Index row_idx = I[i] / nrows_block;                                                                 \
        GrB_Index col_idx = J[i] / ncols_block;                                                                 \
        nzb_cnt[row_idx * ncols_blocked + col_idx]++;                                                           \
    }                                                                                                           \
                                                                                                                \
    /* Allocate memory for block tuples */                                                                      \
    for (size_t i = 0; i < nblocks; i++) {                                                                      \
        bI[i] = (GrB_Index *) malloc(nzb_cnt[i] * sizeof(GrB_Index));                                           \
        bJ[i] = (GrB_Index *) malloc(nzb_cnt[i] * sizeof(GrB_Index));                                           \
        bX[i] = (_type *) malloc(nzb_cnt[i] * sizeof(_type));                                                   \
        if (bI[i] == NULL || bJ[i] == NULL || bX[i] == NULL) { exit(1); }                                       \
    }                                                                                                           \
                                                                                                                \
    /* Fill the block tuples */                                                                                 \
    memset(nzb_cnt, 0, nblocks * sizeof(size_t));                                                               \
    for (GrB_Index i = 0; i < nvals; i++) {                                                                     \
        GrB_Index row_idx = I[i] / nrows_block;                                                                 \
        GrB_Index col_idx = J[i] / ncols_block;                                                                 \
        GrB_Index block_idx = row_idx * ncols_blocked + col_idx;                                                \
        GrB_Index idx = nzb_cnt[block_idx]++;                                                                   \
                                                                                                                \
        if (full_idxes) {                                                                                       \
            bI[block_idx][idx] = I[i];                                                                          \
            bJ[block_idx][idx] = J[i];                                                                          \
        } else {                                                                                                \
            bI[block_idx][idx] = I[i] % nrows_block;                                                            \
            bJ[block_idx][idx] = J[i] % ncols_block;                                                            \
        }                                                                                                       \
                                                                                                                \
        bX[block_idx][idx] = X[i];                                                                              \
    }                                                                                                           \
                                                                                                                \
    /* Allocate the output matrices */                                                                          \
    bA = (GrB_Matrix *) malloc(nblocks * sizeof(GrB_Matrix));                                                   \
    for (size_t i = 0; i < nblocks; i++) {                                                                      \
        if (full_idxes) {                                                                                       \
            OK(GrB_Matrix_new(&bA[i], GrB_##_grb_type, nrows, ncols));                                          \
        } else {                                                                                                \
            OK(GrB_Matrix_new(&bA[i], GrB_##_grb_type, nrows_block, ncols_block));                              \
        }                                                                                                       \
        OK(GrB_Matrix_build_##_grb_type(bA[i], bI[i], bJ[i], bX[i], nzb_cnt[i], GrB_PLUS_##_grb_type));         \
    }                                                                                                           \
                                                                                                                \
    for (size_t bi = 0; bi < nrows_blocked; bi++) {                                                             \
        for (size_t bj = 0; bj < ncols_blocked; bj++) {                                                         \
            GrB_Index block_idx = bi * ncols_blocked + bj;                                                      \
            features[block_idx *NUM_FEATURE ] = block_idx;                                                      \
            features[block_idx *NUM_FEATURE +1] = nrows_block;                                                  \
            features[block_idx *NUM_FEATURE +2] = ncols_block;                                                  \
            features[block_idx *NUM_FEATURE +3] = nzb_cnt[block_idx];                                           \
        }                                                                                                       \
    }                                                                                                           \
                                                                                                                \
    /* region Deallocation */                                                                                   \
    for (size_t i = 0; i < nblocks; i++) {                                                                      \
        free(bI[i]);                                                                                            \
        free(bJ[i]);                                                                                            \
        free(bX[i]);                                                                                            \
    }                                                                                                           \
                                                                                                                \
    free(nzb_cnt);                                                                                              \
    free(bI);                                                                                                   \
    free(bJ);                                                                                                   \
    free(bX);                                                                                                   \
                                                                                                                \
    free(I);                                                                                                    \
    free(J);                                                                                                    \
    free(X);                                                                                                    \
    /* endregion */                                                                                             \
                                                                                                                \
    /* Set the output values */                                                                                 \
    *bA_out = bA;                                                                                               \
    *nrows_blocked_out = nrows_blocked;                                                                         \
    *ncols_blocked_out = ncols_blocked;                                                                         \
                                                                                                                \
    return GrB_SUCCESS;                                                                                         \
}

/* TODO: add all types */
CREATE_BLOCKS_DEF(uint64, uint64_t, UINT64)

CREATE_BLOCKS_DEF(double, double, FP64)

GrB_Info create_blocks_type(GrB_Matrix **bA_out, GrB_Index *nrows_blocked_out, GrB_Index *ncols_blocked_out,
                            GrB_Matrix A, size_t nrows_block, size_t ncols_block, bool full_idxes, GrB_Type type,double* features) {
    if (type == GrB_UINT64) {
        return create_blocks_uint64(bA_out, nrows_blocked_out, ncols_blocked_out, A, nrows_block, ncols_block, full_idxes,features);
    } else if (type == GrB_FP64) {
        return create_blocks_double(bA_out, nrows_blocked_out, ncols_blocked_out, A, nrows_block, ncols_block, full_idxes,features);
    } else {
        return GrB_INVALID_VALUE;
    }
}

GrB_Info create_blocks(GrB_Matrix **bA_out, GrB_Index *nrows_blocked_out, GrB_Index *ncols_blocked_out,
                       GrB_Matrix A, size_t nrows_block, size_t ncols_block, bool full_idxes) {
    /* Input matrix tuples */
    GrB_Index *I, *J;
    uint64_t *X;

    /* Blocked tuples and matrix blocks */
    GrB_Index **bI, **bJ;
    uint64_t **bX;
    GrB_Matrix *bA;

    GrB_Index nrows, ncols, nvals;
    size_t nblocks, nrows_blocked, ncols_blocked;
    size_t *nzb_cnt;

    /* Get input matrix info and calculate the number of blocks */
    OK(GrB_Matrix_nvals(&nvals, A));
    OK(GrB_Matrix_nrows(&nrows, A));
    OK(GrB_Matrix_ncols(&ncols, A));

    if (nrows_block == 0) { GrB_Matrix_nrows(&nrows_block, A); }
    if (ncols_block == 0) { GrB_Matrix_ncols(&ncols_block, A); }

    nrows_blocked = ((nrows + nrows_block - 1) / nrows_block);
    ncols_blocked = ((ncols + ncols_block - 1) / ncols_block);
    nblocks = nrows_blocked * ncols_blocked;

    /* Allocate the output arrays */
    nzb_cnt = (size_t *) calloc(nblocks, sizeof(size_t));
    bI = (GrB_Index **) malloc(nblocks * sizeof(GrB_Index *));
    bJ = (GrB_Index **) malloc(nblocks * sizeof(GrB_Index *));
    bX = (uint64_t **) malloc(nblocks * sizeof(uint64_t *));
    if (nzb_cnt == NULL || bI == NULL || bJ == NULL) { exit(1); }

    /* Get tuples form the input matrix */
    I = (GrB_Index *) malloc((nvals + 1) * sizeof(GrB_Index));
    J = (GrB_Index *) malloc((nvals + 1) * sizeof(GrB_Index));
    X = (GrB_Index *) malloc((nvals + 1) * sizeof(uint64_t));
    if (I == NULL || J == NULL || X == NULL) { exit(1); }
    OK(GrB_Matrix_extractTuples_UINT64(I, J, X, &nvals, A));

    /* Calculate the number of elements per block */
    for (GrB_Index i = 0; i < nvals; i++) {
        GrB_Index row_idx = I[i] / nrows_block;
        GrB_Index col_idx = J[i] / ncols_block;
        nzb_cnt[row_idx * ncols_blocked + col_idx]++;
    }

    /* Allocate memory for block tuples */
    for (size_t i = 0; i < nblocks; i++) {
        bI[i] = (GrB_Index *) malloc(nzb_cnt[i] * sizeof(GrB_Index));
        bJ[i] = (GrB_Index *) malloc(nzb_cnt[i] * sizeof(GrB_Index));
        bX[i] = (uint64_t *) malloc(nzb_cnt[i] * sizeof(uint64_t));
        if (bI[i] == NULL || bJ[i] == NULL || bX[i] == NULL) { exit(1); }
    }

    /* Fill the block tuples */
    memset(nzb_cnt, 0, nblocks * sizeof(size_t));
    for (GrB_Index i = 0; i < nvals; i++) {
        GrB_Index row_idx = I[i] / nrows_block;
        GrB_Index col_idx = J[i] / ncols_block;
        GrB_Index block_idx = row_idx * ncols_blocked + col_idx;
        GrB_Index idx = nzb_cnt[block_idx]++;

        if (full_idxes) {
            bI[block_idx][idx] = I[i];
            bJ[block_idx][idx] = J[i];
        } else {
            bI[block_idx][idx] = I[i] % nrows_block;
            bJ[block_idx][idx] = J[i] % ncols_block;
        }

        bX[block_idx][idx] = X[i];
    }

    /* Allocate the output matrices */
    bA = (GrB_Matrix *) malloc(nblocks * sizeof(GrB_Matrix));
    for (size_t i = 0; i < nblocks; i++) {

        if (full_idxes) {
            OK(GrB_Matrix_new(&bA[i], TYPE_GrB, nrows, ncols));
        } else {
            OK(GrB_Matrix_new(&bA[i], TYPE_GrB, nrows_block, ncols_block));
        }
        OK(GrB_Matrix_build_UINT64(bA[i], bI[i], bJ[i], bX[i], nzb_cnt[i], GrB_PLUS_UINT64));
    }

//    GxB_print(A, GxB_COMPLETE);
//    for (size_t bi = 0; bi < nrows_blocked; bi++) {
//        for (size_t bj = 0; bj < ncols_blocked; bj++) {
//            printf("Block (%lu, %lu): \n", bi, bj);
//
//            GrB_Index block_idx = bi * ncols_blocked + bj;
//            for (size_t i = 0; i < nzb_cnt[block_idx]; i++) {
//                printf("\t(%lu, %lu) = %u\n", bI[block_idx][i], bJ[block_idx][i], bX[block_idx][i]);
//            }
//
//            if (bA[block_idx] != NULL) {
//                GxB_print(bA[block_idx], GxB_COMPLETE);
//            }
//            printf("--------------------------------\n");
//        }
//    }


    /* region Deallocation */
    for (size_t i = 0; i < nblocks; i++) {
        free(bI[i]);
        free(bJ[i]);
        free(bX[i]);
    }

    free(nzb_cnt);
    free(bI);
    free(bJ);
    free(bX);

    free(I);
    free(J);
    free(X);
    /* endregion */

    /* Set the output values */
    *bA_out = bA;
    *nrows_blocked_out = nrows_blocked;
    *ncols_blocked_out = ncols_blocked;

    return GrB_SUCCESS;
}

GrB_Info create_blocks_features(GrB_Matrix **bA_out, GrB_Index *nrows_blocked_out, GrB_Index *ncols_blocked_out,
                                GrB_Matrix A, size_t nrows_block, size_t ncols_block, bool full_idxes, double *features) {
    /* Input matrix tuples */
    GrB_Index *I, *J;
    uint64_t *X;

    /* Blocked tuples and matrix blocks */
    GrB_Index **bI, **bJ;
    uint64_t **bX;
    GrB_Matrix *bA;

    GrB_Index nrows, ncols, nvals;
    size_t nblocks, nrows_blocked, ncols_blocked;
    size_t *nzb_cnt;

    /* Get input matrix info and calculate the number of blocks */
    OK(GrB_Matrix_nvals(&nvals, A));
    OK(GrB_Matrix_nrows(&nrows, A));
    OK(GrB_Matrix_ncols(&ncols, A));

    if (nrows_block == 0) { GrB_Matrix_nrows(&nrows_block, A); }
    if (ncols_block == 0) { GrB_Matrix_ncols(&ncols_block, A); }

    nrows_blocked = ((nrows + nrows_block - 1) / nrows_block);
    ncols_blocked = ((ncols + ncols_block - 1) / ncols_block);
    nblocks = nrows_blocked * ncols_blocked;

    /* Allocate the output arrays */
    nzb_cnt = (size_t *) calloc(nblocks, sizeof(size_t));
    bI = (GrB_Index **) malloc(nblocks * sizeof(GrB_Index *));
    bJ = (GrB_Index **) malloc(nblocks * sizeof(GrB_Index *));
    bX = (uint64_t **) malloc(nblocks * sizeof(uint64_t *));
    if (nzb_cnt == NULL || bI == NULL || bJ == NULL) { exit(1); }

    /* Get tuples form the input matrix */
    I = (GrB_Index *) malloc((nvals + 1) * sizeof(GrB_Index));
    J = (GrB_Index *) malloc((nvals + 1) * sizeof(GrB_Index));
    X = (GrB_Index *) malloc((nvals + 1) * sizeof(uint64_t));
    if (I == NULL || J == NULL || X == NULL) { exit(1); }
    OK(GrB_Matrix_extractTuples_UINT64(I, J, X, &nvals, A));

    /* Calculate the number of elements per block */
    for (GrB_Index i = 0; i < nvals; i++) {
        GrB_Index row_idx = I[i] / nrows_block;
        GrB_Index col_idx = J[i] / ncols_block;
        nzb_cnt[row_idx * ncols_blocked + col_idx]++;
    }

    /* Allocate memory for block tuples */
    for (size_t i = 0; i < nblocks; i++) {
        bI[i] = (GrB_Index *) malloc(nzb_cnt[i] * sizeof(GrB_Index));
        bJ[i] = (GrB_Index *) malloc(nzb_cnt[i] * sizeof(GrB_Index));
        bX[i] = (uint64_t *) malloc(nzb_cnt[i] * sizeof(uint64_t));
        if (bI[i] == NULL || bJ[i] == NULL || bX[i] == NULL) { exit(1); }
    }

    /* Fill the block tuples */
    memset(nzb_cnt, 0, nblocks * sizeof(size_t));
    for (GrB_Index i = 0; i < nvals; i++) {
        GrB_Index row_idx = I[i] / nrows_block;
        GrB_Index col_idx = J[i] / ncols_block;
        GrB_Index block_idx = row_idx * ncols_blocked + col_idx;
        GrB_Index idx = nzb_cnt[block_idx]++;

        if (full_idxes) {
            bI[block_idx][idx] = I[i];
            bJ[block_idx][idx] = J[i];
        } else {
            bI[block_idx][idx] = I[i] % nrows_block;
            bJ[block_idx][idx] = J[i] % ncols_block;
        }

        bX[block_idx][idx] = X[i];
    }

    /* Allocate the output matrices */
    bA = (GrB_Matrix *) malloc(nblocks * sizeof(GrB_Matrix));
    for (size_t i = 0; i < nblocks; i++) {

        if (full_idxes) {
            OK(GrB_Matrix_new(&bA[i], TYPE_GrB, nrows, ncols));
        } else {
            OK(GrB_Matrix_new(&bA[i], TYPE_GrB, nrows_block, ncols_block));
        }
        OK(GrB_Matrix_build_UINT64(bA[i], bI[i], bJ[i], bX[i], nzb_cnt[i], GrB_PLUS_UINT64));
    }

    //GxB_print(A, GxB_COMPLETE);
    for (size_t bi = 0; bi < nrows_blocked; bi++) {
        for (size_t bj = 0; bj < ncols_blocked; bj++) {
            GrB_Index block_idx = bi * ncols_blocked + bj;
            features[block_idx *NUM_FEATURE ] = block_idx;
            features[block_idx *NUM_FEATURE +1] = nrows_block;
            features[block_idx *NUM_FEATURE +2] = ncols_block;
            features[block_idx *NUM_FEATURE +3] = nzb_cnt[block_idx];
        }
    }
    //for (size_t bi = 0; bi < nrows_blocked; bi++) {
    //    for (size_t bj = 0; bj < ncols_blocked; bj++) {
    //        printf("Block (%lu, %lu): \n", bi, bj);

    //        GrB_Index block_idx = bi * ncols_blocked + bj;
    //        for (size_t i = 0; i < nzb_cnt[block_idx]; i++) {
    //            printf("\t(%lu, %lu) = %u\n", bI[block_idx][i], bJ[block_idx][i], bX[block_idx][i]);
    //        }

    //        if (bA[block_idx] != NULL) {
    //            GxB_print(bA[block_idx], GxB_COMPLETE);
    //        }
    //        printf("--------------------------------\n");
    //    }
    //}


    /* region Deallocation */
    for (size_t i = 0; i < nblocks; i++) {
        free(bI[i]);
        free(bJ[i]);
        free(bX[i]);
    }

    free(nzb_cnt);
    free(bI);
    free(bJ);
    free(bX);

    free(I);
    free(J);
    free(X);
    /* endregion */

    /* Set the output values */
    *bA_out = bA;
    *nrows_blocked_out = nrows_blocked;
    *ncols_blocked_out = ncols_blocked;

    return GrB_SUCCESS;
}


GrB_Info free_blocks(GrB_Matrix *bA, size_t nrows_blocked, size_t ncols_blocked) {
    size_t nblocks = nrows_blocked * ncols_blocked;
    for (size_t i = 0; i < nblocks; i++) {
        if (bA[i] != NULL) {
            OK(GrB_Matrix_free(&bA[i]));
        }
    }
    free(bA);

    return GrB_SUCCESS;
}

void duplicate_blocks(GrB_Matrix **bC_out, GrB_Matrix *bA, size_t nrows_blocked, size_t ncols_blocked, bool deep_copy) {
    size_t nblocks = nrows_blocked * ncols_blocked;
    GrB_Matrix *bC = (GrB_Matrix *) malloc(nblocks * sizeof(GrB_Matrix));

    for (size_t i = 0; i < nblocks; i++) {
        if (deep_copy) {
            OK(GrB_Matrix_dup(&bC[i], bA[i]));
        } else {
            bC[i] = bA[i];
        }
    }

    *bC_out = bC;
}


#endif //GRB_FUSION_BLOCK_MATRIX_H
