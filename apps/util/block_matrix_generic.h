#ifndef GRB_FUSION_BLOCK_MATRIX_GENERIC_H
#define GRB_FUSION_BLOCK_MATRIX_GENERIC_H

#define CREATE_BLOCKS_DEF(_name, _type, _grb_type)                                                              \
GrB_Info create_blocks_##_name(GrB_Matrix **bA_out, GrB_Matrix A, size_t nrows_blocked, size_t ncols_blocked,   \
                                size_t nrows_block, size_t ncols_block, bool full_idxes) {                      \
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
    size_t nblocks;                                                                                             \
    size_t *nzb_cnt;                                                                                            \
                                                                                                                \
    /* Get input matrix info and calculate the number of blocks */                                              \
    OK(GrB_Matrix_nvals(&nvals, A));                                                                            \
    OK(GrB_Matrix_nrows(&nrows, A));                                                                            \
    OK(GrB_Matrix_ncols(&ncols, A));                                                                            \
                                                                                                                \
    if (nrows_blocked == 0 && nrows_block == 0) {                                                               \
        nrows_block = nrows;                                                                                    \
        nrows_blocked = 1;                                                                                      \
    } else if (nrows_block == 0) {                                                                              \
        nrows_block = ((nrows + nrows_blocked - 1) / nrows_blocked);                                            \
    } else if (nrows_blocked == 0) {                                                                            \
        nrows_blocked = ((nrows + nrows_block - 1) / nrows_block);                                              \
    } else {                                                                                                    \
        assert(nrows_blocked == ((nrows + nrows_block - 1) / nrows_block) ||                                    \
               nrows_block == ((nrows + nrows_blocked - 1) / nrows_blocked));                                   \
    }                                                                                                           \
                                                                                                                \
    if (ncols_blocked == 0 && ncols_block == 0) {                                                               \
        ncols_block = ncols;                                                                                    \
        ncols_blocked = 1;                                                                                      \
    } else if (ncols_block == 0) {                                                                              \
        ncols_block = ((ncols + ncols_blocked - 1) / ncols_blocked);                                            \
    } else if (ncols_blocked == 0) {                                                                            \
        ncols_blocked = ((ncols + ncols_block - 1) / ncols_block);                                              \
    } else {                                                                                                    \
        assert(ncols_blocked == ((ncols + ncols_block - 1) / ncols_block) ||                                    \
               ncols_block == ((ncols + ncols_blocked - 1) / ncols_blocked));                                   \
    }                                                                                                           \
                                                                                                                \
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
                                                                                                                \
    return GrB_SUCCESS;                                                                                         \
}

/* TODO: add all types */
CREATE_BLOCKS_DEF(uint64, uint64_t, UINT64)

CREATE_BLOCKS_DEF(double, double, FP64)

GrB_Info create_blocks_g(GrB_Matrix **bA_out, GrB_Matrix A, GrB_Index nrows_blocked, GrB_Index ncols_blocked,
                         size_t nrows_block, size_t ncols_block, bool full_idxes, GrB_Type type) {
    if (type == GrB_UINT64) {
        return create_blocks_uint64(bA_out, A, nrows_blocked, ncols_blocked, nrows_block, ncols_block, full_idxes);
    } else if (type == GrB_FP64) {
        return create_blocks_double(bA_out, A, nrows_blocked, ncols_blocked, nrows_block, ncols_block, full_idxes);
    } else {
        return GrB_INVALID_VALUE;
    }
}

GrB_Info free_blocks_g(GrB_Matrix *bA, size_t nrows_blocked, size_t ncols_blocked) {
    size_t nblocks = nrows_blocked * ncols_blocked;
    for (size_t i = 0; i < nblocks; i++) {
        if (bA[i] != NULL) {
            OK(GrB_Matrix_free(&bA[i]));
        }
    }
    free(bA);

    return GrB_SUCCESS;
}

#endif //GRB_FUSION_BLOCK_MATRIX_GENERIC_H
