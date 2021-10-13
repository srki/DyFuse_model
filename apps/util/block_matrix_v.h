#ifndef GRB_FUSION_BLOCK_MATRIX_V_H
#define GRB_FUSION_BLOCK_MATRIX_V_H

#include <GraphBLAS.h>
#include <stdlib.h>

#ifndef OK
#error OK undefined
#define OK(A) A;
#endif

#ifndef TYPE_GrB
#error TYPE_GrB undefined
#define TYPE_GrB GrB_BOOL
#endif

struct blocked_coord_t {
    GrB_Index row, col;
    size_t row_blocked, col_blocked;
    TYPE val;
};

int cmp_row_func(const void *rhs, const void *lhs) {
    return ((struct blocked_coord_t *) rhs)->row - ((struct blocked_coord_t *) lhs)->row;
}

int cmp_col_func(const void *rhs, const void *lhs) {
    return ((struct blocked_coord_t *) rhs)->col - ((struct blocked_coord_t *) lhs)->col;
}

GrB_Info create_blocks_v(GrB_Matrix **bA_out, GrB_Matrix A,
                         size_t nrows_blocked, size_t *row_block_sizes,
                         size_t ncols_blocked, size_t *col_block_sizes) {
    /* Input matrix tuples and matrix blocks*/
    GrB_Index *I, *J;
    TYPE *X;
    size_t *block_tuples_start;
    GrB_Matrix *bA;

    GrB_Index nrows, ncols, nvals;
    size_t *nzb_cnt;

    /* Get input matrix info and calculate the number of blocks */
    OK(GrB_Matrix_nvals(&nvals, A));
    OK(GrB_Matrix_nrows(&nrows, A));
    OK(GrB_Matrix_ncols(&ncols, A));

    /* Calculate prefix sums for row_block_sizes and col_block_sizes */
    size_t *row_block_start = malloc((nrows_blocked + 1) * sizeof(size_t));
    size_t *col_block_start = malloc((ncols_blocked + 1) * sizeof(size_t));

    row_block_start[0] = 0;
    for (size_t i = 0; i < nrows_blocked; i++) {
        row_block_start[i + 1] = row_block_start[i] + row_block_sizes[i];
    }

    col_block_start[0] = 0;
    for (size_t i = 0; i < ncols_blocked; i++) {
        col_block_start[i + 1] = col_block_start[i] + col_block_sizes[i];
    }

    /* Allocate tuple arrays and extract tuples */
    I = malloc(nvals * sizeof(GrB_Index));
    J = malloc(nvals * sizeof(GrB_Index));
    X = malloc(nvals * sizeof(TYPE));
    GrB_Matrix_extractTuples(I, J, X, &nvals, A);

    /* Create & fill field coord array */
    struct blocked_coord_t *blocked_coords = malloc(nvals * sizeof(struct blocked_coord_t));
    for (GrB_Index i = 0; i < nvals; i++) {
        blocked_coords[i].row = I[i];
        blocked_coords[i].col = J[i];
        blocked_coords[i].val = X[i];
    }

    qsort(blocked_coords, nvals, sizeof(struct blocked_coord_t), cmp_row_func);

    size_t nextCoord = 0;
    for (size_t curr_block = 0; curr_block < nrows_blocked; curr_block++) {
        while (nextCoord < nvals && blocked_coords[nextCoord].row < row_block_start[curr_block + 1]) {
            blocked_coords[nextCoord].row_blocked = curr_block;
            nextCoord++;
        }
    }

    qsort(blocked_coords, nvals, sizeof(struct blocked_coord_t), cmp_col_func);

    nextCoord = 0;
    for (size_t curr_block = 0; curr_block < ncols_blocked; curr_block++) {
        while (nextCoord < nvals && blocked_coords[nextCoord].col < col_block_start[curr_block + 1]) {
            blocked_coords[nextCoord].col_blocked = curr_block;
            nextCoord++;
        }
    }

    /* Calculate the number of nonzeros per block */
    nzb_cnt = calloc(nrows_blocked * ncols_blocked, sizeof(size_t));
    for (size_t i = 0; i < nvals; i++) {
        nzb_cnt[blocked_coords[i].row_blocked * ncols_blocked + blocked_coords[i].col_blocked]++;
    }

    /* Split block tuples */
    block_tuples_start = malloc(nrows_blocked * ncols_blocked * sizeof(size_t));
    size_t tuple_cnt = 0;
    for (size_t i = 0; i < nrows_blocked * ncols_blocked; i++) {
        block_tuples_start[i] = tuple_cnt;
        tuple_cnt += nzb_cnt[i];
    }

    for (size_t i = 0; i < nvals; i++) {
        size_t block_idx = blocked_coords[i].row_blocked * ncols_blocked + blocked_coords[i].col_blocked;
        size_t tuple_idx = block_tuples_start[block_idx]++;
        I[tuple_idx] = blocked_coords[i].row - row_block_start[blocked_coords[i].row_blocked];
        J[tuple_idx] = blocked_coords[i].col - col_block_start[blocked_coords[i].col_blocked];
        X[tuple_idx] = blocked_coords[i].val;
    }

    /* Create blocks and init blocks*/
    bA = malloc(nrows_blocked * ncols_blocked * sizeof(GrB_Matrix));

    tuple_cnt = 0;
    for (size_t i = 0; i < nrows_blocked * ncols_blocked; i++) {
        OK(GrB_Matrix_new(&bA[i], TYPE_GrB,
                       row_block_sizes[i / ncols_blocked],
                       col_block_sizes[i % ncols_blocked]));
        OK(GrB_Matrix_build(bA[i], &I[tuple_cnt], &J[tuple_cnt], &X[tuple_cnt], nzb_cnt[i], TYPE_PLUS));

        tuple_cnt += nzb_cnt[i];
    }

    *bA_out = bA;

    /* Free the resources */
    free(I);
    free(J);
    free(X);
    free(row_block_start);
    free(col_block_start);
    free(blocked_coords);
    free(nzb_cnt);
    free(block_tuples_start);
}

GrB_Info free_blocks_v(GrB_Matrix *bA, size_t nrows_blocked, size_t ncols_blocked) {
    size_t nblocks = nrows_blocked * ncols_blocked;
    for (size_t i = 0; i < nblocks; i++) {
        if (bA[i] != NULL) {
            OK(GrB_Matrix_free(&bA[i]));
        }
    }
    free(bA);

    return GrB_SUCCESS;
}

#endif //GRB_FUSION_BLOCK_MATRIX_V_H
