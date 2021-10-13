#include "../../util/GrB_util.h"

#define TYPE int
#define TYPE_GrB GrB_INT32
#define TYPE_PLUS GrB_PLUS_INT32
#define TYPE_PLUS_MONOID GrB_PLUS_MONOID_INT32
#define TYPE_PLUS_TIMES GxB_PLUS_TIMES_INT32

#include "../../util/block_matrix_v.h"

void print_dense(GrB_Matrix A) {
    GrB_Index nrows;
    GrB_Index ncols;
    GrB_Index nvals;

    GrB_Matrix_nrows(&nrows, A);
    GrB_Matrix_ncols(&ncols, A);
    GrB_Matrix_nvals(&nvals, A);

    TYPE *matrix = calloc(nrows * ncols, sizeof(TYPE));

    GrB_Index *I = malloc(nvals * sizeof(GrB_Index));
    GrB_Index *J = malloc(nvals * sizeof(GrB_Index));
    TYPE *X = malloc(nvals * sizeof(TYPE));

    GrB_Matrix_extractTuples(I, J, X, &nvals, A);

    if (matrix == 0 || I == 0 || J == 0 || X == 0) { exit(1); }

    for (GrB_Index i = 0; i < nvals; i++) {
        matrix[I[i] * ncols + J[i]] = X[i];
    }

    for (size_t i = 0; i < nrows; i++) {
        for (size_t j = 0; j < ncols; j++) {
            printf("%d ", matrix[i * ncols + j]);
        }
        printf("\n");
    }

    free(I);
    free(J);
    free(X);
    free(matrix);
}

int main() {
    GrB_init(GrB_BLOCKING);

    GrB_Matrix A;
    GrB_Matrix *A_blocked;
    GrB_Matrix *B_blocked;
    GrB_Matrix *C_blocked;

    OK(GrB_Matrix_new(&A, TYPE_GrB, 9, 9));


    size_t nrows_blocked_A = 3, ncols_blocked_A = 1;
    size_t *row_block_sizes_A = malloc(nrows_blocked_A * sizeof(size_t));
    size_t *col_block_sizes_A = malloc(ncols_blocked_A * sizeof(size_t));

    row_block_sizes_A[0] = 2;
    row_block_sizes_A[1] = 4;
    row_block_sizes_A[2] = 3;

    col_block_sizes_A[0] = 9;

    size_t nrows_blocked_B = 1, ncols_blocked_B = 3;
    size_t *row_block_sizes_B = malloc(nrows_blocked_B * sizeof(size_t));
    size_t *col_block_sizes_B = malloc(ncols_blocked_B * sizeof(size_t));

    row_block_sizes_B[0] = 9;

    col_block_sizes_B[0] = 2;
    col_block_sizes_B[1] = 4;
    col_block_sizes_B[2] = 3;


    /* region Set elements */
    GrB_Matrix_setElement(A, 1, 0, 1);
    GrB_Matrix_setElement(A, 7, 0, 4);
    GrB_Matrix_setElement(A, 1, 0, 6);

    GrB_Matrix_setElement(A, 2, 1, 1);
    GrB_Matrix_setElement(A, 3, 1, 3);
    GrB_Matrix_setElement(A, 8, 1, 4);
    GrB_Matrix_setElement(A, 2, 1, 7);

    GrB_Matrix_setElement(A, 4, 2, 2);

    GrB_Matrix_setElement(A, 1, 3, 0);
    GrB_Matrix_setElement(A, 6, 3, 4);
    GrB_Matrix_setElement(A, 9, 3, 7);

    GrB_Matrix_setElement(A, 5, 4, 1);
    GrB_Matrix_setElement(A, 3, 4, 5);

    GrB_Matrix_setElement(A, 2, 5, 2);
    GrB_Matrix_setElement(A, 1, 5, 4);

    GrB_Matrix_setElement(A, 9, 6, 0);
    GrB_Matrix_setElement(A, 5, 6, 4);
    GrB_Matrix_setElement(A, 8, 6, 6);
    GrB_Matrix_setElement(A, 2, 6, 8);

    GrB_Matrix_setElement(A, 1, 7, 1);
    GrB_Matrix_setElement(A, 8, 7, 3);
    GrB_Matrix_setElement(A, 6, 7, 5);
    GrB_Matrix_setElement(A, 9, 7, 7);

    GrB_Matrix_setElement(A, 3, 8, 0);
    GrB_Matrix_setElement(A, 4, 8, 3);
    GrB_Matrix_setElement(A, 3, 8, 6);
    GrB_Matrix_setElement(A, 1, 8, 8);
    /* endregion */

    print_dense(A);

    create_blocks_v(&A_blocked, A, nrows_blocked_A, row_block_sizes_A, ncols_blocked_A, col_block_sizes_A);

    printf("1D Row blocking\n");
    for (size_t i = 0; i < nrows_blocked_A; i++) {
        for (size_t j = 0; j < ncols_blocked_A; j++) {
            printf("\n");
            print_dense(A_blocked[i * ncols_blocked_A + j]);
        }
    }

    create_blocks_v(&B_blocked, A, nrows_blocked_B, row_block_sizes_B, ncols_blocked_B, col_block_sizes_B);

    printf("1D Column blocking\n");
    for (size_t i = 0; i < nrows_blocked_B; i++) {
        for (size_t j = 0; j < ncols_blocked_B; j++) {
            printf("\n");
            print_dense(B_blocked[i * ncols_blocked_B + j]);
        }
    }

    create_blocks_v(&C_blocked, A, nrows_blocked_A, row_block_sizes_A, ncols_blocked_B, col_block_sizes_B);

    printf("2D blocking\n");
    for (size_t i = 0; i < nrows_blocked_A; i++) {
        for (size_t j = 0; j < ncols_blocked_B; j++) {
            printf("\n");
            print_dense(C_blocked[i * ncols_blocked_B + j]);
        }
    }

    /* mxm */
    TYPE sum = 0;
    for (size_t i = 0; i < nrows_blocked_A; i++) {
        for (size_t j = 0; j < ncols_blocked_B; j++) {
            GrB_Matrix C;
            GrB_Matrix_new(&C, TYPE_GrB, row_block_sizes_A[i], col_block_sizes_B[j]);

            OK(GrB_mxm(C, GrB_NULL, TYPE_PLUS, TYPE_PLUS_TIMES, A_blocked[i], B_blocked[j], GrB_NULL));
            OK(GrB_Matrix_reduce_INT32(&sum, TYPE_PLUS, TYPE_PLUS_MONOID, C, GrB_NULL));

            GrB_free(&C);
        }
    }

    printf("%d\n", sum);

    OK(GrB_free(&A));
    free_blocks_v(A_blocked, nrows_blocked_A, ncols_blocked_A);
    free_blocks_v(B_blocked, ncols_blocked_B, nrows_blocked_B);
    free_blocks_v(C_blocked, nrows_blocked_A, ncols_blocked_B);
    free(row_block_sizes_A);
    free(col_block_sizes_A);
    free(row_block_sizes_B);
    free(col_block_sizes_B);
    GrB_finalize();

    return 0;
}

