#include <GraphBLAS.h>
#include <otx/otx.h>
#include "../util/reader_c.h"


void ktruss(GrB_Matrix A, size_t k) {
    GrB_Index nrows;
    GrB_Index nnz_cur, nnz_last;
    GxB_Scalar s = NULL;
    GrB_Matrix C = NULL, T = NULL;

    GxB_Scalar_new(&s, GrB_UINT64);
    GxB_Scalar_setElement_UINT64(s, (uint64_t) k - 2);
    GrB_Matrix_nrows(&nrows, A);

    T = A;
    GrB_Matrix_nvals(&nnz_cur, A);
    do {
        nnz_last = nnz_cur;
        GrB_Matrix_new(&C, GrB_UINT64, nrows, nrows);
        GrB_mxm(C, T, NULL, GxB_PLUS_PAIR_UINT64, T, T, GrB_DESC_S);

        GxB_Matrix_select(C, NULL, NULL, GxB_GE_THUNK, C, s, GrB_NULL);
        if (A != T) { GrB_Matrix_free(&T); }
        T = C;

        GrB_Matrix_nvals(&nnz_cur, C);
    } while (nnz_last != nnz_cur);
    GrB_Matrix_free(&T);

    printf("%lu\n", nnz_last);
}

int main(int argc, char *argv[]) {
    int numThreads;
    char *matrix_path;
    uint64_t k;
    GrB_Index nrows;
    GrB_Matrix A;

    /* Init and set the number of threads */
    GrB_init(GrB_NONBLOCKING);
    if (arg_to_int32_def(argc, argv, "-nt|--numThreads", &numThreads, 1) != OTX_SUCCESS) { return 1; }
    GxB_Global_Option_set(GxB_GLOBAL_NTHREADS, numThreads);

    /* Read the matrix */
    if (arg_to_str_def(argc, argv, "-g|--graph", &matrix_path, INPUTS_DIR"/simple") != OTX_SUCCESS) { return 2; }
    if (read_Matrix_UINT64(&A, matrix_path, true) != GrB_SUCCESS) { return 3; }

    /* Remove A's diagonal */
    GxB_Scalar s;
    GxB_Scalar_new(&s, GrB_UINT64);
    GxB_Scalar_setElement_UINT64(s, (uint64_t) 0);
    GxB_Matrix_select(A, GrB_NULL, GrB_NULL, GxB_OFFDIAG, A, s, GrB_NULL);

    /* Convert A to matrix to symmetric 0-1 matrix */
    GrB_Matrix_nrows(&nrows, A);
    GrB_Matrix_assign_INT32(A, A, GrB_NULL, 1, GrB_ALL, nrows, GrB_ALL, nrows, GrB_DESC_S);
    GrB_Matrix_eWiseAdd_BinaryOp(A, GrB_NULL, GrB_NULL, GrB_TIMES_UINT64, A, A, GrB_DESC_T1);

    if (arg_to_uint64_def(argc, argv, "-k", &k, 3)) { return 4; }
    assert(k >= 3);
    ktruss(A, k);

    GrB_finalize();
}