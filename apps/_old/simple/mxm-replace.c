#include <GraphBLAS.h>
#include <otx/otx.h>
#include "../../util/reader_c.h"

void cast_matrix(GrB_Matrix *A, GrB_Type type, GrB_UnaryOp cast_op) {
    GrB_Index nrows, ncols;
    GrB_Matrix_nrows(&nrows, *A);
    GrB_Matrix_ncols(&ncols, *A);

    GrB_Matrix tmpA;
    GrB_Matrix_new(&tmpA, type, nrows, ncols);
    GrB_Matrix_apply(tmpA, GrB_NULL, GrB_NULL, cast_op, *A, GrB_NULL);

    GrB_free(A);
    *A = tmpA;
}

int main(int argc, char *argv[]) {
    char *A_path;
    GrB_Matrix A;

    GrB_Index nrows, ncols;

    /* Init and set the number of threads */
    GrB_init(GrB_NONBLOCKING);

    /* Read the matrices */
    if (arg_to_str_def(argc, argv, "-A", &A_path, INPUTS_DIR"/simple") != OTX_SUCCESS) { return 1; }
    if (read_Matrix_FP64(&A, A_path, false) != GrB_SUCCESS) { return 1; }

    GrB_Matrix_nrows(&nrows, A);
    GrB_Matrix_ncols(&ncols, A);

    GrB_Matrix C;
    GrB_Matrix_new(&C, GrB_UINT64, nrows, ncols);

    GrB_mxm(C, GrB_NULL, GrB_NULL, GrB_PLUS_TIMES_SEMIRING_UINT64, A, A, GrB_NULL);
    GxB_print(C, GxB_COMPLETE);

    GrB_mxm(C, A, GrB_NULL, GrB_PLUS_TIMES_SEMIRING_UINT64, A, A, GrB_NULL);
    GxB_print(C, GxB_COMPLETE);
}