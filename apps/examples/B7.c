/* Copied from http://people.eecs.berkeley.edu/~aydin/GraphBLAS_API_C_v13.pdf and modified. */

#include <GraphBLAS.h>
#include "../util/reader_c.h"


uint64_t triangle_count(GrB_Matrix L) {
    GrB_Index n;
    GrB_Matrix_nrows(&n, L);

    GrB_Matrix C;
    GrB_Matrix_new(&C, GrB_UINT64, n, n);
    GrB_mxm(C, L, GrB_NULL, GxB_PLUS_TIMES_UINT64, L, L, GrB_DESC_T1);

    uint64_t count;
    GrB_reduce(&count, GrB_NULL, GxB_PLUS_UINT64_MONOID, C, GrB_NULL);

    GrB_free(&C);

    return count;
}

int main(int argc, char *argv[]) {
    GrB_Matrix A;
    uint64_t count;

    GrB_init(GrB_NONBLOCKING);

    if (read_Matrix_UINT64(&A, argc > 1 ? argv[1] : INPUTS_DIR"/simple", false) != GrB_SUCCESS) { return 1; }

    /* TODO: add select */
    count = triangle_count(A);
    printf("Number of triangles: %" PRIu64 "\n", count);

    return 0;
}