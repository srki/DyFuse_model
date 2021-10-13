#include <GraphBLAS.h>
#include <otx/atx.h>
#include "../../util/reader_c.h"
#include "../../util/timer.h"
#include "../../util/GrB_util.h"


int main(int argc, char *argv[]) {
    int numThreads;
    size_t niter;
    char *A_path;
    char *B_path;
    GrB_Matrix A;
    GrB_Matrix B;
    GrB_Matrix C;
    GrB_Index nrows;
    GrB_Index ncols;
    double sum;

    /* Init and set the number of threads */
    GrB_init(GrB_BLOCKING);
    if (arg_to_int32_def(argc, argv, "-nt|--numThreads", &numThreads, 1) != OTX_SUCCESS) { return 1; }
    GxB_Global_Option_set(GxB_NTHREADS, numThreads);

    if (arg_to_uint64_def(argc, argv, "--niter", &niter, 1) != OTX_SUCCESS) { return 2; }

    /* Read/init the matrices */
    {
        unsigned long long start = get_time_ns();
        if (arg_to_str_def(argc, argv, "-A", &A_path, INPUTS_DIR"/simple") != OTX_SUCCESS) { return 3; }
        if (read_Matrix_FP64(&A, A_path, false) != GrB_SUCCESS) { return 4; }
        if (arg_to_str_def(argc, argv, "-B", &B_path, A_path) != OTX_SUCCESS) { return 5; }
        if (read_Matrix_FP64(&B, B_path, false) != GrB_SUCCESS) { return 6; }
        GrB_Matrix_nrows(&nrows, A);
        GrB_Matrix_ncols(&ncols, B);
        GrB_Matrix_new(&C, GrB_FP64, nrows, ncols);
        unsigned long long end = get_time_ns();
        printf("Read time: %lf\n", (end - start) / 1e9);
    }

    /* Execute mxm */
    for (size_t i = 0; i < niter; i++) {
        unsigned long long start = get_time_ns();
        GrB_mxm(C, GrB_NULL, GrB_NULL, GrB_PLUS_TIMES_SEMIRING_FP64, A, B, GrB_NULL);
        GrB_Matrix_wait(&C);
        unsigned long long end = get_time_ns();
        GrB_reduce(&sum, GrB_NULL, GrB_PLUS_MONOID_FP64, C, GrB_NULL);
        printf("Execution time: %lf\n", (end - start) / 1e9);
        printf("LOG,%s,0,0,0,%d,%lf,%lf\n", get_file_name(A_path), numThreads, (end - start) / 1e9, sum);
    }

    return 0;
}
