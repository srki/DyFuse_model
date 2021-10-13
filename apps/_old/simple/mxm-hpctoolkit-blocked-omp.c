#include <GraphBLAS.h>
#include <otx/atx.h>
#include "../../util/reader_c.h"
#include "../../util/timer.h"
#include "../../util/GrB_util.h"

#define TYPE_GrB GrB_FP64

#include "../../util/block_matrix.h"

#if defined(HPCTOOLKIT)

#include <hpctoolkit.h>

#else
void hpctoolkit_sampling_start() {}
void hpctoolkit_sampling_stop() {}
#endif


int main(int argc, char *argv[]) {
    int numThreads;
    size_t niter, niterw;
    char *A_path;
    char *B_path;
    GrB_Matrix A;
    GrB_Matrix B;
    GrB_Matrix C;
    GrB_Index nrows;
    GrB_Index ncols;
    double sum;

    GrB_Matrix *bA, *bB, *bC;
    GrB_Index m, n, l;
    GrB_Index bm, bn, bl;
    size_t nblocks_m, nblocks_n, nblocks_l;
    size_t block_size_m, block_size_n, block_size_l;


    /* Init and set the number of threads */
    {
        GrB_init(GrB_BLOCKING);
        if (arg_to_int32_def(argc, argv, "-nt|--numThreads", &numThreads, 1) != OTX_SUCCESS) { return 1; }
        GxB_Global_Option_set(GxB_NTHREADS, 1);
        omp_set_num_threads(numThreads);
        if (arg_to_uint64_def(argc, argv, "--niter", &niter, 1) != OTX_SUCCESS) { return 2; }
        if (arg_to_uint64_def(argc, argv, "--niterw", &niterw, 1) != OTX_SUCCESS) { return 2; }
        if (arg_to_uint64_def(argc, argv, "-bm", &nblocks_m, 1) != OTX_SUCCESS) { return 3; }
        if (arg_to_uint64_def(argc, argv, "-bn", &nblocks_n, 1) != OTX_SUCCESS) { return 4; }
        if (arg_to_uint64_def(argc, argv, "-bl", &nblocks_l, 1) != OTX_SUCCESS) { return 5; }
    }

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

    /* Block the matrices */
    {
        GrB_Matrix_nrows(&m, A);
        GrB_Matrix_ncols(&n, B);
        GrB_Matrix_ncols(&l, A);
        block_size_m = (m + nblocks_m - 1) / nblocks_m;
        block_size_n = (n + nblocks_n - 1) / nblocks_n;
        block_size_l = (l + nblocks_l - 1) / nblocks_l;

        create_blocks(&bA, &bm, &bl, A, block_size_m, block_size_l, 0);
        create_blocks(&bB, &bl, &bn, B, block_size_l, block_size_n, 0);
        create_blocks(&bC, &bm, &bn, C, block_size_m, block_size_n, 0);
    }

//    /* Execute mxm */
    hpctoolkit_sampling_start();
    for (size_t iter = 0; iter < niter + niterw; iter++) {
        unsigned long long start = get_time_ns();

#pragma omp parallel // NOLINT(openmp-use-default-none)
#pragma omp single
        for (size_t i = 0; i < bm; i++) {
            for (size_t j = 0; j < bn; j++) {
                for (size_t k = 0; k < bl; k++) {
#pragma omp task // NOLINT(openmp-use-default-none)
                    {
                        BURBLE(OK(GrB_mxm(bC[i * bn + j], GrB_NULL, GrB_NULL, GrB_PLUS_TIMES_SEMIRING_FP64,
                                          bA[i * bl + k], bB[k * bn + j], GrB_DESC_R)));
                    }
                }
            }
        }

        unsigned long long end = get_time_ns();

        for (size_t i = 0; i < bm; i++) {
            for (size_t j = 0; j < bn; j++) {
                GrB_Matrix_clear(bC[i * bn + j]);
            }
        }
        printf("Execution time: %lf\n", (double) (end - start) / 1e9);
        printf("LOG,%s,%zu,%zu,%zu,%d,%lf,%lf\n", get_file_name(A_path), nblocks_m, nblocks_n, nblocks_l, numThreads,
               (double) (end - start) / 1e9, sum);
    }

    GrB_finalize();

    return 0;
}
