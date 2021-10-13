#include <otx/otx.h>
#include <mio/mio.h>
#include "../util/GrB_util.h"

//#define BURBLE(_CODE) do { GxB_set(GxB_BURBLE, true); _CODE; GxB_set(GxB_BURBLE, false); } while(0);
#define BURBLE(_CODE) _CODE;

#define TYPE uint32_t
#define TYPE_FORMAT PRIu32
#define TYPE_GrB GrB_UINT32
#define TYPE_ONE GxB_ONE_UINT32
#define TYPE_PLUS GrB_PLUS_UINT32
#define TYPE_PLUS_MONOID GrB_PLUS_MONOID_UINT32
#define TYPE_PLUS_TIMES GxB_PLUS_TIMES_UINT32

#include "../util/block_matrix.h"
#include "../util/timer.h"



void test_blocked_omp(const char *name, GrB_Matrix A, GrB_Matrix B, GrB_Matrix M, size_t nblocks_m, size_t nblocks_n,
                      size_t nblocks_l,
                      size_t niter) {
    double total_time, mxm_time, reduce_time;
    GrB_Matrix C;
    GrB_Matrix *bA, *bB, *bC, *bM;
    GrB_Index m, n, l;
    GrB_Index bm, bn, bl;
    size_t block_size_m, block_size_n, block_size_l;
    TYPE sum, psum;

    GrB_Matrix_nrows(&m, A);
    GrB_Matrix_nrows(&n, B);
    GrB_Matrix_ncols(&l, A);
    block_size_m = (m + nblocks_m - 1) / nblocks_m;
    block_size_n = (n + nblocks_n - 1) / nblocks_n;
    block_size_l = (l + nblocks_l - 1) / nblocks_l;

    GrB_Matrix_new(&C, TYPE_GrB, m, n);

    create_blocks(&bA, &bm, &bl, A, block_size_m, block_size_l, 0);
    create_blocks(&bB, &bn, &bl, B, block_size_n, block_size_l, 0);
    create_blocks(&bC, &bm, &bn, C, block_size_m, block_size_n, 0);
    create_blocks(&bM, &bm, &bn, M, block_size_m, block_size_n, 0);

    if (nblocks_m != bm || nblocks_n != bn || nblocks_l != bl) {
        exit(5);
    }

    for (int iter = 0; iter < niter; iter++) {
        total_time = 0.0;
        mxm_time = 0.0;
        reduce_time = 0.0;
        sum = 0;

        timer_start();
#pragma omp parallel private(psum) // NOLINT(openmp-use-default-none)
#pragma omp single
        {
            for (size_t i = 0; i < bm; i++) {
                for (size_t j = 0; j < bn; j++) {
                    for (size_t k = 0; k < bl; k++) {
#pragma omp task // NOLINT(openmp-use-default-none)
                        {
                            BURBLE(OK(GrB_mxm(bC[i * bn + j], bM[i * bn + j], GrB_NULL, TYPE_PLUS_TIMES,
                                              bA[i * bl + k], bB[j * bl + k], GrB_DESC_RT1)));

                            GrB_reduce(&psum, GrB_NULL, TYPE_PLUS_MONOID, bC[i * bn + j], GrB_NULL);
#pragma omp atomic // NOLINT(openmp-use-default-none)
                            sum += psum;
                        }
                    }
                }
            }
        }
        total_time = timer_end();

        printf("LOG,omp,%s,%zu,%zu,%zu,%lf,%lf,%lf,%" TYPE_FORMAT "\n", name,
               nblocks_m, nblocks_n, nblocks_l, total_time, mxm_time, reduce_time, sum);
    }

    free_blocks(bA, bm, bl);
    free_blocks(bB, bl, bn);
    free_blocks(bC, bm, bn);
    free_blocks(bM, bm, bn);
    GrB_free(&C);
}

int main(int argc, char *argv[]) {
    int numThreads;
    size_t niter;
    char *A_path;
    int32_t shuffle;
    GrB_Matrix L;
    GrB_Matrix U;
    GrB_Matrix M;
    GrB_Index nrows;
    GrB_Index ncols;
    uint64_t numBlocksM, numBlocksN, numBlocksK;
    uint64_t blockSizeM, blockSizeN, blockSizeK;

    /* Init and set the number of threads */
    GrB_init(GrB_BLOCKING);
    if (arg_to_int32_def(argc, argv, "-nt|--numThreads", &numThreads, 1) != OTX_SUCCESS) { return 1; }
    GxB_Global_Option_set(GxB_NTHREADS, numThreads);

    if (arg_to_uint64_def(argc, argv, "--niter", &niter, 1) != OTX_SUCCESS) { return 2; }
    if (arg_to_int32_def(argc, argv, "--shuffle", &shuffle, 0) != OTX_SUCCESS) { return 3; }
    if (arg_to_str_def(argc, argv, "-A", &A_path, INPUTS_DIR"/simple") != OTX_SUCCESS) { return 4; }

    readLU(&L, &U, A_path, shuffle, TYPE_GrB);

    GrB_Matrix_dup(&M, L);
    GrB_Matrix_nrows(&nrows, L);
    GrB_Matrix_ncols(&ncols, U);

    /* Determine block size */
    if (arg_to_uint64_def(argc, argv, "-nbm|--numBlocksM", &numBlocksM, 0) != OTX_SUCCESS) { return 5; }
    if (arg_to_uint64_def(argc, argv, "-nbn|--numBlocksN", &numBlocksN, 0) != OTX_SUCCESS) { return 6; }
    if (arg_to_uint64_def(argc, argv, "-nbk|--numBlocksK", &numBlocksK, 0) != OTX_SUCCESS) { return 7; }

    GxB_Global_Option_set(GxB_NTHREADS, 1);
    omp_set_num_threads(numThreads);
    test_blocked_omp(get_file_name(A_path), L, U, M, numBlocksM, numBlocksN, numBlocksK, niter);

    GrB_free(&L);
    GrB_free(&U);
    GrB_free(&M);

    return 0;
}