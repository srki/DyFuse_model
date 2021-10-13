#include <GraphBLAS.h>
#include <otx/otx.h>
#include "../util/GrB_util.h"
#include "../util/reader_c.h"
#define TYPE_GrB GrB_FP64

#include "../util/block_matrix_feature.h"
#include "../util/timer.h"


void baseline(const char *name, GrB_Matrix A, GrB_Matrix B, size_t witer, size_t niter, int numThreads) {
    double time, time_total = 0;
    GrB_Matrix C = GrB_NULL;
    GrB_Index nrowsA, ncolsA, nrowsB, ncolsB;
    GrB_Matrix_nrows(&nrowsA, A);
    GrB_Matrix_ncols(&ncolsA, A);
    GrB_Matrix_nrows(&nrowsB, B);
    GrB_Matrix_ncols(&ncolsB, B);

    double result = 0;
    for (size_t iter = 0; iter < witer + niter; ++iter) {
        timer_start();
        result = 0;
        OK(GrB_Matrix_new(&C, GrB_FP64, nrowsA * nrowsB, ncolsA * ncolsB));
        OK(GrB_kronecker(C, GrB_NULL, GrB_NULL, GrB_TIMES_FP64, A, B, GrB_NULL));
        OK(GrB_Matrix_reduce_FP64(&result, GrB_NULL, GrB_PLUS_MONOID_FP64, C, GrB_NULL));
        GrB_Matrix_free(&C);
        time = timer_end();

        if (iter >= witer) { time_total += time; }
    }

    printf("LOG;%s;baseline;%lf;%lf\n", name, time_total / (double) niter, result);
}

void blocked(const char *name, GrB_Matrix A, GrB_Matrix B, size_t witer, size_t niter, size_t nblocksI, size_t nblocksJ,
             int numThreads, double* featuresM) {
    double time, time_total = 0;

    GrB_Matrix *bA;

    GrB_Index nrowsA, ncolsA, nrowsB, ncolsB;
    GrB_Matrix_nrows(&nrowsA, A);
    GrB_Matrix_ncols(&ncolsA, A);
    GrB_Matrix_nrows(&nrowsB, B);
    GrB_Matrix_ncols(&ncolsB, B);

    size_t nrows_block_A, ncols_block_A;
    nrows_block_A = (nrowsA + nblocksI - 1) / nblocksI;
    ncols_block_A = (nrowsB + nblocksJ - 1) / nblocksJ;


    size_t nrows_blocked_A, ncols_blocked_A;
    create_blocks_type(&bA, &nrows_blocked_A, &ncols_blocked_A, A, nrows_block_A, ncols_block_A, false, GrB_FP64,featuresM);


    double result = 0;
    for (size_t iter = 0; iter < witer + niter; ++iter) {
        timer_start();
        result = 0;
        for (size_t i = 0; i < nrows_blocked_A; ++i) {
            for (size_t j = 0; j < ncols_blocked_A; ++j) {
                GrB_Matrix block_A = bA[i * ncols_blocked_A + j];
                GrB_Index nrows_A_block;
                GrB_Index ncols_A_block;
                GrB_Index nrows_B_block;
                GrB_Index ncols_B_block;
                GrB_Matrix_nrows(&nrows_A_block, block_A);
                GrB_Matrix_ncols(&ncols_A_block, block_A);
                //GrB_Matrix_nrows(&nrows_B_block, B);
                //GrB_Matrix_ncols(&ncols_B_block, B);

                //printf("%s,blocks=%lu,nrowsA=%lu,ncolsA=%lu,nnzA=%lf,nrowsB=%lu,ncolsB=%lu,nrowsC=%lu,ncolsC=%lu\n",
                //       name, nblocksI, nrows_A_block,ncols_A_block,featuresM[4*(i * ncols_blocked_A + j)+3],
                //       nrows_B_block,ncols_B_block,
                //       nrows_A_block * nrowsB, ncols_A_block * ncolsB);
                GrB_Matrix block_C;
                OK(GrB_Matrix_new(&block_C, GrB_FP64, nrows_A_block * nrowsB, ncols_A_block * ncolsB));

                OK(GrB_Matrix_kronecker_BinaryOp(block_C, GrB_NULL, GrB_NULL, GrB_TIMES_FP64, block_A, B, GrB_NULL));
                GrB_Matrix_reduce_FP64(&result, GrB_PLUS_FP64, GrB_PLUS_MONOID_FP64, block_C, GrB_NULL);
                GrB_Matrix_free(&block_C);
            }
        }
        time = timer_end();

        if (iter >= witer) { time_total += time; }
    }
    free_blocks(bA, nrows_blocked_A, ncols_blocked_A);

    printf("LOG;%s;%zux%zu;%lf;%lf\n", name, nblocksI, nblocksJ, time_total / (double) niter, result);
}

int main(int argc, char *argv[]) {
    char *A_path;
    uint64_t witer, niter;
    int num_threads;
    size_t nblocks, min_blocks, max_blocks;
    uint8_t block_row, block_col;
    GrB_Matrix A;

    /* region read program arguments */
    if (arg_to_str_def(argc, argv, "-A", &A_path, INPUTS_DIR"/simple") != OTX_SUCCESS) { return 1; }
    if (arg_to_uint64_def(argc, argv, "--niter", &niter, 1) != OTX_SUCCESS) { return 2; }
    if (arg_to_uint64_def(argc, argv, "--witer", &witer, 1) != OTX_SUCCESS) { return 2; }
    if (arg_to_int32_def(argc, argv, "-nt|--nthreads", &num_threads, 1) != OTX_SUCCESS) { return 3; }
    if (arg_to_uint64_def(argc, argv, "-minb|--minBlocks", &min_blocks, 1) != OTX_SUCCESS) { return 4; }
    if (arg_to_uint64_def(argc, argv, "-maxb|--maxBlocks", &max_blocks, 4) != OTX_SUCCESS) { return 5; }
    if (arg_to_uint8_def(argc, argv, "--blockRow", &block_row, 1) != OTX_SUCCESS) { return 6; }
    if (arg_to_uint8_def(argc, argv, "--blockCol", &block_col, 1) != OTX_SUCCESS) { return 6; }
    /* endregion */

    //printf("kron; matrix: %s, num iterations: %lu-%lu, num threads: %d; blocks: %lu-%lu\n",
    //       get_file_name(A_path), witer, niter, num_threads, min_blocks, max_blocks);

    /* region init libraries */
    GrB_init(GrB_BLOCKING);
    GxB_Global_Option_set(GxB_NTHREADS, num_threads);
    /* endregion */

    if (read_Matrix_FP64(&A, A_path, false) != GrB_SUCCESS) { return 16; }

    baseline(get_file_name(A_path), A, A, witer, niter, num_threads);

    nblocks = min_blocks;
    while (nblocks <= max_blocks) {
        double *featuresM;
        if (!(featuresM = (double *) malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double))))
            ABORT("Malloc fails for featuresM[].");
        //blocked(get_file_name(A_path), A, A, witer, niter, nblocks, nblocks, num_threads,featuresM);
        blocked(get_file_name(A_path), A, A, witer, niter, block_row ? nblocks : 1, block_col ? nblocks : 1,num_threads,featuresM);

        if (nblocks == 1) { nblocks = 2; }
        else if (((nblocks - 1) & nblocks) == 0) { nblocks = nblocks * 3 / 2; }
        else { nblocks = ((nblocks - 1) & nblocks) * 2; }
    }

    GrB_Matrix_free(&A);
    GrB_finalize();
}