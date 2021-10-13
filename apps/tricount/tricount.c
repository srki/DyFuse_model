#include <otx/otx.h>
#include <mio/mio.h>
#include "../util/GrB_util.h"

//#define TYPE uint32_t
//#define TYPE_FORMAT PRIu32
//#define TYPE_GrB GrB_UINT32
//#define TYPE_ONE GxB_ONE_UINT32
//#define TYPE_PLUS GrB_PLUS_UINT32
//#define TYPE_PLUS_MONOID GrB_PLUS_MONOID_UINT32
//#define TYPE_PLUS_TIMES GxB_PLUS_TIMES_UINT32

#define TYPE uint64_t
#define TYPE_FORMAT PRIu64
#define TYPE_GrB GrB_UINT64
#define TYPE_ONE GxB_ONE_UINT64
#define TYPE_PLUS GrB_PLUS_UINT64
#define TYPE_PLUS_MONOID GrB_PLUS_MONOID_UINT64
#define TYPE_PLUS_TIMES GxB_PLUS_TIMES_UINT64

#include "../util/block_matrix.h"
#include "../util/timer.h"



void
test_blocked(char *name, GrB_Matrix A, GrB_Matrix B, GrB_Matrix M, size_t nblocks_m, size_t nblocks_n, size_t nblocks_l,
             size_t niter, int shuffle) {
    double total_time, mxm_time, reduce_time;
    GrB_Matrix C;
    GrB_Matrix *bA, *bB, *bC, *bM;
    GrB_Index m, n, l;
    GrB_Index bm, bn, bl;
    size_t block_size_m, block_size_n, block_size_l;
    TYPE sum;

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
        for (size_t i = 0; i < bm; i++) {
            for (size_t j = 0; j < bn; j++) {
                for (size_t k = 0; k < bl; k++) {
                    timer_start();
                    BURBLE(OK(GrB_mxm(bC[i * bn + j], bM[i * bn + j], GrB_NULL, TYPE_PLUS_TIMES,
                                      bA[i * bl + k], bB[j * bl + k], GrB_DESC_RT1)));
                    mxm_time += timer_end();

                    timer_start();
                    GrB_reduce(&sum, TYPE_PLUS, TYPE_PLUS_MONOID, bC[i * bn + j], GrB_NULL);
                    reduce_time += timer_end();
                }
            }
        }
        total_time = timer_end();

        printf("LOG,grb,%s,%zu,%zu,%zu,%d,%lf,%lf,%lf,%" TYPE_FORMAT "\n", name,
               nblocks_m, nblocks_n, nblocks_l, shuffle, total_time, mxm_time, reduce_time, sum);
    }
    free_blocks(bA, bm, bl);
    free_blocks(bB, bl, bn);
    free_blocks(bC, bm, bn);
    free_blocks(bM, bm, bn);
    GrB_free(&C);
}

void
test_blocked_individual(char *name, GrB_Matrix A, GrB_Matrix B, GrB_Matrix M, size_t nblocks_m, size_t nblocks_n, size_t nblocks_l,
             size_t niter, int shuffle) {
    double total_time, mxm_time, reduce_time, block_time;
    GrB_Matrix C;
    GrB_Matrix *bA, *bB, *bC, *bM;
    GrB_Index m, n, l;
    GrB_Index bm, bn, bl;
    size_t block_size_m, block_size_n, block_size_l;
    TYPE sum;

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
        for (int nthreads = 1; nthreads <= 12; nthreads++) {
            GxB_Global_Option_set(GxB_GLOBAL_NTHREADS, nthreads);

            for (size_t i = 0; i < bm; i++) {
                for (size_t j = 0; j < bn; j++) {
                    for (size_t k = 0; k < bl; k++) {
                        timer_start();
                        BURBLE(OK(GrB_mxm(bC[i * bn + j], bM[i * bn + j], GrB_NULL, TYPE_PLUS_TIMES,
                                          bA[i * bl + k], bB[j * bl + k], GrB_DESC_RT1)));
                        block_time = timer_end();
                        mxm_time += block_time;
                        printf("LOG-block,grb,%s,%zu,%zu,%zu,%zu,%zu,%zu,%d,%lf,%d\n", name,
                               nblocks_m, nblocks_n, nblocks_l, i, j, k, shuffle, block_time, nthreads);

                        timer_start();
                        GrB_reduce(&sum, TYPE_PLUS, TYPE_PLUS_MONOID, bC[i * bn + j], GrB_NULL);
                        reduce_time += timer_end();
                    }
                }
            }
        }
        total_time = timer_end();

        printf("LOG,grb,%s,%zu,%zu,%zu,%d,%lf,%lf,%lf,%" TYPE_FORMAT "\n", name,
               nblocks_m, nblocks_n, nblocks_l, shuffle, total_time, mxm_time, reduce_time, sum);
    }
    free_blocks(bA, bm, bl);
    free_blocks(bB, bl, bn);
    free_blocks(bC, bm, bn);
    free_blocks(bM, bm, bn);
    GrB_free(&C);
}

int main(int argc, char *argv[]) {
    double total_time, mxm_time, reduce_time;
    int numThreads;
    size_t niter;
    char *A_path;
    int32_t shuffle;
    GrB_Matrix L;
    GrB_Matrix U;
    GrB_Matrix M;
    GrB_Matrix C;
    GrB_Index nrows;
    GrB_Index ncols;
    TYPE sum;
    size_t nblocks;

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

    /* region baseline */
    for (size_t i = 0; i < niter; i++) {
        GrB_Matrix_new(&C, TYPE_GrB, nrows, ncols);

        timer_start();
        BURBLE(GrB_mxm(C, M, GrB_NULL, TYPE_PLUS_TIMES, L, U, GrB_DESC_T1));
        mxm_time = timer_end();

        timer_start();
        GrB_Matrix_reduce_UINT64(&sum, GrB_NULL, TYPE_PLUS_MONOID, C, GrB_NULL);
        reduce_time = timer_end();
        GrB_Matrix_free(&C);

        total_time = mxm_time + reduce_time;

        printf("LOG,grb,%s,0,0,0,%d,%lf,%lf,%lf,%" TYPE_FORMAT "\n", get_file_name(A_path), shuffle, total_time, mxm_time, reduce_time,
               sum);
    }
    /* endregion */

//    test_blocked_individual(get_file_name(A_path), L, U, M, 6, 6, 1, niter, 0);

//    for (size_t i = 0; i < niter; i++) {
//        GrB_Matrix_new(&C, TYPE_GrB, nrows, ncols);
//
//        timer_start();
//        BURBLE(GrB_mxm(C, L, GrB_NULL, TYPE_PLUS_TIMES, L, U, GrB_DESC_T1));
//        mxm_time = timer_end();
//
//        timer_start();
//        GrB_Matrix_reduce_UINT32(&sum, GrB_NULL, TYPE_PLUS_MONOID, C, GrB_NULL);
//        reduce_time = timer_end();
//        GrB_Matrix_free(&C);
//
//        total_time = mxm_time + reduce_time;
//
//        printf("LOG,grb,%s,0,0,0,%d,%lf,%lf,%lf,%" TYPE_FORMAT "\n", get_file_name(A_path), shuffle, total_time, mxm_time, reduce_time,
//               sum);
//    }
//    /* endregion */
//
    /* region blocked */
//    nblocks = 6;
//    while (nblocks <= 6) {
//        test_blocked(get_file_name(A_path), L, U, M, nblocks, nblocks, 1, niter, shuffle);
//
//        if (nblocks == 1) { nblocks = 2; }
//        else if (((nblocks - 1) & nblocks) == 0) { nblocks = nblocks * 3 / 2; }
//        else { nblocks = ((nblocks - 1) & nblocks) * 2; }
//    }
//    nblocks = 1;
//    while (nblocks <= 8) {
//        printf("%ld\n", nblocks);
//        test_blocked(get_file_name(A_path), L, U, M, nblocks, nblocks, nblocks, niter, shuffle);
//
//        if (nblocks == 1) { nblocks = 2; }
//        else if (((nblocks - 1) & nblocks) == 0) { nblocks = nblocks * 3 / 2; }
//        else { nblocks = ((nblocks - 1) & nblocks) * 2; }
//    }

    /* endregion */

    GrB_free(&L);
    GrB_free(&U);
    GrB_free(&M);
    GrB_free(&C);

    return 0;
}