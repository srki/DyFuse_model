#include <GraphBLAS.h>
#include <otx/atx.h>
#include "../../util/reader_c.h"
#include "../../util/timer.h"

#define OK(A) do {GrB_Info inf = A; if (inf != GrB_SUCCESS) { fprintf(stderr, "Error, line: %d \n", __LINE__); } } while(0);
#define TYPE uint64_t
#define TYPE_FORMAT PRIu64
#define TYPE_GrB GrB_UINT64
#define TYPE_ONE GxB_ONE_UINT64
#define TYPE_PLUS GrB_PLUS_UINT64
#define TYPE_PLUS_MONOID GrB_PLUS_MONOID_UINT64
#define TYPE_PLUS_TIMES GxB_PLUS_TIMES_UINT64

#include "../../util/block_matrix.h"

/* region Create blocks function declarations */

//GrB_Info create_blocks(GrB_Matrix **bA_out, GrB_Index *nrows_blocked_out, GrB_Index *ncols_blocked_out,
//                       GrB_Matrix A, size_t nrows_block, size_t ncols_block, bool full_idxes);
//
//GrB_Info free_blocks(GrB_Matrix *bA, size_t nrows_blocked, size_t ncols_blocked);

/* endregion */

//#define BURBLE(_CODE) do { GxB_set(GxB_BURBLE, true); _CODE; GxB_set(GxB_BURBLE, false); } while(0);
#define BURBLE(_CODE) _CODE;

//#define TYPE double
//#define TYPE_FORMAT "lf"
//#define TYPE_GrB GrB_FP64
//#define TYPE_PLUS GrB_PLUS_FP64
//#define TYPE_PLUS_MONOID GrB_PLUS_MONOID_FP64
//#define TYPE_PLUS_TIMES GxB_PLUS_TIMES_FP64




/* region loops */

#define BASIC_IJK_LOOP(_M, _N, _L, _BODY)           \
    for (size_t i = 0; i < _M; i++) {               \
        for (size_t j = 0; j < _N; j++) {           \
            for (size_t k = 0; k < _L; k++) {       \
                _BODY;                              \
            }                                       \
        }                                           \
    }

#define ZORDER_IJK_LOOP(_M, _N, _L, _BODY)                      \
    {                                                           \
        size_t _p[] = {0, 0, 0};                                \
        {                                                       \
            size_t i = _p[2], j = _p[1], k = _p[0];             \
            _BODY;                                              \
        }                                                       \
                                                                \
        while (true) {                                          \
            size_t carry = 0x1;                                 \
            do {                                                \
                for (size_t idx = 0; idx < 3; idx++) {          \
                    /* cary, x = x & cary, x ^ carry */         \
                    size_t tmpCarry = carry;                    \
                    carry = _p[idx] & carry;                    \
                    _p[idx] ^= tmpCarry;                        \
                }                                               \
                                                                \
                carry <<= 1u;                                   \
            } while (carry != 0);                               \
                                                                \
            size_t i = _p[2], j = _p[1], k = _p[0];             \
                                                                \
            if (i >= _M && j >= _N && k >= _L) { break; }       \
            if (i >= _M || j >= _N || k >= _L) { continue; }    \
                                                                \
            _BODY;                                              \
        }                                                       \
    }

#define IJK_LOOP BASIC_IJK_LOOP

/* endregion */

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

char *get_file_name(char *path) {
    size_t len = strlen(path);
    while (path[len] != '/') { len--; }
    return path + len + 1;
}

void test(char * name, GrB_Matrix A, GrB_Matrix B, size_t niter, int shuffle) {
    double total_time, mxm_time, reduce_time;
    GrB_Matrix C;
    GrB_Index m, n;
    TYPE sum;

    GrB_Matrix_nrows(&m, A);
    GrB_Matrix_ncols(&n, B);
    GrB_Matrix_new(&C, TYPE_GrB, m, n);

    for (int iter = 0; iter < niter; iter++) {
        timer_start();
        {
            timer_start();
            BURBLE(OK(GrB_mxm(C, GrB_NULL, GrB_NULL, TYPE_PLUS_TIMES, A, B, GrB_DESC_R)));
            mxm_time = timer_end();
        }

        {
            timer_start();
            OK(GrB_reduce(&sum, GrB_NULL, TYPE_PLUS_MONOID, C, GrB_NULL));
            reduce_time = timer_end();
        }

        total_time = timer_end();
        printf("LOG-A,grb,%s,0,0,0,%d,%lf,%lf,%lf,%" TYPE_FORMAT "\n", name,
               shuffle, total_time, mxm_time, reduce_time, sum);
    }

    GrB_Matrix_free(&C);
}

void test_blocked(char *name, GrB_Matrix A, GrB_Matrix B, size_t nblocks_m, size_t nblocks_n, size_t nblocks_l,
                  size_t niter, int shuffle) {
    printf("Start\n");
    double total_time, mxm_time, reduce_time;
    GrB_Matrix C;
    GrB_Matrix *bA, *bB, *bC;
    GrB_Index m, n, l;
    GrB_Index bm, bn, bl;
    size_t block_size_m, block_size_n, block_size_l;
    TYPE sum;

    GrB_Matrix_nrows(&m, A);
    GrB_Matrix_ncols(&n, B);
    GrB_Matrix_ncols(&l, A);
    block_size_m = (m + nblocks_m - 1) / nblocks_m;
    block_size_n = (n + nblocks_n - 1) / nblocks_n;
    block_size_l = (l + nblocks_l - 1) / nblocks_l;

    GrB_Matrix_new(&C, TYPE_GrB, m, n);

    create_blocks(&bA, &bm, &bl, A, block_size_m, block_size_l, 0);
    create_blocks(&bB, &bl, &bn, B, block_size_l, block_size_n, 0);
    create_blocks(&bC, &bm, &bn, C, block_size_m, block_size_n, 0);
    printf("Blocked\n");

    if (nblocks_m != bm || nblocks_n != bn || nblocks_l != bl) {
        exit(5);
    }

    for (int iter = 0; iter < niter; iter++) {
        total_time = 0.0;
        mxm_time = 0.0;
        reduce_time = 0.0;
        sum = 0;
        timer_start();

        IJK_LOOP(bm, bn, bl, {
            timer_start();
            BURBLE(OK(GrB_mxm(bC[i * bn + j], GrB_NULL, GrB_NULL, TYPE_PLUS_TIMES,
                              bA[i * bl + k], bB[k * bn + j], GrB_DESC_R)));
            mxm_time += timer_end();

            timer_start();
            GrB_reduce(&sum, TYPE_PLUS, TYPE_PLUS_MONOID, bC[i * bn + j], GrB_NULL);
            reduce_time += timer_end();
        })

        total_time = timer_end();

        printf("LOG-A,grb,%s,%zu,%zu,%zu,%d,%lf,%lf,%lf,%" TYPE_FORMAT "\n", name,
               nblocks_m, nblocks_n, nblocks_l, shuffle, total_time, mxm_time, reduce_time, sum);
    }
    free_blocks(bA, bm, bl);
    free_blocks(bB, bl, bn);
    free_blocks(bC, bm, bn);
    GrB_free(&C);
}

int main(int argc, char *argv[]) {
    size_t nblocks;
    double total_time, mxm_time, reduce_time;
    int numThreads;
    size_t niter;
    char *A_path;
    int shuffle;
    GrB_Index nrows, ncols;
    GrB_Matrix A;
    GrB_Matrix B;
    GrB_Matrix C;
    uint64_t flops_AxA;
    GrB_Index nzA;
    GrB_Index nzAxA;

    /* Init and set the number of threads */
    GrB_init(GrB_NONBLOCKING);
    if (arg_to_int32_def(argc, argv, "-nt|--numThreads", &numThreads, 1) != OTX_SUCCESS) { return 1; }
    GxB_Global_Option_set(GxB_GLOBAL_NTHREADS, numThreads);

    if (arg_to_uint64_def(argc, argv, "-ni|--niter", &niter, 1) != OTX_SUCCESS) { return 1; }
    if (arg_to_int32_def(argc, argv, "--shuffle", &shuffle, 0) != OTX_SUCCESS) { return 3; }

    /* Read the matrices */
    if (arg_to_str_def(argc, argv, "-A", &A_path, INPUTS_DIR"/simple") != OTX_SUCCESS) { return 1; }
    if (read_Matrix_FP64(&A, A_path, shuffle) != GrB_SUCCESS) { return 1; }
    cast_matrix(&A, GrB_UINT64, GxB_ONE_UINT64);
    OK(GrB_Matrix_dup(&B, A));

    GrB_Matrix_nvals(&nzA, A);
    GrB_Matrix_nrows(&nrows, A);
    GrB_Matrix_ncols(&ncols, A);

//    /* region Calculate stats */
//
//    if (nrows == ncols) {
//        OK(GrB_Matrix_new(&C, GrB_UINT64, nrows, ncols));
//        BURBLE(OK(GrB_mxm(C, GrB_NULL, GrB_NULL, GrB_PLUS_TIMES_SEMIRING_UINT64, A, A, GrB_NULL)));
//        OK(GrB_Matrix_reduce_UINT64(&flops_AxA, GrB_NULL, GrB_PLUS_MONOID_UINT64, C, GrB_NULL));
//        OK(GrB_Matrix_nvals(&nzAxA, C));
//        OK(GrB_Matrix_free(&C));
//    }
//
//    /* endregion */
//
//    printf("LOG-B,%s,%lu,%lu,%lu,%lu,%lu\n",
//           get_file_name(A_path), nrows, ncols, nzA, flops_AxA, nzAxA);



    test(get_file_name(A_path), A, B, niter, shuffle);

    nblocks = 6;
    while (nblocks <= 6) {
        test_blocked(get_file_name(A_path), A, B, nblocks, nblocks, 1, niter, shuffle);


        if (nblocks == 1) { nblocks = 2; }
        else if (((nblocks - 1) & nblocks) == 0) { nblocks = nblocks * 3 / 2; }
        else { nblocks = ((nblocks - 1) & nblocks) * 2; }
    }


    return 0;
}
