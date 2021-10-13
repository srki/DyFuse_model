#include <GraphBLAS.h>
#include <otx/atx.h>
#include "../../util/reader_c.h"
#include "../../util/timer.h"

#define TYPE uint64_t
#define TYPE_FORMAT PRIu64
#define TYPE_GrB GrB_UINT64
#define TYPE_ONE GxB_ONE_UINT64
#define TYPE_PLUS GrB_PLUS_UINT64
#define TYPE_PLUS_MONOID GrB_PLUS_MONOID_UINT64
#define TYPE_PLUS_TIMES GxB_PLUS_TIMES_UINT64
#include "../../util/GrB_util.h"

/* region Create blocks function declarations */

GrB_Info create_blocks(GrB_Matrix **bA_out, GrB_Index *nrows_blocked_out, GrB_Index *ncols_blocked_out,
                       GrB_Matrix A, size_t nrows_block, size_t ncols_block, bool full_idxes);

GrB_Info free_blocks(GrB_Matrix *bA, size_t nrows_blocked, size_t ncols_blocked);

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


void test(const char *name, GrB_Matrix A, GrB_Matrix B, size_t niter, int shuffle) {
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
        printf("LOG-A,omp,%s,0,0,0,%d,%lf,%lf,%lf,%" TYPE_FORMAT "\n", name,
               shuffle, total_time, mxm_time, reduce_time, sum);
    }

    GrB_Matrix_free(&C);
}

void test_blocked(const char *name, GrB_Matrix A, GrB_Matrix B, size_t nblocks_m, size_t nblocks_n, size_t nblocks_l,
                  size_t niter, int shuffle) {
    double total_time, mxm_time, reduce_time;
    GrB_Matrix C;
    GrB_Matrix *bA, *bB, *bC;
    GrB_Index m, n, l;
    GrB_Index bm, bn, bl;
    size_t block_size_m, block_size_n, block_size_l;
    TYPE sum, psum;

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
                            BURBLE(OK(GrB_mxm(bC[i * bn + j], GrB_NULL, GrB_NULL, TYPE_PLUS_TIMES,
                                              bA[i * bl + k], bB[k * bn + j], GrB_DESC_R)));

                            GrB_reduce(&psum, TYPE_PLUS, TYPE_PLUS_MONOID, bC[i * bn + j], GrB_NULL);
#pragma omp atomic // NOLINT(openmp-use-default-none)
                            sum += psum;
                        }
                    }
                }
            }
        }

        total_time = timer_end();

        printf("LOG-A,omp,%s,%zu,%zu,%zu,%d,%lf,%lf,%lf,%" TYPE_FORMAT "\n", name,
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
    GrB_Index nzA;

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

    /* region blocked */
    GxB_Global_Option_set(GxB_NTHREADS, 1);
    omp_set_num_threads(numThreads);

    nblocks = 1;
    while (nblocks <= 64) {
        if (nblocks * nblocks >= numThreads) {
            test_blocked(get_file_name(A_path), A, B, nblocks, nblocks, 1, niter, shuffle);
        }

        if (nblocks == 1) { nblocks = 2; }
        else if (((nblocks - 1) & nblocks) == 0) { nblocks = nblocks * 3 / 2; }
        else { nblocks = ((nblocks - 1) & nblocks) * 2; }
    }
    /* endregion */

    GrB_free(&A);
    GrB_free(&B);

    return 0;
}

/* region Create blocks function definitions */

GrB_Info create_blocks(GrB_Matrix **bA_out, GrB_Index *nrows_blocked_out, GrB_Index *ncols_blocked_out,
                       GrB_Matrix A, size_t nrows_block, size_t ncols_block, bool full_idxes) {
    /* Input matrix tuples */
    GrB_Index *I, *J;
    uint64_t *X;

    /* Blocked tuples and matrix blocks */
    GrB_Index **bI, **bJ;
    uint64_t **bX;
    GrB_Matrix *bA;

    GrB_Index nrows, ncols, nvals;
    size_t nblocks, nrows_blocked, ncols_blocked;
    size_t *nzb_cnt;

    /* Get input matrix info and calculate the number of blocks */
    OK(GrB_Matrix_nvals(&nvals, A));
    OK(GrB_Matrix_nrows(&nrows, A));
    OK(GrB_Matrix_ncols(&ncols, A));

    if (nrows_block == 0) { GrB_Matrix_nrows(&nrows_block, A); }
    if (ncols_block == 0) { GrB_Matrix_ncols(&ncols_block, A); }

    nrows_blocked = ((nrows + nrows_block - 1) / nrows_block);
    ncols_blocked = ((ncols + ncols_block - 1) / ncols_block);
    nblocks = nrows_blocked * ncols_blocked;

    /* Allocate the output arrays */
    nzb_cnt = calloc(nblocks, sizeof(size_t));
    bI = malloc(nblocks * sizeof(GrB_Index *));
    bJ = malloc(nblocks * sizeof(GrB_Index *));
    bX = malloc(nblocks * sizeof(uint64_t *));
    if (nzb_cnt == NULL || bI == NULL || bJ == NULL) { exit(1); }

    /* Get tuples form the input matrix */
    I = malloc((nvals + 1) * sizeof(GrB_Index));
    J = malloc((nvals + 1) * sizeof(GrB_Index));
    X = malloc((nvals + 1) * sizeof(uint64_t));
    if (I == NULL || J == NULL || X == NULL) { exit(1); }
    OK(GrB_Matrix_extractTuples(I, J, X, &nvals, A));

    /* Calculate the number of elements per block */
    for (GrB_Index i = 0; i < nvals; i++) {
        GrB_Index row_idx = I[i] / nrows_block;
        GrB_Index col_idx = J[i] / ncols_block;
        nzb_cnt[row_idx * ncols_blocked + col_idx]++;
    }

    /* Allocate memory for block tuples */
    for (size_t i = 0; i < nblocks; i++) {
        bI[i] = malloc(nzb_cnt[i] * sizeof(GrB_Index));
        bJ[i] = malloc(nzb_cnt[i] * sizeof(GrB_Index));
        bX[i] = malloc(nzb_cnt[i] * sizeof(uint64_t));
        if (bI[i] == NULL || bJ[i] == NULL || bX[i] == NULL) { exit(1); }
    }

    /* Fill the block tuples */
    memset(nzb_cnt, 0, nblocks * sizeof(size_t));
    for (GrB_Index i = 0; i < nvals; i++) {
        GrB_Index row_idx = I[i] / nrows_block;
        GrB_Index col_idx = J[i] / ncols_block;
        GrB_Index block_idx = row_idx * ncols_blocked + col_idx;
        GrB_Index idx = nzb_cnt[block_idx]++;

        if (full_idxes) {
            bI[block_idx][idx] = I[i];
            bJ[block_idx][idx] = J[i];
        } else {
            bI[block_idx][idx] = I[i] % nrows_block;
            bJ[block_idx][idx] = J[i] % ncols_block;
        }

        bX[block_idx][idx] = X[i];
    }

    /* Allocate the output matrices */
    bA = malloc(nblocks * sizeof(GrB_Matrix));
    for (size_t i = 0; i < nblocks; i++) {
//        if (nzb_cnt[i] == 0) {
//            bA[i] = NULL;
//            continue;
//        }

        if (full_idxes) {
            OK(GrB_Matrix_new(&bA[i], TYPE_GrB, nrows, ncols));
        } else {
            OK(GrB_Matrix_new(&bA[i], TYPE_GrB, nrows_block, ncols_block));
        }
        OK(GrB_Matrix_build(bA[i], bI[i], bJ[i], bX[i], nzb_cnt[i], GrB_PLUS_INT64));
    }

//    GxB_print(A, GxB_COMPLETE);
//    for (size_t bi = 0; bi < nrows_blocked; bi++) {
//        for (size_t bj = 0; bj < ncols_blocked; bj++) {
//            printf("Block (%lu, %lu): \n", bi, bj);
//
//            GrB_Index block_idx = bi * ncols_blocked + bj;
//            for (size_t i = 0; i < nzb_cnt[block_idx]; i++) {
//                printf("\t(%lu, %lu) = %u\n", bI[block_idx][i], bJ[block_idx][i], bX[block_idx][i]);
//            }
//
//            if (bA[block_idx] != NULL) {
//                GxB_print(bA[block_idx], GxB_COMPLETE);
//            }
//            printf("--------------------------------\n");
//        }
//    }


    /* region Deallocation */
    for (size_t i = 0; i < nblocks; i++) {
        free(bI[i]);
        free(bJ[i]);
        free(bX[i]);
    }

    free(nzb_cnt);
    free(bI);
    free(bJ);
    free(bX);

    free(I);
    free(J);
    free(X);
    /* endregion */

    /* Set the output values */
    *bA_out = bA;
    *nrows_blocked_out = nrows_blocked;
    *ncols_blocked_out = ncols_blocked;

    return GrB_SUCCESS;
}

GrB_Info free_blocks(GrB_Matrix *bA, size_t nrows_blocked, size_t ncols_blocked) {
    size_t nblocks = nrows_blocked * ncols_blocked;
    for (size_t i = 0; i < nblocks; i++) {
        if (bA[i] != NULL) {
            OK(GrB_Matrix_free(&bA[i]));
        }
    }
    free(bA);

    return GrB_SUCCESS;
}

/* endregion */