#include <GraphBLAS.h>
#include <otx/otx.h>
#include "../../util/reader_c.h"
#include "../../util/timer.h"

#define TYPE double
#define TYPE_GrB GrB_FP64
#define TYPE_PLUS GrB_PLUS_FP64
#define TYPE_PLUS_MONOID GrB_PLUS_MONOID_FP64
#define TYPE_PLUS_TIMES GxB_PLUS_TIMES_FP64

#define OK(A) do {GrB_Info inf = A; if (inf != GrB_SUCCESS) { fprintf(stderr, "Error, line: %d \n", __LINE__); } } while(0);

#define BASIC_IJK_LOOP(_M, _N, _L, _BODY)           \
    for (size_t i = 0; i < _M; i++) {               \
        for (size_t j = 0; j < _N; j++) {           \
            for (size_t k = 0; k < _L; k++) {       \
                _BODY;                              \
            }                                       \
        }                                           \
    }

#define IJK_LOOP BASIC_IJK_LOOP

#define RUN(_NITER, _MAX_NUM_BLOCKS, _BODY)                 \
    for (size_t b = 1; b <= _MAX_NUM_BLOCKS; b *= 2) {      \
        for (int i = 0; i < _NITER; i++) {                  \
            _BODY;                                          \
        }                                                   \
    }


GrB_Info create_blocks(GrB_Matrix **bA_out, GrB_Index *nrows_blocked_out, GrB_Index *ncols_blocked_out,
                       GrB_Matrix A, size_t nrows_block, size_t ncols_block, bool full_idxes);

GrB_Info free_blocks(GrB_Matrix *bA, size_t nrows_blocked, size_t ncols_blocked);

void test0(GrB_Matrix A, GrB_Matrix B) {
    double mxm_t, reduce_t;
    GrB_Matrix C;
    GrB_Index m, n;
    double sum;

    GrB_Matrix_nrows(&m, A);
    GrB_Matrix_ncols(&n, B);

    GrB_Matrix_new(&C, TYPE_GrB, m, n);
    timer_start();
    GrB_mxm(C, GrB_NULL, GrB_NULL, TYPE_PLUS_TIMES, A, B, GrB_NULL);
    mxm_t = timer_end();

    timer_start();
    GrB_reduce(&sum, GrB_NULL, TYPE_PLUS_MONOID, C, GrB_NULL);
    reduce_t = timer_end();

    printf("%s,0,%lf,%lf,%lf\n", __FUNCTION__, mxm_t, reduce_t, sum);

    GrB_Matrix_free(&C);
}

void test1(GrB_Matrix A, GrB_Matrix B, size_t nblocks) {
    double mxm_t, reduce_t, accum_t;
    GrB_Matrix C;
    GrB_Matrix *bA, *bB, *bC;
    GrB_Index m, n, l;
    GrB_Index bm, bn, bl;
    size_t blockSize;
    double sum;

    GrB_Matrix_nrows(&m, A);
    GrB_Matrix_ncols(&n, B);
    GrB_Matrix_ncols(&l, A);
    blockSize = (m + nblocks - 1) / nblocks;

    GrB_Matrix_new(&C, TYPE_GrB, m, n);

    create_blocks(&bA, &bm, &bl, A, blockSize, blockSize, 0);
    create_blocks(&bB, &bl, &bn, B, blockSize, blockSize, 0);
    create_blocks(&bC, &bm, &bn, C, blockSize, blockSize, 0);

    timer_start();
    IJK_LOOP(bm, bn, bl, {
        OK(GrB_mxm(bC[i * bn + j], GrB_NULL, TYPE_PLUS, TYPE_PLUS_TIMES,
                   bA[i * bl + k], bB[k * bn + j], GrB_NULL));
    });
    mxm_t = timer_end();

    timer_start();
    sum = 0;
    for (size_t i = 0; i < bm; i++) {
        for (size_t j = 0; j < bn; j++) {
            double tmp;
            GrB_reduce(&tmp, GrB_NULL, TYPE_PLUS_MONOID, bC[i * bn + j], GrB_NULL);
            sum += tmp;
        }
    }
    reduce_t = timer_end();

    printf("%s,%zu,%zu,%lf,%lf,%lf\n", __FUNCTION__, nblocks, nblocks, mxm_t, reduce_t, sum);

    free_blocks(bA, bm, bl);
    free_blocks(bB, bl, bn);
    free_blocks(bC, bm, bn);
    GrB_free(&C);
}

/*
 * Does not accumulate directly
 */
void test2(GrB_Matrix A, GrB_Matrix B, size_t nblocks) {
    double mxm_t, reduce_t, accum_t;
    GrB_Matrix C;
    GrB_Matrix *bA, *bB, *bC, *tbC;
    GrB_Index m, n, l;
    GrB_Index bm, bn, bl;
    size_t blockSize;
    double sum;

    GrB_Matrix_nrows(&m, A);
    GrB_Matrix_ncols(&n, B);
    GrB_Matrix_ncols(&l, A);
    blockSize = (m + nblocks - 1) / nblocks;

    GrB_Matrix_new(&C, TYPE_GrB, m, n);

    create_blocks(&bA, &bm, &bl, A, blockSize, blockSize, 0);
    create_blocks(&bB, &bl, &bn, B, blockSize, blockSize, 0);
    create_blocks(&bC, &bm, &bn, C, blockSize, blockSize, 0);
    create_blocks(&tbC, &bm, &bn, C, blockSize, blockSize, 0);

    accum_t = 0;
    timer_start();
    IJK_LOOP(bm, bn, bl, {
        OK(GrB_mxm(tbC[i * bn + j], GrB_NULL, GrB_NULL, TYPE_PLUS_TIMES,
                   bA[i * bl + k], bB[k * bn + j], GrB_NULL));

        timer_start();
        OK(GrB_Matrix_eWiseAdd_BinaryOp(bC[i * bn + j], GrB_NULL, GrB_NULL, TYPE_PLUS,
                                        bC[i * bn + j], tbC[i * bn + j], GrB_NULL));
        accum_t += timer_end();
    })
    mxm_t = timer_end();

    timer_start();
    sum = 0;
    for (size_t i = 0; i < bm; i++) {
        for (size_t j = 0; j < bn; j++) {
            double tmp;
            GrB_reduce(&tmp, GrB_NULL, TYPE_PLUS_MONOID, bC[i * bn + j], GrB_NULL);
            sum += tmp;
        }
    }
    reduce_t = timer_end();

    printf("%s,%zu,%zu,%lf,%lf,%lf,%lf\n", __FUNCTION__, nblocks, nblocks, mxm_t, reduce_t, accum_t, sum);

    free_blocks(bA, bm, bl);
    free_blocks(bB, bl, bn);
    free_blocks(bC, bm, bn);
    free_blocks(tbC, bm, bn);
    GrB_free(&C);
}

void test3(GrB_Matrix A, GrB_Matrix B, size_t nblocks) {
    double mxm_t, reduce_t;
    GrB_Matrix C;
    GrB_Matrix *bA, *bB, *bC;
    GrB_Index m, n, l;
    GrB_Index bm, bn, bl;
    size_t blockSize;
    double sum;

    GrB_Matrix_nrows(&m, A);
    GrB_Matrix_ncols(&n, B);
    GrB_Matrix_ncols(&l, A);
    blockSize = (m + nblocks - 1) / nblocks;

    GrB_Matrix_new(&C, TYPE_GrB, m, n);

    create_blocks(&bA, &bm, &bl, A, blockSize, blockSize, 0);
    create_blocks(&bB, &bl, &bn, B, blockSize, blockSize, 0);
    create_blocks(&bC, &bm, &bn, C, blockSize, blockSize, 0);

    sum = 0;
    reduce_t = 0.0;
    timer_start();
    IJK_LOOP(bm, bn, bl, {
        OK(GrB_mxm(bC[i * bn + j], GrB_NULL, GrB_NULL, TYPE_PLUS_TIMES,
                   bA[i * bl + k], bB[k * bn + j], GrB_NULL));

        double tmp;
        timer_start();
        GrB_reduce(&tmp, GrB_NULL, TYPE_PLUS_MONOID, bC[i * bn + j], GrB_NULL);
        reduce_t += timer_end();
        sum += tmp;
    })
    mxm_t = timer_end();

    printf("%s,%zu,%zu,%lf,%lf,%lf\n", __FUNCTION__, nblocks, nblocks, mxm_t, reduce_t, sum);

    free_blocks(bA, bm, bl);
    free_blocks(bB, bl, bn);
    free_blocks(bC, bm, bn);
    GrB_free(&C);
}

void test4(GrB_Matrix A, GrB_Matrix B, size_t nblocks) {
    double mxm_t, reduce_t, accum_t;
    GrB_Matrix C;
    GrB_Matrix *bA, *bB, *bC;
    GrB_Index m, n, l;
    GrB_Index bm, bn, bl;
    size_t blockSize;
    double sum;

    GrB_Matrix_nrows(&m, A);
    GrB_Matrix_ncols(&n, B);
    GrB_Matrix_ncols(&l, A);
    blockSize = (m + nblocks - 1) / nblocks;

    GrB_Matrix_new(&C, TYPE_GrB, m, n);

    create_blocks(&bA, &bm, &bl, A, blockSize, 0, 0);
    create_blocks(&bB, &bl, &bn, B, 0, blockSize, 0);
    create_blocks(&bC, &bm, &bn, C, blockSize, blockSize, 0);

    timer_start();
    IJK_LOOP(bm, bn, bl, {
        OK(GrB_mxm(bC[i * bn + j], GrB_NULL, GrB_NULL, TYPE_PLUS_TIMES,
                   bA[i * bl + k], bB[k * bn + j], GrB_NULL));
    });

//#pragma omp parallel
//#pragma omp single
//    for (size_t i = 0; i < bm; i++) {
//        for (size_t j = 0; j < bn; j++) {
//            for (size_t k = 0; k < bl; k++) {
//#pragma omp task
//                {
//                        GrB_Info inf = GrB_mxm(bC[i * bn + j], ((void *) 0), ((void *) 0), GxB_PLUS_TIMES_FP64,
//                                               bA[i * bl + k], bB[k * bn + j], ((void *) 0));
//                }
//            }
//        }
//    }
    mxm_t = timer_end();


    timer_start();
    sum = 0;
    for (size_t i = 0; i < bm; i++) {
        for (size_t j = 0; j < bn; j++) {
            double tmp;
            GrB_reduce(&tmp, GrB_NULL, TYPE_PLUS_MONOID, bC[i * bn + j], GrB_NULL);
            sum += tmp;
        }
    }
    reduce_t = timer_end();

    printf("%s,%zu,%zu,%lf,%lf,%lf\n", __FUNCTION__, nblocks, nblocks, mxm_t, reduce_t, sum);

    free_blocks(bA, bm, bl);
    free_blocks(bB, bl, bn);
    free_blocks(bC, bm, bn);
    GrB_free(&C);
}

void test5(GrB_Matrix A, GrB_Matrix B, size_t nblocks) {
    double mxm_t, reduce_t, accum_t;
    GrB_Matrix C;
    GrB_Matrix *bA, *bB, *bC;
    GrB_Index m, n, l;
    GrB_Index bm, bn, bl;
    size_t blockSize;
    double sum;

    GrB_Matrix_nrows(&m, A);
    GrB_Matrix_ncols(&n, B);
    GrB_Matrix_ncols(&l, A);
    blockSize = (m + nblocks - 1) / nblocks;

    GrB_Matrix_new(&C, TYPE_GrB, m, n);

    create_blocks(&bA, &bm, &bl, A, 0, blockSize, 0);
    create_blocks(&bB, &bl, &bn, B, blockSize, 0, 0);
    create_blocks(&bC, &bm, &bn, C, 0, 0, 0);

    timer_start();
    IJK_LOOP(bm, bn, bl, {
        OK(GrB_mxm(bC[i * bn + j], GrB_NULL, TYPE_PLUS, TYPE_PLUS_TIMES,
                   bA[i * bl + k], bB[k * bn + j], GrB_NULL));
    });
    mxm_t = timer_end();

    timer_start();
    sum = 0;
    for (size_t i = 0; i < bm; i++) {
        for (size_t j = 0; j < bn; j++) {
            double tmp;
            GrB_reduce(&tmp, GrB_NULL, TYPE_PLUS_MONOID, bC[i * bn + j], GrB_NULL);
            sum += tmp;
        }
    }
    reduce_t = timer_end();

    printf("%s,%zu,%zu,%lf,%lf,%lf\n", __FUNCTION__, nblocks, nblocks, mxm_t, reduce_t, sum);

    free_blocks(bA, bm, bl);
    free_blocks(bB, bl, bn);
    free_blocks(bC, bm, bn);
    GrB_free(&C);
}

int main(int argc, char *argv[]) {
    int numThreads;
    size_t niter;
    char *A_path;
    char *B_path;
    GrB_Matrix A;
    GrB_Matrix B;

    /* Init and set the number of threads */
    GrB_init(GrB_NONBLOCKING);
    if (arg_to_int32_def(argc, argv, "-nt|--numThreads", &numThreads, 1) != OTX_SUCCESS) { return 1; }
    GxB_Global_Option_set(GxB_GLOBAL_NTHREADS, numThreads);

    if (arg_to_uint64_def(argc, argv, "-ni|--numIter", &niter, 1) != OTX_SUCCESS) { return 1; }

    /* Read the matrices */
    timer_start();
    if (arg_to_str_def(argc, argv, "-A", &A_path, INPUTS_DIR"/simple") != OTX_SUCCESS) { return 1; }
    if (read_Matrix_FP64(&A, A_path, false) != GrB_SUCCESS) { return 1; }
    if (arg_to_str_def(argc, argv, "-B", &B_path, A_path) != OTX_SUCCESS) { return 1; }
    if (read_Matrix_FP64(&B, B_path, false) != GrB_SUCCESS) { return 1; }
    printf("Read time: %lf\n", timer_end());

    timer_start();
    RUN(niter, 1, test0(A, B))
//    RUN(niter, 8, test1(A, B, b))
//    RUN(niter, 8, test2(A, B, b))
    RUN(niter, 64, test3(A, B, b))
//    RUN(niter, 64, test4(A, B, b))
//    RUN(niter, 8, test5(A, B, b))
    printf("Iterations: %lf\n", timer_end());

    GrB_free(&A);
    GrB_free(&B);

    GrB_finalize();
}

GrB_Info create_blocks(GrB_Matrix **bA_out, GrB_Index *nrows_blocked_out, GrB_Index *ncols_blocked_out,
                       GrB_Matrix A, size_t nrows_block, size_t ncols_block, bool full_idxes) {
    /* Input matrix tuples */
    GrB_Index *I, *J;
    TYPE *X;

    /* Blocked tuples and matrix blocks */
    GrB_Index **bI, **bJ;
    TYPE **bX;
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
    bX = malloc(nblocks * sizeof(TYPE *));
    if (nzb_cnt == NULL || bI == NULL || bJ == NULL) { exit(1); }

    /* Get tuples form the input matrix */
    I = malloc((nvals + 1) * sizeof(GrB_Index));
    J = malloc((nvals + 1) * sizeof(GrB_Index));
    X = malloc((nvals + 1) * sizeof(TYPE));
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
        bX[i] = malloc(nzb_cnt[i] * sizeof(TYPE));
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
        OK(GrB_Matrix_build(bA[i], bI[i], bJ[i], bX[i], nzb_cnt[i], TYPE_PLUS));
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