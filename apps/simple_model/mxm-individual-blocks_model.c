#include <otx/otx.h>
#include <mio/mio.h>
#include "../util/GrB_util.h"

#define TYPE uint32_t
#define TYPE_FORMAT PRIu32
#define TYPE_GrB GrB_UINT32
#define TYPE_ONE GxB_ONE_UINT32
#define TYPE_PLUS GrB_PLUS_UINT32
#define TYPE_PLUS_MONOID GrB_PLUS_MONOID_UINT32
#define TYPE_PLUS_TIMES GxB_PLUS_TIMES_UINT32

#include "../util/block_matrix.h"
#include "../util/timer.h"
#include "../util/get_feature.h"
#include "../util/reader_c.h"

#define ABORT(err_msg) \
 { char msg[256];\
   sprintf(msg,"%s at line %d in file %s\n",err_msg,__LINE__, __FILE__);\
   printf("%s", msg);  \
   exit(-1);}

void
test_blocks(const char *name, GrB_Matrix A, GrB_Matrix B, GrB_Matrix M, size_t nblocks_m, size_t nblocks_n,
            size_t nblocks_l,
            size_t niter, int shuffle, int nthreads,  double* featuresA,double* featuresB, double* featuresM,
            char* decomp) {
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
    //create_blocks(&bA, &bm, &bl, A, block_size_m, block_size_l, 0);
    //create_blocks(&bB, &bn, &bl, B, block_size_n, block_size_l, 0);
    //create_blocks(&bC, &bm, &bn, C, block_size_m, block_size_n, 0);
    //create_blocks(&bM, &bm, &bn, M, block_size_m, block_size_n, 0);

    create_blocks_features(&bA, &bm, &bl, A, block_size_m, block_size_l, 0,featuresA);
    create_blocks_features(&bB, &bn, &bl, B, block_size_n, block_size_l, 0,featuresB);
    create_blocks(&bC, &bm, &bn, C, block_size_m, block_size_n, 0);
    create_blocks_features(&bM, &bm, &bn, M, block_size_m, block_size_n, 0,featuresM);



    if (nblocks_m != bm || nblocks_n != bn || nblocks_l != bl) {
        exit(5);
    }
    int skip=2;
    for (int iter = 0; iter < niter+skip; iter++) {
        total_time = 0.0;
        mxm_time = 0.0;
        reduce_time = 0.0;
        sum = 0;

        for (size_t i = 0; i < bm; i++) {
            for (size_t j = 0; j < bn; j++) {
                for (size_t k = 0; k < bl; k++) {
                    int bidA=i * bl + k;
                    int bidB=j * bl + k;
                    int bidM=i * bn + j;
                    timer_start();
                    BURBLE(OK(GrB_mxm(bC[i * bn + j], bM[i * bn + j], GrB_NULL, TYPE_PLUS_TIMES,
                                      bA[i * bl + k], bB[j * bl + k], GrB_DESC_RT1)));
                    block_time = timer_end();
                    mxm_time += block_time;

                    timer_start();
                    GrB_reduce(&sum, TYPE_PLUS, TYPE_PLUS_MONOID, bC[i * bn + j], GrB_NULL);
                    reduce_time += timer_end();
                    if (iter>=skip) {
                        printf("Mask-block,%s,%s,%d,%zu,%zu,%zu,%zu,%zu,%zu,"
                               "%d,%lf,%lf,"
                               "A: %d,%.2f,%.2f,%.2f,"
                               "B: %d,%.2f,%.2f,%.2f,"
                               "M: %d,%.2f,%.2f,%.2f\n",
                               name, decomp, nthreads, nblocks_m, nblocks_n, nblocks_l, i, j, k,
                               shuffle, block_time, reduce_time,
                               bidA, featuresA[bidA * NUM_FEATURE], featuresA[bidA * NUM_FEATURE + 1],
                               featuresA[bidA * NUM_FEATURE + 2],
                               bidB, featuresB[bidB * NUM_FEATURE], featuresB[bidB * NUM_FEATURE + 1],
                               featuresB[bidB * NUM_FEATURE + 2],
                               bidM, featuresM[bidM * NUM_FEATURE], featuresM[bidM * NUM_FEATURE + 1],
                               featuresM[bidM * NUM_FEATURE + 2]);
                    }
                }
            }
        }
    }


    free_blocks(bA, bm, bl);
    free_blocks(bB, bl, bn);
    free_blocks(bC, bm, bn);
    free_blocks(bM, bm, bn);
    GrB_free(&C);
}

void test_blocked(char *name, GrB_Matrix A, GrB_Matrix B, size_t nblocks_m, size_t nblocks_n, size_t nblocks_l,
                  size_t niter, int shuffle, int nthreads, double *featuresA, double *featuresB,double *featuresC, char* decomp) {
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

    create_blocks_features(&bA, &bm, &bl, A, block_size_m, block_size_l, 0,featuresA);
    create_blocks_features(&bB, &bl, &bn, B, block_size_l, block_size_n, 0,featuresB);
    create_blocks_features(&bC, &bm, &bn, C, block_size_m, block_size_n, 0,featuresC);

    if (nblocks_m != bm || nblocks_n != bn || nblocks_l != bl) {
        exit(5);
    }
    int skip=2;
    for (int iter = 0; iter < niter+skip; iter++) {
        total_time = 0.0;
        mxm_time = 0.0;
        reduce_time = 0.0;
        sum = 0;
        int count=0;
        //timer_start();
        for (size_t i = 0; i < bm; i++) {
            for (size_t j = 0; j < bn; j++) {
                for (size_t k = 0; k < bl; k++) {
                    int bidA = i * bl + k;
                    int bidB = j * bl + k;
                    int bidC = i * bn + j;
                    timer_start();
                    BURBLE(OK(GrB_mxm(bC[i * bn + j], GrB_NULL, GrB_NULL, TYPE_PLUS_TIMES,
                                      bA[i * bl + k], bB[k * bn + j], GrB_DESC_R)));
                    mxm_time = timer_end();

                    timer_start();
                    GrB_reduce(&sum, TYPE_PLUS, TYPE_PLUS_MONOID, bC[i * bn + j], GrB_NULL);
                    reduce_time = timer_end();
                    if (iter>=skip) {
                        printf("NoMask-block,%s,%s,%d,%zu,%zu,%zu,%zu,%zu,%zu,"
                               "%d,%d,%lf,%lf,"
                               "A,%d,%.2f,%.2f,%.2f,"
                               "B,%d,%.2f,%.2f,%.2f,"
                               "C,%d,%.2f,%.2f,%.2f\n",
                               name, decomp, nthreads, nblocks_m, nblocks_n, nblocks_l, i, j, k,
                               shuffle, count, mxm_time, reduce_time,
                               bidA, featuresA[bidA * NUM_FEATURE], featuresA[bidA * NUM_FEATURE + 1],
                               featuresA[bidA * NUM_FEATURE + 2],
                               bidB, featuresB[bidB * NUM_FEATURE], featuresB[bidB * NUM_FEATURE + 1],
                               featuresB[bidB * NUM_FEATURE + 2],
                               bidC, featuresB[bidC * NUM_FEATURE], featuresC[bidC * NUM_FEATURE + 1],
                               featuresC[bidC * NUM_FEATURE + 2]);
                        count++;
                    }
                }
            }
        }
    }
    free_blocks(bA, bm, bl);
    free_blocks(bB, bl, bn);
    free_blocks(bC, bm, bn);
    GrB_free(&C);
}

int main(int argc, char *argv[]) {
    double total_time, mxm_time, reduce_time;
    int maxThreads;
    size_t niter;
    char *A_path;
    int32_t shuffle;
    GrB_Matrix L;
    GrB_Matrix U;
    GrB_Matrix M;
    GrB_Matrix C;
    GrB_Index nrows;
    GrB_Index ncols;
    uint32_t sum;
    size_t nblocks;
    size_t nthreads;

    /* Init and set the number of threads */
    GrB_init(GrB_BLOCKING);
    if (arg_to_int32_def(argc, argv, "-mt|--maxThreads", &maxThreads, 1) != OTX_SUCCESS) { return 1; }

    if (arg_to_uint64_def(argc, argv, "--niter", &niter, 1) != OTX_SUCCESS) { return 2; }
    if (arg_to_int32_def(argc, argv, "--shuffle", &shuffle, 0) != OTX_SUCCESS) { return 3; }
    if (arg_to_str_def(argc, argv, "-A", &A_path, INPUTS_DIR"/simple") != OTX_SUCCESS) { return 4; }
    readLU(&L, &U, A_path, shuffle, TYPE_GrB);


    GrB_Matrix_dup(&M, L);
    GrB_Matrix_nrows(&nrows, L);
    GrB_Matrix_ncols(&ncols, U);

    /* vary the number of threads */

    //int threads_group=7;
    //int threads[7]={1, 16 ,32 ,48, 64 ,96 ,128};
    int threads_group=4;
    int threads[4]={1,42,84,168};

    for (int i=0;i<threads_group;i++) {
        nthreads = threads[i];
        //printf("thread num %d\n",nthreads);
        //fflush(stdout);
        GxB_Global_Option_set(GxB_GLOBAL_NTHREADS, nthreads);

        nblocks = 1;
        while (nblocks <= 2) {
            double *featuresA, *featuresB,*featuresM;
            if ( !(featuresA = (double*)malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double))) )
                ABORT("Malloc fails for featuresA[].");
            if ( !(featuresB = (double*)malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double))) )
                ABORT("Malloc fails for featuresA[].");
            if ( !(featuresM = (double*)malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double))) )
                ABORT("Malloc fails for featuresA[].");
            //printf("nblocks: %ld\n", nblocks);
            //fflush(stdout);

            test_blocks(get_file_name(A_path), L, U, M, nblocks, nblocks, 1, niter, shuffle, nthreads, featuresA,featuresB,featuresM,"1D");
            //if (nblocks > 1)
            //    test_blocks(get_file_name(A_path), L, U, M, nblocks, nblocks, nblocks, niter, shuffle, nthreads,
            //                featuresA, featuresB, featuresC, featuresM);
            free(featuresA);
            free(featuresB);
            free(featuresM);
            nblocks = nblocks * 2;
        }


        nblocks = 1;
        while (nblocks <= 4) {
            double *featuresA, *featuresB,*featuresM;
            featuresA = (double*) malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double));
            featuresB = (double*) malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double));
            featuresM = (double*) malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double));
            if (nblocks>1)
                test_blocks(get_file_name(A_path), L, U, M, nblocks, nblocks, 1, niter, shuffle, nthreads, featuresA,featuresB,featuresM,"1D");
            nblocks = nblocks+1;
        }
    }
    GrB_free(&L);
    GrB_free(&U);
    GrB_free(&M);
    GrB_free(&C);
#if 1
    GrB_Index nzA;
    GrB_Matrix A;
    GrB_Matrix B;
    if (read_Matrix_FP64(&A, A_path, shuffle) != GrB_SUCCESS) { return 1; }
    cast_matrix(&A, GrB_UINT64, GxB_ONE_UINT64);
    OK(GrB_Matrix_dup(&B, A));

    GrB_Matrix_nvals(&nzA, A);
    GrB_Matrix_nrows(&nrows, A);
    GrB_Matrix_ncols(&ncols, A);

    for (int i=0;i<threads_group;i++) {
        nthreads = threads[i];
        GxB_Global_Option_set(GxB_GLOBAL_NTHREADS, nthreads);
        nblocks = 1;
        while (nblocks <= 8) {
            double *featuresA, *featuresB,*featuresC;
            featuresA =(double*) malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double));
            featuresB =(double*) malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double));
            featuresC =(double*) malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double));

            test_blocked(get_file_name(A_path), A, B, nblocks, nblocks, 1, niter, shuffle,nthreads,featuresA,featuresB,featuresC,"1D");
            //if (nblocks>1) test_blocked(get_file_name(A_path), A, B, nblocks, nblocks, nblocks, niter, shuffle,nthreads,featuresA, featuresB,featuresC);
            nblocks = nblocks *2;
        }

        nblocks = 1;
        while (nblocks <= 4) {
            double *featuresA, *featuresB,*featuresC;
            featuresA =(double*) malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double));
            featuresB =(double*) malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double));
            featuresC =(double*) malloc(nblocks * nblocks * NUM_FEATURE * sizeof(double));

            if (nblocks>1) test_blocked(get_file_name(A_path), A, B, nblocks, nblocks, nblocks, niter, shuffle,nthreads,featuresA, featuresB,featuresC,"2D");
            nblocks = nblocks+1;
        }
    }
#endif


    return 0;
}