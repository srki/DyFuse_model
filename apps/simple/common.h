#ifndef GRB_FUSION_COMMON_H
#define GRB_FUSION_COMMON_H

#include <iostream>
#include "../util/block_matrix_generic.h"
#include "../util/timer.h"

void createMxMBlocks(GrB_Matrix M, GrB_Matrix A, GrB_Matrix B, bool transposeA, bool transposeB,
                     size_t nblocksM, size_t nblocksN, size_t nblocksL, GrB_Type type,
                     GrB_Matrix **bC, GrB_Matrix **bM, GrB_Matrix **bA, GrB_Matrix **bB) {
    GrB_Index nrows, ncols;
    GrB_Matrix_nrows(&nrows, A);
    GrB_Matrix_nrows(&ncols, B);

    GrB_Matrix C;
    GrB_Matrix_new(&C, type, nrows, ncols);

    if (!transposeA) {
        create_blocks_g(bA, A, nblocksM, nblocksL, 0, 0, false, type);
    } else {
        create_blocks_g(bA, A, nblocksL, nblocksM, 0, 0, false, type);
    }

    if (!transposeB) {
        create_blocks_g(bB, B, nblocksL, nblocksN, 0, 0, false, type);
    } else {
        create_blocks_g(bB, B, nblocksN, nblocksL, 0, 0, false, type);
    }

    if (bC != NULL) {
        create_blocks_g(bC, C, nblocksM, nblocksN, 0, 0, false, type);
//    *bC = (GrB_Matrix *) calloc(nblocksM * nblocksN, sizeof(GrB_Matrix));
    }

    if (M != GrB_NULL) {
        create_blocks_g(bM, M, nblocksM, nblocksN, 0, 0, false, type);
    }

    GrB_Matrix_free(&C);
}

void freeMxMBlocks(GrB_Matrix *bC, GrB_Matrix *bM, GrB_Matrix *bA, GrB_Matrix *bB,
                   size_t nblocksM, size_t nblocksN, size_t nblocksL) {
    if (bC != NULL) { free_blocks_g(bC, nblocksM, nblocksN); }
    if (bM != NULL) { free_blocks_g(bM, nblocksM, nblocksN); }
    free_blocks_g(bA, nblocksM, nblocksL);
    free_blocks_g(bB, nblocksL, nblocksN);
}

void baselineTricnt(const char *pathA, GrB_Matrix M, GrB_Matrix L, GrB_Matrix U, size_t witer, size_t niter) {
    GrB_Index nrows, ncols;
    GrB_Matrix_nrows(&nrows, L);
    GrB_Matrix_nrows(&ncols, U);
    GrB_Matrix C;

    double mxmTimeTotal = 0;
    double reduceTimeTotal = 0;
    for (size_t i = 0; i < witer + niter; i++) {

        timer_start();
        GrB_Matrix_new(&C, GrB_UINT64, nrows, ncols);
        OK(GrB_mxm(C, M, GrB_NULL, GrB_PLUS_TIMES_SEMIRING_UINT64, L, U, GrB_DESC_T1));
        double mxmTime = timer_end();

        timer_start();
        uint64_t sum = 0;
        OK(GrB_Matrix_reduce_UINT64(&sum, GrB_NULL, GrB_PLUS_MONOID_UINT64, C, GrB_NULL));
        double reduceTime = timer_end();

        OK(GrB_Matrix_free(&C));

        if (i < witer) { continue; }

        mxmTimeTotal += mxmTime;
        reduceTimeTotal += reduceTime;
        double total_time = mxmTime + reduceTime;

//        std::cout << "LOG-baseline-iter,grb," << get_file_name(pathA) << "," << total_time << ","
//                  << mxmTime << "," << reduceTime << "," << sum << std::endl;
    }

    std::cout << "LOG-baseline,grb," << get_file_name(pathA) << ","
              << mxmTimeTotal / niter << "," << reduceTimeTotal / niter << std::endl;

    OK(GrB_Matrix_free(&C));
}

void baselineBlockedTricnt(const char *pathA, GrB_Matrix M, GrB_Matrix L, GrB_Matrix U, size_t witer, size_t niter,
                           size_t nblocksM, size_t nblocksN, size_t nblocksL) {
    if (nblocksL != 1) { exit(1); }
    GrB_Matrix *bC, *bM, *bL, *bU;
    createMxMBlocks(M, L, U, false, true, nblocksM, nblocksN, nblocksL, GrB_UINT64, &bC, &bM, &bL, &bU);

    double mxmTimeTotal = 0;
    double reduceTimeTotal = 0;
    for (size_t iter = 0; iter < witer + niter; iter++) {
        timer_start();
        for (size_t i = 0; i < nblocksM; i++) {
            for (size_t j = 0; j < nblocksN; j++) {
                for (size_t k = 0; k < nblocksL; k++) {
                    size_t idxC = i * nblocksN + j;
                    size_t idxL = i * nblocksL + k;
                    size_t idxU = j * nblocksL + k;

                    GrB_Index nrows, ncols;
                    GrB_Matrix_nrows(&nrows, bL[idxL]);
                    GrB_Matrix_nrows(&ncols, bU[idxU]);

                    GrB_Matrix_free(&bC[idxC]);
                    GrB_Matrix_new(&bC[idxC], GrB_UINT64, nrows, ncols);

                    OK(GrB_mxm(bC[idxC], bM[idxC], GrB_NULL, GrB_PLUS_TIMES_SEMIRING_UINT64,
                               bL[idxL], bU[idxU], GrB_DESC_T1));
                }
            }
        }
        double mxmTime = timer_end();

        uint64_t sum = 0;
        timer_start();
        for (size_t i = 0; i < nblocksM; i++) {
            for (size_t j = 0; j < nblocksN; j++) {
                for (size_t k = 0; k < nblocksL; k++) {
                    size_t idxC = i * nblocksN + j;
                    OK(GrB_Matrix_reduce_UINT64(&sum, GrB_PLUS_UINT64, GrB_PLUS_MONOID_UINT64, bC[idxC], GrB_NULL));
                    GrB_Matrix_free(&bC[idxC]);
                }
            }
        }
        double reduceTime = timer_end();

        if (iter < witer) { continue; }

        mxmTimeTotal += mxmTime;
        reduceTimeTotal += reduceTime;

//        std::cout << "LOG-baseline-iter,grb-blocked," << get_file_name(pathA) << ","
//                  << nblocksM << "," << nblocksN << "," << nblocksL << ","
//                  << mxmTime << "," << reduceTime << "," << sum << std::endl;
    }

    std::cout << "LOG-baseline,grb-blocked," << get_file_name(pathA) << ","
              << nblocksM << "," << nblocksN << "," << nblocksL << ","
              << mxmTimeTotal / niter << "," << reduceTimeTotal / niter << std::endl;

    freeMxMBlocks(bC, bM, bL, bU, nblocksM, nblocksN, nblocksL);
}

#endif //GRB_FUSION_COMMON_H
