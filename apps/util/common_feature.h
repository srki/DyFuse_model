#ifndef GRB_FUSION_COMMON_H
#define GRB_FUSION_COMMON_H

#include <iostream>
#include "../util/block_matrix_generic_feature.h"
#include "../util/timer.h"


void createMxMBlocks_feature(GrB_Matrix M, GrB_Matrix A, GrB_Matrix B, bool transposeA, bool transposeB,
                             size_t nblocksM, size_t nblocksN, size_t nblocksL, GrB_Type type,
                             GrB_Matrix **bC, GrB_Matrix **bM, GrB_Matrix **bA, GrB_Matrix **bB,
                             double* featureA, double* featureB, double* featureC, double* featureM) {
    GrB_Index nrows, ncols;
    GrB_Matrix_nrows(&nrows, A);
    GrB_Matrix_nrows(&ncols, B);

    GrB_Matrix C;
    GrB_Matrix_new(&C, type, nrows, ncols);

    if (!transposeA) {
        create_blocks_feature_g(bA, A, nblocksM, nblocksL, 0, 0, false, type, featureA);
    } else {
        create_blocks_feature_g(bA, A, nblocksL, nblocksM, 0, 0, false, type, featureA);
    }

    if (!transposeB) {
        create_blocks_feature_g(bB, B, nblocksL, nblocksN, 0, 0, false, type, featureB);
    } else {
        create_blocks_feature_g(bB, B, nblocksN, nblocksL, 0, 0, false, type, featureB);
    }

    if (bC != NULL) {
        create_blocks_feature_g(bC, C, nblocksM, nblocksN, 0, 0, false, type, featureC);
//    *bC = (GrB_Matrix *) calloc(nblocksM * nblocksN, sizeof(GrB_Matrix));
    }

    if (M != GrB_NULL) {
        create_blocks_feature_g(bM, M, nblocksM, nblocksN, 0, 0, false, type, featureM);
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





#endif //GRB_FUSION_COMMON_H
