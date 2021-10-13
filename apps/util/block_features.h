//
// Created by sm108 on 8/29/21.
//

#ifndef GRB_FUSION_BLOCK_FEATURES_H
#define GRB_FUSION_BLOCK_FEATURES_H

#include <omp.h>
#include <numeric>
#include <cassert>
#include "../util/GrB_util.h"

void calculateBlockCnts(GrB_Matrix A, GrB_Index *cnts, size_t nblocksI, size_t nblocksJ, int numThreads) {
    GrB_Index *rowPtr, *colIndices, Ap_size, Aj_size, Ax_size, nrows, ncols, nvals;
    void *values;
    bool iso, jumbled;

    OK(GxB_Matrix_unpack_CSR(A, &rowPtr, &colIndices, &values, &Ap_size, &Aj_size, &Ax_size, &iso, &jumbled, GrB_NULL));

    GrB_Matrix_nrows(&nrows, A);
    GrB_Matrix_ncols(&ncols, A);
    GrB_Matrix_nvals(&nvals, A);

    size_t blockSizeI = (nrows + nblocksI - 1) / nblocksI;
    size_t blockSizeJ = (ncols + nblocksJ - 1) / nblocksJ;

    if (nrows * 2 < numThreads) { numThreads = 1; }

    auto threadCnts = new GrB_Index *[numThreads];
    auto rowCnts = new GrB_Index[nrows];
    auto threadRowCnts = new GrB_Index[numThreads]{};

#pragma omp parallel num_threads(numThreads) default(none) shared(threadCnts, nblocksI, nblocksJ, nrows, numThreads, rowPtr, blockSizeI, blockSizeJ, colIndices, rowCnts, threadRowCnts, std::cout)
    {
        int thisThread = omp_get_thread_num();

        // new distribution
        auto rowsPerThread = rowPtr[nrows] / numThreads;
        // @formatter:off
        GrB_Index begin = thisThread != 0              ? (std::lower_bound(rowPtr + 1, rowPtr + nrows, rowsPerThread * thisThread))       - rowPtr - 1 : 0;
        GrB_Index end   = thisThread != numThreads - 1 ? (std::lower_bound(rowPtr + 1, rowPtr + nrows, rowsPerThread * (thisThread + 1))) - rowPtr - 1 : nrows;
        // @formatter:on

        auto localCnts = new GrB_Index[nblocksI * nblocksJ]{};
        threadCnts[omp_get_thread_num()] = localCnts;

        for (GrB_Index i = begin; i < end; ++i) {
            auto rowIdx = (i / blockSizeI) * nblocksJ;
            for (GrB_Index j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
                auto colIdx = colIndices[j] / blockSizeJ;
                localCnts[rowIdx + colIdx]++;
            }
        }
    }

    for (size_t i = 0; i < nblocksI; ++i) {
        for (size_t j = 0; j < nblocksJ; ++j) {
            auto idx = i * nblocksJ + j;
            GrB_Index cnt = 0;
            for (int k = 0; k < numThreads; ++k) {
                cnt += threadCnts[k][idx];
            }
            cnts[idx] = cnt;
        }
    }

    for (int i = 0; i < numThreads; ++i) { delete[] threadCnts[i]; }
    delete[] threadCnts;
    delete[] rowCnts;
    delete[] threadRowCnts;

    OK(GxB_Matrix_pack_CSR(A, &rowPtr, &colIndices, &values, Ap_size, Aj_size, Ax_size, iso, jumbled, GrB_NULL));
}

void resizeBlockCnts(GrB_Index *cntsOut, size_t nblocksIOut, size_t nblocksJOut,
                     const GrB_Index *cntsIn, size_t nblocksIIn, size_t nblocksJIn) {
    assert(nblocksIIn % nblocksIOut == 0);
    assert(nblocksJIn % nblocksJOut == 0);

    size_t scaleI = nblocksIIn / nblocksIOut;
    size_t scaleJ = nblocksJIn / nblocksJOut;

    std::fill(cntsOut, cntsOut + nblocksIOut * nblocksJOut, 0);

    for (size_t i = 0; i < nblocksIOut; ++i) {
        for (size_t j = 0; j < nblocksJOut; ++j) {
            size_t cnt = 0;
            for (size_t ii = i * scaleI; ii < (i + 1) * scaleI; ++ii) {
                for (size_t jj = j * scaleJ; jj < (j + 1) * scaleJ; ++jj) {
                    cnt += cntsIn[ii * nblocksIIn + jj];
                }
            }
            cntsOut[i * nblocksJOut + j] = cnt;
        }
    }
}

#endif //GRB_FUSION_BLOCK_FEATURES_H
