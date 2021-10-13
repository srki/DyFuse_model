#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>

#include "../util/GrB_util.h"
#include "../util/reader_c.h"
#include "../util/block_features.h"


void printCnts(const GrB_Index *const cnts, size_t nblocksI, size_t nblocksJ) {
    auto width = int(std::log(*std::max_element(cnts, cnts + nblocksI * nblocksJ) - 1) / std::log(10) + 1);
    for (size_t ii = 0; ii < nblocksI; ++ii) {
        for (size_t jj = 0; jj < nblocksJ; ++jj) {
            std::cout << std::setw(width + 1) << cnts[ii * nblocksJ + jj];
        }
        std::cout << std::endl;
    }
}

int main(int argc, char *argv[]) {
    GrB_init(GrB_NONBLOCKING);

    GrB_Matrix A;
    {
        auto start = std::chrono::high_resolution_clock::now();
        OK(read_Matrix_INT32(&A, argv[1], false));
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Read " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
    }

    GrB_Index nrows, ncols, nvals;

    GrB_Matrix_nrows(&nrows, A);
    GrB_Matrix_ncols(&ncols, A);
    GrB_Matrix_nvals(&nvals, A);

    std::cout << nvals << std::endl;

    const int niter = 1;
    const int nblocks = 2;

    long t1 = 0, t2 = 0;
    for (int i = 0; i < niter; ++i) {
        auto *cnts = new GrB_Index[nblocks * nblocks];

        {
            auto start = std::chrono::high_resolution_clock::now();
            calculateBlockCnts(A, cnts, nblocks, nblocks, 12);
            auto end = std::chrono::high_resolution_clock::now();
            auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            t1 += time;
            if (niter == 1) { printCnts(cnts, nblocks, nblocks); }
        }

        {
            auto start = std::chrono::high_resolution_clock::now();
            for (int b = nblocks; b > 0; b /= 2) {
                auto c = new GrB_Index[b * b];
                resizeBlockCnts(c, b, b, cnts, nblocks, nblocks);
                if (niter == 1) {
                    std::cout << std::endl;
                    printCnts(c, b, b);
                }
                delete[] c;
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            t2 += time;

        }

        delete[] cnts;
    }

    std::cout << "nnzs " << t1 / niter << " " << t2 / niter << std::endl;


    GrB_Matrix_free(&A);
    GrB_finalize();

    return 0;
}