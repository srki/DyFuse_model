/* LICENSE PLACEHOLDER */

#include <grb/grb.h>
#include <otx/otx.h>
#include <iomanip>
#include "../util/timer.h"
#include "../util/reader_cxx.h"

const char *get_file_name(const char *path) {
    size_t len = strlen(path);
    while (path[len] != '/') { len--; }
    return path + len + 1;
}


int main(int argc, char *argv[]) {
    grb::init();

    {
        using ElementType = int32_t;

        /* Read/init the matrices */
        auto start = get_time_ns();
        auto pathA = otx::argTo<std::string>(argc, argv, "-A", INPUTS_DIR"/simple");
        auto pathB = otx::argTo<std::string>(argc, argv, "-B", pathA);
        auto A = readMatrix<ElementType>(pathA, false);
        grb::Matrix<ElementType> L{A.nrows(), A.ncols(), "L"};
        grb::Matrix<ElementType> U{A.nrows(), A.ncols(), "U"};
        grb::Matrix<ElementType> M{A.nrows(), A.ncols(), "M"};
        grb::Matrix<ElementType> C{A.nrows(), A.ncols(), "C"};

        auto end = get_time_ns();
        std::cout << "Read time: " << (end - start) / 1e9 << std::endl;

        auto niter = otx::argTo<size_t>(argc, argv, "--niter", 1);

        /* Determine block size */
        auto numBlocksM = otx::argTo<size_t>(argc, argv, {"-nbm", "--numBlocksM"}, 0);
        auto numBlocksN = otx::argTo<size_t>(argc, argv, {"-nbn", "--numBlocksN"}, 0);
        auto numBlocksK = otx::argTo<size_t>(argc, argv, {"-nbk", "--numBlocksK"}, 0);

        size_t blockSizeM = (numBlocksM == 0) ? otx::argTo<size_t>(argc, argv, {"-bsm", "--blockSizeM"}, 0) :
                            (A.nrows() + numBlocksM - 1) / numBlocksM;
        size_t blockSizeN = (numBlocksN == 0) ? otx::argTo<size_t>(argc, argv, {"-bsn", "--blockSizeN"}, 0) :
                            (A.ncols() + numBlocksN - 1) / numBlocksN;
        size_t blockSizeK = (numBlocksK == 0) ? otx::argTo<size_t>(argc, argv, {"-bsk", "--blockSizeK"}, 0) :
                            (A.ncols() + numBlocksK - 1) / numBlocksK;

        grb::select(L, grb::null, GrB_NULL, GxB_TRIL, A, -1, GrB_NULL);
        grb::select(U, grb::null, GrB_NULL, GxB_TRIU, A, 1, GrB_NULL);
        M = L;

        /* Block The matrices */
//        A.block(blockSizeM, blockSizeK);
        L.block(blockSizeM, blockSizeK);
        U.block(blockSizeN, blockSizeK);
        M.block(blockSizeM, blockSizeN);
        C.block(blockSizeM, blockSizeN);

//        A.printDense();
//        std::endl(std::cout);
//        L.printDense();
//        std::endl(std::cout);
//        U.printDense();
//        std::endl(std::cout);
//        M.printDense();

        /* Execute mxm */
        for (int i = 0; i < niter; i++) {
            C.clear();
            auto start = get_time_ns();
            grb::mxm(C, M, GrB_NULL, GxB_PLUS_TIMES_INT32, L, U, GrB_DESC_T1);
//            C.wait();

//            std::cout << "Execution time: " << double(end - start) / 1e9 << std::endl;

            auto r = grb::reduce<ElementType>(GrB_NULL, GrB_PLUS_MONOID_INT32, C, GrB_NULL);
            auto end = get_time_ns();

            auto ncpu = std::getenv("STARPU_NCPU");

            std::cout << "LOG,grb," << get_file_name(pathA.c_str()) << ","
                      << numBlocksM << "," << numBlocksN << "," << numBlocksK << ","
                      << (ncpu != nullptr ? ncpu : "-1") << "," << double(end - start) / 1e9 << ","
                      << std::setprecision(15) << r << std::endl;
        }
    }

    grb::finalize();
}


