#include <grb/grb.h>
#include <otx/otx.h>
#include <iomanip>
#include "../../util/reader_cxx.h"
#include "../../util/timer.h"

const char *get_file_name(const char *path) {
    size_t len = strlen(path);
    while (path[len] != '/') { len--; }
    return path + len + 1;
}

int main(int argc, char *argv[]) {
    grb::init();

    {
        using ElementType = double;

        /* Read/init the matrices */
        auto start = get_time_ns();
        auto pathA = otx::argTo<std::string>(argc, argv, "-A", INPUTS_DIR"/simple");
        auto pathB = otx::argTo<std::string>(argc, argv, "-B", pathA);
        auto A = readMatrix<ElementType>(pathA, false);
        auto B = readMatrix<ElementType>(pathA, false);
        auto C = grb::Matrix<ElementType>{A.nrows(), B.ncols()};
        A.setName("A");
        B.setName("B");
        C.setName("C");
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
                            (B.ncols() + numBlocksN - 1) / numBlocksN;
        size_t blockSizeK = (numBlocksK == 0) ? otx::argTo<size_t>(argc, argv, {"-bsk", "--blockSizeK"}, 0) :
                            (A.ncols() + numBlocksK - 1) / numBlocksK;

        /* Block The matrices */
        A.block(blockSizeM, blockSizeK);
        B.block(blockSizeK, blockSizeN);
        C.block(blockSizeM, blockSizeN);

        /* Execute mxm */
        for (int i = 0; i < niter; i++) {
            C.clear();
            auto start = get_time_ns();
            grb::mxm(C, grb::null, GrB_NULL, GxB_PLUS_TIMES_FP64, A, B, GrB_NULL);
            C.wait();
            auto end = get_time_ns();
            std::cout << "Execution time: " << (end - start) / 1e9 << std::endl;

            auto r = grb::reduce<ElementType>(GrB_NULL, GrB_PLUS_MONOID_FP64, C, GrB_NULL);
            std::cout << std::endl;

            auto ncpu = std::getenv("STARPU_NCPU");

            std::cout << "LOG," << get_file_name(pathA.c_str()) << ","
                      << numBlocksM << "," << numBlocksN << "," << numBlocksK << ","
                      << (ncpu != nullptr ? ncpu : "-1") << "," << (end - start) / 1e9 << ","
                      << std::setprecision(15) << r << std::endl;
        }
    }

    grb::finalize();
}