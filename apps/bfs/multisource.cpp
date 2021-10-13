#include <grb/grb.h>
#include <otx/otx.h>
#include <iomanip>
#include "../util/reader_cxx.h"
#include "../util/timer.h"
#include "../util/GrB_util.h"

double initTime = 0;
double outerTime = 0;
std::vector<double> addTime;
std::vector<double> mxmTime;

void bfs(grb::Matrix<int32_t> &A, uint64_t *s, uint64_t nsver, int32_t &depth) {
    initTime = 0;
    addTime.clear();
    mxmTime.clear();

    auto initStart = get_time_ns();
    uint64_t n = A.nrows();

    auto *i_nsver = new uint64_t[nsver];
    auto *ones = new int32_t[nsver];
    for (int i = 0; i < nsver; ++i) {
        i_nsver[i] = i;
        ones[i] = 1;
    }

    grb::Matrix<int32_t> numsp{n, nsver, "numsp"};
    numsp.build(s, i_nsver, ones, nsver, GrB_PLUS_INT32);
    delete[] i_nsver;
    delete[] ones;

    grb::Matrix<int32_t> frontier{n, nsver, "frontier"};
    grb::extract(frontier, numsp, GrB_NULL, A, GrB_ALL, n, s, nsver, GrB_DESC_RCT0);
    grb::wait();
    auto initEnd = get_time_ns();
    initTime = double(initEnd - initStart) / 1e9;

    int32_t d = 0;
    uint64_t nvals;

    do {
        auto addStart = get_time_ns();
        grb::eWiseAdd(numsp, grb::null, GrB_NULL, GrB_PLUS_INT32, numsp, frontier, GrB_NULL);
        numsp.wait();
        auto addEnd = get_time_ns();
        addTime.push_back(double(addEnd - addStart) / 1e9);
//        auto frontierGrb = frontier.getImpl().getData()->getBlock(0);
//        GxB_Matrix_fprint(frontierGrb, "frontier A", GxB_SUMMARY, stdout);

        auto mxmStart = get_time_ns();
        grb::mxm(frontier, numsp, GrB_NULL, GxB_PLUS_TIMES_INT32, A, frontier, GrB_DESC_RCT0);
        frontier.wait();
        auto mxmEnd = get_time_ns();
        mxmTime.push_back(double(mxmEnd - mxmStart) / 1e9);

        nvals = frontier.nvals();
        d++;
    } while (nvals);

    grb::wait();

    depth = d;
}

void bfs(grb::Matrix<int32_t> &A, uint64_t *s, uint64_t nsver, size_t blockSize,
         unsigned long long &blockingTime, int32_t &depth) {
    initTime = 0;
    addTime.clear();
    mxmTime.clear();

    auto initStart = get_time_ns();
    uint64_t n = A.nrows();
    A.block(A.nrows(), blockSize);

    auto *i_nsver = new uint64_t[nsver];
    auto *ones = new int32_t[nsver];
    for (int i = 0; i < nsver; ++i) {
        i_nsver[i] = i;
        ones[i] = 1;
    }

    grb::Matrix<int32_t> numsp{n, nsver, "numsp", blockSize, blockSize};
    numsp.build(s, i_nsver, ones, nsver, GrB_PLUS_INT32);
    delete[] i_nsver;
    delete[] ones;

    grb::Matrix<int32_t> frontier{n, nsver, "frontier", blockSize, blockSize};
    grb::extract(frontier, numsp, GrB_NULL, A, GrB_ALL, n, s, nsver, GrB_DESC_RCT0);
    auto initEnd = get_time_ns();
    initTime = double(initEnd - initStart) / 1e9;

    int32_t d = 0;
    uint64_t nvals;

    blockingTime = 0;

    do {
        auto addStart = get_time_ns();
        grb::eWiseAdd(numsp, grb::null, GrB_NULL, GrB_PLUS_INT32, numsp, frontier, GrB_NULL);
        numsp.wait();
        auto addEnd = get_time_ns();
        addTime.push_back(double(addEnd - addStart) / 1e9);

//        auto frontierGrb = frontier.getImpl().getData()->getBlock(0);
//        GxB_Matrix_fprint(frontierGrb, "frontier A", GxB_SUMMARY, stdout);
        auto start = get_time_ns();
        grb::Matrix<int32_t> F(frontier.nrows(), frontier.ncols(), "F", blockSize, blockSize);
        frontier.wait();
        grb::apply(F, grb::null, GrB_NULL, GrB_IDENTITY_INT32, frontier, GrB_DESC_R);
        F.block(F.nrows(), blockSize);
        F.wait();
        auto end = get_time_ns();
        blockingTime += end - start;
//        GxB_Matrix_fprint(frontierGrb, "frontier B", GxB_SUMMARY, stdout);

        auto mxmStart = get_time_ns();
        grb::mxm(frontier, numsp, GrB_NULL, GxB_PLUS_TIMES_INT32, A, frontier, GrB_DESC_RCT0);
        frontier.wait();
        auto mxmEnd = get_time_ns();
        mxmTime.push_back(double(mxmEnd - mxmStart) / 1e9);

        frontier.wait();
        nvals = frontier.nvals();
        d++;
    } while (nvals);
    grb::wait();

    depth = d;
}


int main(int argc, char *argv[]) {
    grb::init();

    {
        /* Read/init the matrices */
        auto matrixPath = otx::argTo<std::string>(argc, argv, {"-g", "--graph"}, INPUTS_DIR"/simple");
        auto A = readMatrix<int32_t>(matrixPath.c_str(), true);
        A.setName("A");

        auto witer = otx::argTo<size_t>(argc, argv, {"--witer"}, 1);
        auto niter = otx::argTo<size_t>(argc, argv, {"--niter"}, 1);
        auto oiter = otx::argTo<size_t>(argc, argv, {"--oiter"}, 1);

        /* Determine block size */
        auto numBlocks = otx::argTo<size_t>(argc, argv, {"-nb", "--numBlocks"}, 0);
        size_t blockSize;
        if (numBlocks == 0) {
            blockSize = otx::argTo<size_t>(argc, argv, {"-bs", "--blockSize"}, 0);
        } else {
            blockSize = (A.nrows() + numBlocks - 1) / numBlocks;
        }

        /* Create batch indices */
        auto nsver = otx::argTo<size_t>(argc, argv, {"-nv", "--nsver"}, 1);
        auto s = new uint64_t[nsver];
        std::generate_n(s, nsver, [next = size_t(0)]() mutable { return next++; });

        for (int i = 0; i < oiter; i++) {
            {
                double totalBFS = 0;
                for (int j = 0; j < witer + niter; j++) {
                    int32_t depth;

                    auto start = get_time_ns();
                    bfs(A, s, nsver, depth);
                    auto end = get_time_ns();

                    if (j < witer) { continue; }
                    totalBFS += double(end - start) / 1e9;

                    // @formatter:off
                    std::cout << std::setw(10) << "LOG-iter;"
                              << std::setw(25) << get_file_name(matrixPath.c_str()) << ";"
                              << std::setw(5) << numBlocks << ";"
                              << std::setw(5) << nsver << ";"
                              << std::setw(15) << std::setprecision(10) << std::fixed << double(end - start) / 1e9 << ";"
                              << std::setw(15) << std::setprecision(10) << std::fixed << initTime << ";"
                              << std::setw(15) << std::setprecision(10) << std::fixed << std::accumulate(addTime.begin(), addTime.end(), 0.0) << ";"
                              << std::setw(15) << std::setprecision(10) << std::fixed << std::accumulate(mxmTime.begin(), mxmTime.end(), 0.0) << ";"
                              << std::setw(15) << std::setprecision(10) << std::fixed << 0 << ";"
                              << std::setw(15) << depth
                              << std::endl;
                    // @formatter:on

                    std::cout << std::setw(10) << "LOG-add";
                    for (auto &it: addTime) {
                        std::cout << "; " << std::setw(15) << std::setprecision(10) << it << std::fixed;
                    }
                    std::cout << std::endl;

                    std::cout << std::setw(10) << "LOG-mxm";
                    for (auto &it: mxmTime) {
                        std::cout << "; " << std::setw(15) << std::setprecision(10) << it << std::fixed;
                    }
                    std::cout << std::endl;
                }


                std::cout << "LOG-mean;"
                          << get_file_name(matrixPath.c_str()) << ";"
                          << numBlocks << ";"
                          << nsver << ";"
                          << totalBFS / double(niter) << ";"
                          << 0
                          << std::endl;
            }

            {
                double totalBFS = 0;
                double totalBlocking = 0;

                for (int j = 0; j < witer + niter; j++) {
                    unsigned long long blockingTime;
                    int32_t depth;

                    auto start = get_time_ns();
                    bfs(A, s, nsver, blockSize, blockingTime, depth);
                    auto end = get_time_ns();

                    if (j < witer) { continue; }
                    totalBFS += double(end - start) / 1e9;
                    totalBlocking += double(blockingTime) / 1e9;

                    // @formatter:off
                    std::cout << std::setw(10) << "LOG-iter;"
                              << std::setw(25) << get_file_name(matrixPath.c_str()) << ";"
                              << std::setw(5) << numBlocks << ";"
                              << std::setw(5) << nsver << ";"
                              << std::setw(15) << std::setprecision(10) << std::fixed << double(end - start) / 1e9 << ";"
                              << std::setw(15) << std::setprecision(10) << std::fixed << initTime << ";"
                              << std::setw(15) << std::setprecision(10) << std::fixed << std::accumulate(addTime.begin(), addTime.end(), 0.0) << ";"
                              << std::setw(15) << std::setprecision(10) << std::fixed << std::accumulate(mxmTime.begin(), mxmTime.end(), 0.0) << ";"
                              << std::setw(15) << std::setprecision(10) << std::fixed << double(blockingTime) / 1e9 << ";"
                              << std::setw(15) << depth
                              << std::endl;
                    // @formatter:on

                    std::cout << std::setw(10) << "LOG-add";
                    for (auto &it: addTime) {
                        std::cout << "; " << std::setw(15) << std::setprecision(10) << it << std::fixed;
                    }
                    std::cout << std::endl;

                    std::cout << std::setw(10) << "LOG-mxm";
                    for (auto &it: mxmTime) {
                        std::cout << "; " << std::setw(15) << std::setprecision(10) << it << std::fixed;
                    }
                    std::cout << std::endl;
                }

                std::cout << "LOG-mean;"
                          << get_file_name(matrixPath.c_str()) << ";"
                          << numBlocks << ";"
                          << nsver << ";"
                          << totalBFS / double(niter) << ";"
                          << totalBlocking / double(niter)
                          << std::endl;
            }
        }

        delete[] s;
    }

    grb::finalize();
}

