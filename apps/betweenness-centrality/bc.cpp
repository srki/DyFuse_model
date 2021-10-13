/* https://people.eecs.berkeley.edu/~aydin/GABB17.pdf */

#include <grb/grb.h>
#include <otx/otx.h>
#include "../util/reader_cxx.h"
#include "../util/timer.h"

void BC_update(grb::Vector<float> &delta, grb::Matrix<int32_t> &A, uint64_t *s, uint64_t nsver, size_t blockSize) {
    uint64_t n = A.nrows();
    delta = grb::Vector<float>{n, "delta", blockSize};

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

    auto sigmas = new grb::Matrix<bool> *[n]{};
    int32_t d = 0;
    uint64_t nvals;

    do {
        sigmas[d] = new grb::Matrix<bool>{n, nsver, "sigmas[" + std::to_string(d) + "]", blockSize, blockSize};
        grb::apply(*sigmas[d], grb::null, GrB_NULL, GrB_IDENTITY_BOOL, frontier, GrB_NULL);
        grb::eWiseAdd(numsp, grb::null, GrB_NULL, GrB_PLUS_INT32, numsp, frontier, GrB_NULL);
        grb::mxm(frontier, numsp, GrB_NULL, GxB_PLUS_TIMES_INT32, A, frontier, GrB_DESC_RCT0);
        nvals = frontier.nvals();
        d++;
    } while (nvals);
    std::cout << "BFS: " << d << std::endl;

    grb::Matrix<float> nspinv{n, nsver, "nspinv", blockSize, blockSize};
    grb::apply(nspinv, grb::null, GrB_NULL, GrB_MINV_FP32, numsp, GrB_NULL);

    grb::Matrix<float> bcu{n, nsver, "bcu", blockSize, blockSize};
    grb::assign(bcu, grb::null, GrB_NULL, 1.0f, GrB_ALL, n, GrB_ALL, nsver, GrB_NULL);

    grb::Matrix<float> w{n, nsver, "w", blockSize, blockSize};
    for (int i = d - 1; i > 0; --i) {
        grb::eWiseMult(w, *sigmas[i], GrB_NULL, GrB_TIMES_FP32, bcu, nspinv, GrB_DESC_R);
        grb::mxm(w, *sigmas[i - 1], GrB_NULL, GxB_PLUS_TIMES_FP32, A, w, GrB_DESC_R);
        grb::eWiseMult(bcu, grb::null, GrB_PLUS_FP32, GrB_TIMES_FP32, w, numsp, GrB_NULL);
    }
    grb::assign(delta, grb::null, GrB_NULL, -(float) nsver, GrB_ALL, n, GrB_NULL);
    grb::reduce(delta, grb::null, GrB_PLUS_FP32, GrB_PLUS_FP32, bcu, GrB_NULL);

    for (int i = 0; i < d; i++) {
        delete sigmas[i];
    }
    delete[] sigmas;
}


int main(int argc, char *argv[]) {
    grb::init();

    {
        /* Read/init the matrices */
        auto matrixPath = otx::argTo<std::string>(argc, argv, {"-g", "--graph"}, INPUTS_DIR"/simple");
        grb::Vector<float> delta;
        auto A = readMatrix<int32_t>(matrixPath.c_str(), true);
        A.setName("A");

        /* Determine block size */
        auto numBlocks = otx::argTo<size_t>(argc, argv, {"-nb", "--numBlocks"}, 0);
        size_t blockSize;
        if (numBlocks == 0) {
            blockSize = otx::argTo<size_t>(argc, argv, {"-bs", "--blockSize"}, 0);
        } else {
            blockSize = (A.nrows() + numBlocks - 1) / numBlocks;
        }

        /* Block the input matrix */
        A.block(blockSize, blockSize);

        /* Create batch indices */
        auto nsver = otx::argTo<size_t>(argc, argv, {"-nv", "--nsver"}, 1);
        auto s = new uint64_t[nsver];
        std::generate_n(s, nsver, [next = size_t(0)]() mutable { return next++; });

        /* Execute betweenness centrality for the first batch */
        for (int i = 0; i < 100; i++) {
            auto start = get_time_ns();
            BC_update(delta, A, s, nsver, blockSize);
            auto end = get_time_ns();
            std::cout << "Execution time: " << (end - start) / 1e9 << std::endl;

//            grb::wait();
            auto sum = grb::reduce<float>(GrB_NULL, GrB_PLUS_MONOID_FP32, delta, GrB_NULL);
            printf("%f\n", sum);
        }
        delete[] s;

        /* Verification */
//        delta.print();
        auto sum = grb::reduce<float>(GrB_NULL, GrB_PLUS_MONOID_FP32, delta, GrB_NULL);
        printf("%f\n", sum);
    }

    grb::finalize();
}

