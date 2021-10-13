#include <grb/grb.h>
#include <otx/otx.h>
#include "../util/reader_cxx.h"
#include "../util/timer.h"

void BFS(grb::Vector<int32_t> &v, grb::Matrix<bool> &A, uint64_t s, size_t blockSize) {
    uint64_t n = A.nrows();

    v = grb::Vector<int32_t>{n, "v", blockSize};

    auto q = grb::Vector<bool>{n, "q", blockSize};
    q.setElement(true, s);

    /* BFS traversal and label the vertices */
    int32_t d = 0;
    bool succ;
    do {
        ++d;
        grb::assign(v, q, nullptr, d, GrB_ALL, n, GrB_NULL);
        grb::vxm(q, v, nullptr, GxB_LOR_LAND_BOOL, q, A, GrB_DESC_RC);
        succ = grb::reduce<bool>(nullptr, GxB_LOR_BOOL_MONOID, q, GrB_NULL);
    } while (succ);
}

int main(int argc, char *argv[]) {
    grb::init();

    {
        /* Read/init the matrices */
        auto matrixPath = otx::argTo<std::string>(argc, argv, {"-g", "--graph"}, INPUTS_DIR"/simple");
        grb::Vector<int32_t> v;
        auto A = readMatrix<bool>(matrixPath.c_str());
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

        /* Execute BFS from random sources */
        auto nsver = otx::argTo<size_t>(argc, argv, {"-nv", "--nsver"}, 1);
        srand(0);
        auto n = A.nrows();

        auto start = get_time_ns();
        for (size_t i = 0; i < nsver; i++) {
            size_t source = rand() % n;
            BFS(v, A, source, blockSize);
            grb::wait();
//            v.print();
            auto sum = grb::reduce<int32_t>(GrB_NULL, GrB_PLUS_MONOID_INT32, v, GrB_NULL);
            printf("Source: %zu; sum: %d\n", source, sum);
        }
        auto end = get_time_ns();
        std::cout << "Execution time: " << (end - start) / 1e9 << std::endl;
    }

    grb::finalize();
}