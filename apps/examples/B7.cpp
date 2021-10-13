
#include <grb/grb.h>
#include "../util/reader_cxx.h"

uint64_t triangleCount(grb::Matrix<uint64_t> &L) {
    auto n = L.nrows();

    auto C = grb::Matrix<uint64_t>{n, n};
    grb::mxm(C, L, GrB_NULL, GxB_PLUS_TIMES_UINT64, L, L, GrB_DESC_T1);

    auto count = grb::reduce<uint64_t>(GrB_NULL, GxB_PLUS_UINT64_MONOID, C, GrB_NULL);;

    return count;
}

int main(int argc, char *argv[]) {
    grb::init();

    {
        auto A = readMatrix<uint64_t>(argc > 1 ? argv[1] : INPUTS_DIR"/simple");
        A.setName("A");

        /* TODO: add select */
        auto count = triangleCount(A);
        std::cout << "Number of triangles: " << count << std::endl;
    }

    grb::finalize();
}