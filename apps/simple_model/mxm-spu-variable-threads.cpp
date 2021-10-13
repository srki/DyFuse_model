#include <starpu.h>
#include <iostream>
#include <omp.h>
#include <random>
#include <otx/otx.h>

extern "C" {
#include <GraphBLAS.h>
}

#define TYPE uint32_t
#define TYPE_FORMAT PRIu32
#define TYPE_GrB GrB_UINT32
#define TYPE_ONE GxB_ONE_UINT32
#define TYPE_PLUS GrB_PLUS_UINT32
#define TYPE_PLUS_MONOID GrB_PLUS_MONOID_UINT32
#define TYPE_PLUS_TIMES GxB_PLUS_TIMES_UINT32

#include "../util/GrB_util.h"
#include "../util/block_matrix.h"
#include "../util/timer.h"


struct Arg {
    GrB_Matrix A;
    GrB_Matrix B;
    GrB_Matrix C;
    GrB_Matrix M;
    int numThreads;

    Arg() = default;

    Arg(GrB_Matrix a, GrB_Matrix b, GrB_Matrix c, GrB_Matrix m, int numThreads)
            : A(a), B(b), C(c), M(m), numThreads(numThreads) {}
};

void
test_blocks(const char *name, GrB_Matrix A, GrB_Matrix B, GrB_Matrix M, size_t nblocks_m, size_t nblocks_n,
            size_t nblocks_l,
            size_t niter, size_t *nthreadsPerBLock, GrB_Type type) {
    GrB_Index m, n, l;
    GrB_Matrix_nrows(&m, A);
    GrB_Matrix_nrows(&n, B);
    GrB_Matrix_ncols(&l, A);

    size_t block_size_m, block_size_n, block_size_l;
    block_size_m = (m + nblocks_m - 1) / nblocks_m;
    block_size_n = (n + nblocks_n - 1) / nblocks_n;
    block_size_l = (l + nblocks_l - 1) / nblocks_l;

    GrB_Matrix C;
    GrB_Matrix_new(&C, type, m, n);

    GrB_Matrix *bA, *bB, *bC, *bM;
    GrB_Index bm, bn, bl;
    create_blocks(&bA, &bm, &bl, A, block_size_m, block_size_l, false);
    create_blocks(&bB, &bn, &bl, B, block_size_n, block_size_l, false);
    create_blocks(&bC, &bm, &bn, C, block_size_m, block_size_n, false);
    create_blocks(&bM, &bm, &bn, M, block_size_m, block_size_n, false);

    if (nblocks_m != bm || nblocks_n != bn || nblocks_l != bl) {
        std::cerr << "Error while creating blocks." << std::endl;
        exit(1);
    }

    starpu_codelet cl{};
    starpu_codelet_init(&cl);
    cl.cpu_funcs[0] = [](void *buffers[], void *clArgs) {
        auto arg = (Arg *) clArgs;

        GrB_Descriptor desc;
        GrB_Descriptor_new(&desc);
        GrB_Descriptor_set(desc, GrB_INP1, GrB_TRAN);
        GrB_Descriptor_set(desc, GrB_OUTP, GrB_REPLACE);
        GrB_Descriptor_set(desc, GxB_DESCRIPTOR_NTHREADS, static_cast<GrB_Desc_Value>(arg->numThreads));
//        std::cout << arg->numThreads << std::endl;

        GrB_mxm(arg->C, arg->M, GrB_NULL, GrB_PLUS_TIMES_SEMIRING_UINT32, arg->A, arg->B, desc);

        uint32_t sum = 0;
        GrB_Matrix_reduce_UINT32(&sum, GrB_NULL, GrB_PLUS_MONOID_UINT32, arg->C, GrB_NULL);
        std::cout << sum << std::endl;
    };
    cl.nbuffers = 0;

    auto args = new Arg[bm * bn * bl];
    for (int iter = 0; iter < niter; iter++) {

        timer_start();
        for (size_t i = 0; i < bm; i++) {
            for (size_t j = 0; j < bn; j++) {
                for (size_t k = 0; k < bl; k++) {
                    auto task = starpu_task_create();
                    task->cl = &cl;
                    task->cl_arg = &(args[i * bn * bl + j * bl + k]
                                             = Arg(bA[i * bl + k], bB[j * bl + k], bC[i * bn + j], bM[i * bn + j],
                                                   nthreadsPerBLock[i * bn * bl + j * bl + k]));
                    task->cl_arg_size = sizeof(Arg);
                    task->cl_arg_free = 0;
                    STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starput_task_submit");
                }
            }
        }

        starpu_task_wait_for_all();
        auto time = timer_end();
        std::cout << time << std::endl;
    }

    delete[] args;

    free_blocks(bA, bm, bl);
    free_blocks(bB, bl, bn);
    free_blocks(bC, bm, bn);
    free_blocks(bM, bm, bn);
    GrB_Matrix_free(&C);
}

int main(int argc, char *argv[]) {
    STARPU_CHECK_RETURN_VALUE(starpu_init(NULL), "starpu_init");
    OK(GrB_init(GrB_NONBLOCKING))
//    GxB_Global_Option_set(GxB_BURBLE, true);

    auto path = otx::argTo<std::string>(argc, argv, "-A", INPUTS_DIR"/simple");
    auto niter = otx::argTo<size_t>(argc, argv, {"--niter"}, 3);
    auto opt = otx::argTo<int>(argc, argv, "-o", 0);

    GrB_Matrix L, U;
    readLU(&L, &U, path.c_str(), false, GrB_UINT32);

    auto nthreadsPerBlock = new size_t[6 * 6];
    switch (opt) {
        case 0:
            for (auto i = 0; i < 6 * 6; i++) { nthreadsPerBlock[i] = 1; }
            break;
        case 1:
            for (auto i = 0; i < 6 * 6; i++) { nthreadsPerBlock[i] = 12; }
            break;
        case 2:
            for (auto i = 0; i < 6 * 6; i++) { nthreadsPerBlock[i] = 1; }
            nthreadsPerBlock[0] = 7;
            break;
        default:
            std::cerr << "Option not specified" << std::endl;
    }

    std::cout << "Start " << opt << std::endl;
    test_blocks(get_file_name(path.c_str()), L, U, L, 6, 6, 1, niter, nthreadsPerBlock, GrB_UINT32);

    GrB_Matrix_free(&L);
    GrB_Matrix_free(&U);

    OK(GrB_finalize());
    starpu_shutdown();
}