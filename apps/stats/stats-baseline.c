//
// Created by sm108 on 1/7/21.
//

#include <otx/atx.h>
#include <GraphBLAS.h>

#define TYPE_GrB GrB_FP64

#include "../util/GrB_util.h"
#include "../util/block_matrix.h"
#include "../util/reader_c.h"
#include "../util/timer.h"
#include "../util/exec_info.h"


void baseline(const char *name, GrB_Matrix M, GrB_Matrix A, GrB_Matrix B,
              unsigned transpose_A, unsigned transpose_B, size_t niter, size_t ncores,
              GrB_Type type, GrB_Semiring semiring, GrB_Monoid monoid) {
    double time, time_total = 0.0;

    GrB_Descriptor desc;
    GrB_Index nrows;
    GrB_Index ncols;
    GrB_Matrix C;

    OK(GrB_Descriptor_new(&desc));
    OK(GxB_Desc_set(desc, GrB_OUTP, GrB_REPLACE));
    if (transpose_A) { GxB_Desc_set(desc, GrB_INP0, GrB_TRAN); }
    if (transpose_B) { GxB_Desc_set(desc, GrB_INP1, GrB_TRAN); }
    OK(GxB_Desc_set(desc, GxB_DESCRIPTOR_NTHREADS, ncores));

    #if defined(HAVE_EXEC_INFO_SUPPORT)
    struct exec_info *info = calloc(1, sizeof(struct exec_info)); // TODO: free
    OK(GxB_Desc_set(desc, GxB_EXEC_INFO, info));
    #endif

    OK(GrB_Matrix_nrows(&nrows, A));
    OK(GrB_Matrix_ncols(&ncols, B));
    OK(GrB_Matrix_new(&C, type, nrows, ncols));

    for (size_t iter = 0; iter < niter; iter++) {
        GrB_Matrix_clear(C);

        #if defined(HAVE_EXEC_INFO_SUPPORT)
        memset(info, 0, sizeof(struct exec_info));
        #endif

        timer_start();
        OK(GrB_mxm(C, M != NULL ? M : GrB_NULL, GrB_NULL, semiring, A, B, desc));
        time = timer_end();

        if (type == GrB_FP64) {
            double sum = 0.0;
            OK(GrB_Matrix_reduce_FP64(&sum, GrB_NULL, GrB_PLUS_MONOID_FP64, C, GrB_NULL));
            printf("LOG;baseline-iter;%s;%lf;%lf\n", name, time, sum);
        } else if (type == GrB_UINT64) {
            uint64_t sum = 0;
            OK(GrB_Matrix_reduce_UINT64(&sum, GrB_NULL, GrB_PLUS_MONOID_UINT64, C, GrB_NULL));
            printf("LOG;baseline-iter;%s;%lf;%lu\n", name, time, sum);
        }

        time_total += time;
    }

    printf("LOG;baseline-mean;%s;%lf\n", name, time_total / niter);
    OK(GrB_Matrix_free(&C));
}
int main(int argc, char *argv[]) {
    char *A_path;
    uint64_t niter;
    int num_threads;
    size_t nblocks, min_blocks, max_blocks;
    unsigned mxm;
    unsigned tranA, tranB;
    GrB_Matrix A, B, M;
    GrB_Type type;
    GrB_Semiring semiring;
    GrB_Monoid monoid;

    /* region read program arguments */
    if (arg_to_str_def(argc, argv, "-A", &A_path, INPUTS_DIR"/simple") != OTX_SUCCESS) { return 1; }
    if (arg_to_uint64_def(argc, argv, "--niter", &niter, 1) != OTX_SUCCESS) { return 2; }
    if (arg_to_int32_def(argc, argv, "-nt|--nthreads", &num_threads, 1) != OTX_SUCCESS) { return 3; }
    if (arg_to_uint32_def(argc, argv, "--mxm", &mxm, 0) != OTX_SUCCESS) { return 6; }
    /* endregion */

    printf("%s; matrix: %s, num iterations: %lu, num threads: %d\n", mxm ? "mxm" : "tricnt",
           get_file_name(A_path), niter, num_threads);

    /* region init libraries */
    GrB_init(GrB_BLOCKING);
    GxB_Global_Option_set(GxB_NTHREADS, num_threads);
    /* endregion */

    /* region read matrix */
    timer_start();
    if (mxm) {
        if (read_Matrix_FP64(&A, A_path, false) != GrB_SUCCESS) { return 16; }
        GrB_Matrix_dup(&B, A);
        tranA = tranB = 0;
        M = NULL;
        type = GrB_FP64;
        semiring = GrB_PLUS_TIMES_SEMIRING_FP64;
        monoid = GrB_PLUS_MONOID_FP64;
    } else {
        readLU(&A, &B, A_path, 0, GrB_UINT64);
        GrB_Matrix_dup(&M, A);
        tranA = 0;
        tranB = 1;
        type = GrB_UINT64;
        semiring = GrB_PLUS_TIMES_SEMIRING_UINT64;
        monoid = GrB_PLUS_MONOID_UINT64;
    }

    printf("Read time: %lf\n", timer_end());
    /* endregion */

    baseline(get_file_name(A_path), M, A, B, tranA, tranB, niter, num_threads, type, semiring, monoid);

    /* region free resources and terminate libraries */
    GrB_Matrix_free(&A);
    GrB_Matrix_free(&B);
    GrB_finalize();
    /* endregion */

    return 0;
}