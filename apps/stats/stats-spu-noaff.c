#define _GNU_SOURCE

#include <sched.h>

#include <starpu.h>
#include <GraphBLAS.h>
#include <otx/otx.h>
#include "../util/timer.h"

#define TYPE_GrB GrB_UINT64

#include "../util/GrB_util.h"
#include "../util/block_matrix.h"
#include "../util/reader_c.h"
#include "../util/exec_info.h"

size_t delay(size_t log_delay) {
    return log_delay > 1 ? delay(log_delay - 1) + delay(log_delay - 1) : 2;
}

typedef struct {
    GrB_Matrix M;
    GrB_Matrix C;
    GrB_Matrix A;
    GrB_Matrix B;
    int block_parallelism;
    GrB_Descriptor desc;
    GrB_Semiring semi;

    size_t i;
    size_t j;
    #if defined(HAVE_EXEC_INFO_SUPPORT)
    struct exec_info *info;
    #endif
} args_t;

void mxm_func(void *buffers[], void *clArgs) {
    args_t *args = clArgs;

#if defined(HAVE_EXEC_INFO_SUPPORT)
    memset(args->info, 0, sizeof(struct exec_info));
#endif

    GrB_mxm(args->C, args->M, GrB_NULL, args->semi, args->A, args->B, args->desc);

#if defined(HAVE_EXEC_INFO_SUPPORT)
    printf("BLOCK;%zu;%zu;%d;%d;%lf;", args->i,  args->j, 0, 0, 0.0);
    print_block_info(args->info, 0);
#endif


}

void spu(const char *name, GrB_Matrix *bC, GrB_Matrix *bM, GrB_Matrix *bA, GrB_Matrix *bB, unsigned transpose_A,
         unsigned transpose_B, size_t niter, size_t ncores, size_t block_parallelism, size_t nblocks, int use_task,
         GrB_Type type, GrB_Semiring semiring, GrB_Monoid monoid) {
    double time, time_total = 0.0;

    GrB_Descriptor desc;

#if defined(HAVE_EXEC_INFO_SUPPORT)
    struct exec_info *info = malloc(sizeof(struct exec_info));
    assert(block_parallelism == ncores);
#endif

    OK(GrB_Descriptor_new(&desc));
    OK(GxB_Desc_set(desc, GrB_OUTP, GrB_REPLACE));
    if (transpose_A) { GxB_Desc_set(desc, GrB_INP0, GrB_TRAN); }
    if (transpose_B) { GxB_Desc_set(desc, GrB_INP1, GrB_TRAN); }
    OK(GxB_Desc_set(desc, GxB_DESCRIPTOR_NTHREADS, block_parallelism));

#if defined(HAVE_EXEC_INFO_SUPPORT)
    OK(GxB_Desc_set(desc, GxB_EXEC_INFO, info));
#endif


    int *workers;
    unsigned ctx;
    struct starpu_codelet cl;
    size_t nworkers;

    /* region create context and block workers */
    workers = malloc(ncores * sizeof(int));
    for (int i = 0; i < ncores; i++) { workers[i] = i; }
    ctx = starpu_sched_ctx_create(workers, ncores, "", STARPU_SCHED_CTX_POLICY_NAME, "", 0);
    free(workers);

    nworkers = ncores / block_parallelism;
    for (int i = 0; i < nworkers; i++) {
        starpu_sched_ctx_block_workers_in_parallel_range(ctx, i * (int) block_parallelism + 1,
                                                         (int) block_parallelism - 1, 1);
    }
    /* endregion */

    /* region init codelet */
    starpu_codelet_init(&cl);
    cl.cpu_funcs[0] = mxm_func;
    cl.nbuffers = 0;
    /* endregion */

    for (size_t iter = 0; iter < niter; iter++) {
        timer_start();
        for (size_t i = 0; i < nblocks; i++) {
            for (size_t j = 0; j < nblocks; j++) {
                struct starpu_task *task = starpu_task_create();
                task->cl = &cl;
                args_t *args = malloc(sizeof(args_t));
                args->C = bC[i * nblocks + j];
                args->M = bM != NULL ? bM[i * nblocks + j] : GrB_NULL;
                args->A = bA[i];
                args->B = bB[j];
                args->desc = desc;
                args->block_parallelism = block_parallelism;
                args->semi = semiring;
                args->i = i;
                args->j = j;

#if defined(HAVE_EXEC_INFO_SUPPORT)
                args->info = info;
#endif

                task->cl_arg = args;
                task->cl_arg_size = sizeof(args_t);
                task->cl_arg_free = 1;
                STARPU_CHECK_RETURN_VALUE(starpu_task_submit_to_ctx(task, ctx), "starpu_task_submit_to_ctx");
            }
        }
        starpu_task_wait_for_all();
        time = timer_end();
        time_total += time;

        if (type == GrB_FP64) {
            double sum = 0.0;
            for (size_t b = 0; b < nblocks * nblocks; b++) {
                OK(GrB_Matrix_reduce_FP64(&sum, GrB_PLUS_FP64, GrB_PLUS_MONOID_FP64, bC[b], GrB_NULL));
            }
            printf("LOG;spu-iter;%s;%zu;%lf;%lf\n", name, nblocks, time, sum);
        } else if (type == GrB_UINT64) {
            uint64_t sum = 0;
            for (size_t b = 0; b < nblocks * nblocks; b++) {
                OK(GrB_Matrix_reduce_UINT64(&sum, GrB_PLUS_UINT64, GrB_PLUS_MONOID_UINT64, bC[b], GrB_NULL));
            }
            printf("LOG;spu-iter;%s;%zu;%lf;%lu\n", name, nblocks, time, sum);
        }
    }
    printf("LOG;spu-mean;%s;%zu;%zu;%zu;%lf\n", name, nblocks, nworkers, block_parallelism, time_total / niter);

    /* region unblock workers and free resources */
    for (int i = 0; i < nworkers; i++) {
        starpu_sched_ctx_unblock_workers_in_parallel_range(ctx, i * (int) block_parallelism + 1,
                                                           (int) block_parallelism - 1, 1);
    }
    starpu_sched_ctx_delete(ctx);
    /* endregion */
}

void blocked(const char *name, GrB_Matrix M, GrB_Matrix A, GrB_Matrix B, unsigned transpose_A, unsigned transpose_B,
             size_t niter, size_t ncores, size_t nblocks, GrB_Type type, GrB_Semiring semiring, GrB_Monoid monoid) {
    assert(!transpose_A);
    GrB_Matrix C;
    GrB_Matrix *bA, *bB, *bC, *bM;

    GrB_Index m, n, l;
    GrB_Index bm, bn, bl;
    size_t block_size_m, block_size_n, block_size_l;

    GrB_Matrix_nrows(&m, A);
    GrB_Matrix_ncols(&n, B);
    GrB_Matrix_ncols(&l, A);

    block_size_m = (m + nblocks - 1) / nblocks;
    block_size_n = (n + nblocks - 1) / nblocks;
    block_size_l = l;

    GrB_Matrix_new(&C, type, m, n);

    create_blocks_type(&bA, &bm, &bl, A, block_size_m, block_size_l, 0, type);

    if (transpose_B) {
        create_blocks_type(&bB, &bn, &bl, B, block_size_n, block_size_l, 0, type);
    } else {
        create_blocks_type(&bB, &bl, &bn, B, block_size_l, block_size_n, 0, type);
    }

    create_blocks_type(&bC, &bm, &bn, C, block_size_m, block_size_n, 0, type);

    if (M != GrB_NULL) {
        create_blocks_type(&bM, &bm, &bn, M, block_size_m, block_size_n, 0, type);
    } else {
        bM = NULL;
    }

    if (nblocks != bm || nblocks != bn || 1 != bl) {
        exit(5);
    }

    for (size_t i = 1; i <= ncores; i++) {
//        if (ncores % i != 0) { continue; }
        if (i != ncores) { continue; }
        spu(name, bC, bM, bA, bB, transpose_A, transpose_B, niter, ncores, i, nblocks, 0, type, semiring, monoid);
    }

    free_blocks(bA, bm, bl);
    free_blocks(bB, bl, bn);
    free_blocks(bC, bm, bn);
    GrB_free(&C);
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
    if (arg_to_uint64_def(argc, argv, "-minb|--minBlocks", &min_blocks, 1) != OTX_SUCCESS) { return 4; }
    if (arg_to_uint64_def(argc, argv, "-maxb|--maxBlocks", &max_blocks, 4) != OTX_SUCCESS) { return 5; }
    if (arg_to_uint32_def(argc, argv, "--mxm", &mxm, 0) != OTX_SUCCESS) { return 6; }
    /* endregion */

    printf("%s; matrix: %s, num iterations: %lu, num threads: %d; blocks: %lu-%lu\n", mxm ? "mxm" : "tricnt",
           get_file_name(A_path), niter, num_threads, min_blocks, max_blocks);

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

    STARPU_CHECK_RETURN_VALUE(starpu_init(NULL), "starpu_init");

    nblocks = min_blocks;
    while (nblocks <= max_blocks) {
        blocked(get_file_name(A_path), M, A, B, tranA, tranB, niter, num_threads, nblocks, type, semiring, monoid);

        if (nblocks == 1) { nblocks = 2; }
        else if (((nblocks - 1) & nblocks) == 0) { nblocks = nblocks * 3 / 2; }
        else { nblocks = ((nblocks - 1) & nblocks) * 2; }
    }

    starpu_shutdown();

    /* region free resources and terminate libraries */
    GrB_Matrix_free(&A);
    GrB_Matrix_free(&B);
    GrB_finalize();
    /* endregion */

    return 0;
}
