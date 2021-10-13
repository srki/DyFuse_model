//
// Created by sm108 on 12/28/20.
//

#include <atomic>
#include <cstddef>
#include <omp.h>
#include <starpu.h>
#include <otx/otx.h>
#include <iostream>

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

#include "../../util/GrB_util.h"
#include "../../util/block_matrix.h"
#include "../../util/timer.h"


#define NUM_QUEUES 3
#define NUM_CORES 1

/* region Util */

unsigned create_ctx(int start, int end) {
    auto nworkers = end - start + 1;
    auto workers = new int[nworkers];
    for (int i = 0; i < nworkers; i++) { workers[i] = start + i; }

//    char ctxName[80];
//    sprintf(ctxName, "ctx_%d_%d", start, end);

    auto ctx = starpu_sched_ctx_create(workers, nworkers, "", STARPU_SCHED_CTX_POLICY_NAME, "", 0);
    delete[] workers;

    return ctx;
}

/* endregion */

std::atomic_size_t tasks_left[NUM_QUEUES] = {};
std::atomic_size_t worker_state[NUM_QUEUES] = {};

int queueNumWorkers[NUM_QUEUES];


typedef struct {
    GrB_Matrix M;
    GrB_Matrix A;
    GrB_Matrix B;
    GrB_Matrix C;
    size_t id;
    int queueId;
    unsigned ctx;
} args_t;


void mxm_func(void *buffers[], void *clArgs) {
    auto args = static_cast<args_t *>(clArgs);
//    printf("%2d %3zu\n", args->queueId, args->id);

    /* region perform mxm */
    GrB_Descriptor desc;
    GrB_Descriptor_new(&desc);
    GxB_Desc_set(desc, GrB_INP1, GrB_TRAN);
    GxB_Desc_set(desc, GrB_OUTP, GrB_REPLACE);
    GxB_Desc_set(desc, GxB_DESCRIPTOR_NTHREADS, static_cast<GrB_Desc_Value>(queueNumWorkers[args->queueId]));
//    std::cout << queueNumWorkers[args->queueId] << std::endl;

    GrB_mxm(args->C, args->M, GrB_NULL, TYPE_PLUS_TIMES, args->A, args->B, desc);

    uint32_t sum = 0;
    GrB_Matrix_reduce_UINT32(&sum, GrB_NULL, GrB_PLUS_MONOID_UINT32, args->C, GrB_NULL);
//    std::cout << sum << std::endl;
    /* endregion */

    /* region unblock treads if necessary */
    auto left = tasks_left[args->queueId].fetch_add(-1) - 1;

    if (args->queueId == NUM_QUEUES - 1) { return; }

    auto requiredCtxs = NUM_CORES / queueNumWorkers[args->queueId];
    if (left >= requiredCtxs) { return; }

    auto workerId = starpu_worker_get_id();
    auto unblockWorkerStride = queueNumWorkers[args->queueId + 1];
//    starpu_sched_ctx_unblock_workers_in_parallel_range(args->ctx,
//                                                       workerId + unblockWorkerStride,
//                                                       queueNumWorkers[args->queueId] / unblockWorkerStride - 1,
//                                                       unblockWorkerStride);
    starpu_sched_ctx_unblock_workers_in_parallel_range(args->ctx, 1, 5, 1);
    /* endregion */
}

void init_starpu_worker_data() {
    for (size_t i = 0; i < NUM_QUEUES; i++) {

    }
}

void tricnt(const char *name, GrB_Matrix A, GrB_Matrix B, GrB_Matrix M,
            size_t nblocks_m, size_t nblocks_n, size_t nblocks_l,
            size_t niter, size_t *nthreadsPerBLock, GrB_Type type) {
    auto ctx = create_ctx(0, NUM_CORES - 1);

    /* region block the inputs */
    if (nblocks_l != 1) {
        std::cerr << "nblocks l must be 1" << std::endl;
    }

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
//        exit(1);
    }
    /* endregion */

    /* region create codelet */
    starpu_codelet cl{};
    starpu_codelet_init(&cl);
    cl.cpu_funcs[0] = mxm_func;
    cl.nbuffers = 0;
    /* endregion */

    for (int i = 0; i < NUM_QUEUES; i++) {
        tasks_left[i] = 0;
        for (int j = 0; j < nblocks_m * nblocks_n; j++) {
            if (nthreadsPerBLock[j] == queueNumWorkers[i]) {
                tasks_left[i]++;
            }
        }
        printf("%2dt - %zu\n", queueNumWorkers[i], tasks_left[i].load());
    }

    for (int iter = 0; iter < niter; iter++) {
//        if (tasksLeft[0] != 0) {
//            starpu_sched_ctx_block_workers_in_parallel_range(ctx, 1, 11, 1);
//        } else if (tasksLeft[1] != 0) {
//            starpu_sched_ctx_block_workers_in_parallel_range(ctx, 1, 5, 1);
//        }

        timer_start();
        for (int qId = 0; qId < NUM_QUEUES; qId++) {
            for (size_t i = 0; i < bm; i++) {
                for (size_t j = 0; j < bn; j++) {
                    if (nthreadsPerBLock[j] != queueNumWorkers[qId]) { continue; }
                    auto task = starpu_task_create();
                    task->cl = &cl;
                    auto args = static_cast<args_t *>(malloc(sizeof(args_t)));
                    args->id = i * bn + j;
                    args->A = bA[i * bl];
                    args->B = bB[j * bl];
                    args->C = bC[i * bn + j];
                    args->M = bM[i * bn + j];
                    args->queueId = qId;
                    args->ctx = ctx;
                    task->cl_arg = args;
                    task->cl_arg_size = sizeof(args_t);
                    task->cl_arg_free = 1;
                    STARPU_CHECK_RETURN_VALUE(starpu_task_submit_to_ctx(task, ctx), "starpu_task_submit_to_ctx");
                }
            }
        }

        starpu_task_wait_for_all();
        auto time = timer_end();
        std::cout << time << std::endl;

        starpu_sched_ctx_unblock_workers_in_parallel_range(ctx, 1, NUM_CORES - 1, 1);
    }

    starpu_sched_ctx_delete(ctx);
}

int main(int argc, char *argv[]) {
    STARPU_CHECK_RETURN_VALUE(starpu_init(NULL), "starpu_init");
    OK(GrB_init(GrB_NONBLOCKING))

    queueNumWorkers[0] = 12;
    queueNumWorkers[1] = 6;
    queueNumWorkers[2] = 1;

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
            nthreadsPerBlock[0] = 6;
            break;
        default:
            std::cerr << "Option not specified" << std::endl;
    }

    std::cout << "Start " << opt << std::endl;
    tricnt(get_file_name(path.c_str()), L, U, L, 6, 6, 1, niter, nthreadsPerBlock, GrB_UINT32);

    GrB_Matrix_free(&L);
    GrB_Matrix_free(&U);

    OK(GrB_finalize());
    starpu_shutdown();
}