//
// Created by sm108 on 12/28/20.
//

#include <atomic>
#include <cstddef>
#include <omp.h>
#include <starpu.h>

#define NUM_QUEUES 3
#define NUM_CORES 12

/* region Util */

size_t delay(size_t log_delay) {
    return log_delay > 0 ? delay(log_delay - 1) + delay(log_delay - 1) : 1;
}

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

unsigned mainCtx;
std::atomic_size_t tasksLeft[NUM_QUEUES] = {};
int queueNumWorkers[NUM_QUEUES];

typedef struct {
    size_t logDelay;
    size_t id;
    int queueId;
} args_t;


void mxm_func(void *buffers[], void *clArgs) {
    auto args = static_cast<args_t *>(clArgs);
    printf("%2d %3zu\n", args->queueId, args->id);

if (args->queueId == 0) {
#pragma omp parallel  num_threads(6)
    {
        printf("Here\n");
        delay(args->logDelay);
    }
}

    delay(args->logDelay);
    auto left = tasksLeft[args->queueId].fetch_add(-1) - 1;

    if (args->queueId == NUM_QUEUES - 1) { return; }

    auto requiredCtxs = NUM_CORES / queueNumWorkers[args->queueId];
    if (left >= requiredCtxs) { return; }

    auto workerId = starpu_worker_get_id();
    auto unblockWorkerStride = queueNumWorkers[args->queueId + 1];
    starpu_sched_ctx_unblock_workers_in_parallel_range(mainCtx,
                                                       workerId + unblockWorkerStride,
                                                       queueNumWorkers[args->queueId] / unblockWorkerStride - 1,
                                                       unblockWorkerStride);
//    for (int i = 0; i < queueNumWorkers[args->queueId] / unblockWorkerStride - 1; i++) {
//        printf("%d ", workerId + unblockWorkerStride + i * unblockWorkerStride);
//    }
//    printf("\n");

}

int main(int argc, char *argv[]) {
    STARPU_CHECK_RETURN_VALUE(starpu_init(NULL), "starpu_init");

    mainCtx = create_ctx(0, NUM_CORES - 1);

    queueNumWorkers[0] = NUM_CORES;
    queueNumWorkers[1] = 4;
    queueNumWorkers[2] = 1;

    tasksLeft[0] = 2;
    tasksLeft[1] = 5;
    tasksLeft[2] = 12;

    starpu_sched_ctx_block_workers_in_parallel_range(mainCtx, 1, NUM_CORES - 1, 1);
//    delay(32);
    /* region create codelet */
    starpu_codelet cl{};
    starpu_codelet_init(&cl);
    cl.cpu_funcs[0] = mxm_func;
    cl.nbuffers = 0;
    /* endregion */

    for (int i = 0; i < NUM_QUEUES; i++) {
        for (int j = 0; j < tasksLeft[i]; j++) {
            struct starpu_task *task = starpu_task_create();
            task->cl = &cl;
            auto args = static_cast<args_t *>(malloc(sizeof(args_t)));
            args->logDelay = 30;
            args->id = j;
            args->queueId = i;
            task->cl_arg = args;
            task->cl_arg_size = sizeof(args_t);
            task->cl_arg_free = 1;
            STARPU_CHECK_RETURN_VALUE(starpu_task_submit_to_ctx(task, mainCtx), "starpu_task_submit_to_ctx");
        }
    }
    starpu_task_wait_for_all();

    starpu_sched_ctx_unblock_workers_in_parallel_range(mainCtx, 1, NUM_CORES - 1, 1);

    starpu_shutdown();
}