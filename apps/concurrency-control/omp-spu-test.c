#define _GNU_SOURCE

#include <sched.h>

#include <starpu.h>
#include <omp.h>

#define DELAY 32

size_t delay(size_t log_delay) {
    return log_delay > 1 ? delay(log_delay - 1) + delay(log_delay - 1) : 2;
}

unsigned create_ctx(int start, int end) {
    int nworkers = end - start + 1;
    int *workers = malloc(nworkers * sizeof(int));
    for (int i = 0; i < nworkers; i++) { workers[i] = start + i; }

    unsigned ctx = starpu_sched_ctx_create(workers, nworkers, "", STARPU_SCHED_CTX_POLICY_NAME, "", 0);
    free(workers);

    return ctx;
}

typedef struct {

} args_t;

void spu_cpu_kernel(void *buffers[], void *clArgs) {
    int myId = sched_getcpu();
    printf("A %d\n", sched_getcpu());
    delay(DELAY);


    #pragma omp parallel num_threads(3) default(none) shared(myId)
    {
        printf("D %d\n", sched_getcpu());
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(myId + omp_get_thread_num(), &mask);
        int result = sched_setaffinity(0, sizeof(mask), &mask);
        printf("E %d\n", sched_getcpu());
//            delay(DELAY);
    }

    #pragma omp parallel num_threads(3) default(none)
    {
        printf("A Thread %d/%d; CPU %u\n", omp_get_thread_num(), omp_get_num_threads(), sched_getcpu());
        delay(DELAY);
        printf("B Thread %d/%d; CPU %u\n", omp_get_thread_num(), omp_get_num_threads(), sched_getcpu());
    }
}


int main(int argc, char *argv[]) {
    omp_set_nested(1);
    const size_t NUM_CORES = 12;

    STARPU_CHECK_RETURN_VALUE(starpu_init(NULL), "starpu_init");

    /* region create codelet */
    struct starpu_codelet cl;
    starpu_codelet_init(&cl);
    cl.cpu_funcs[0] = spu_cpu_kernel;
    cl.nbuffers = 0;
    /* endregion */

    /* region block workers */
    unsigned ctx = create_ctx(0, NUM_CORES - 1);
    starpu_sched_ctx_block_workers_in_parallel_range(ctx, 0, 12, 1);
    starpu_sched_ctx_unblock_workers_in_parallel_range(ctx, 0, 4, 3);
    /* endregion */

    for (int i = 0; i < 4; i++) {
        struct starpu_task *task = starpu_task_create();
        task->cl = &cl;
        task->cl_arg = NULL;
        STARPU_CHECK_RETURN_VALUE(starpu_task_submit_to_ctx(task, ctx), "starpu_task_submit_to_ctx");
    }

    starpu_task_wait_for_all();


    starpu_sched_ctx_unblock_workers_in_parallel_range(ctx, 0, 12, 1);

    starpu_shutdown();
    return 0;
}
