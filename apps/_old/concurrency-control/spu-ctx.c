
#include <starpu.h>

void create_worker_array(int **workers, int start, int end) {
    int nworkers = end - start + 1;
    *workers = calloc(nworkers, sizeof(int));
    for (int i = 0; i < nworkers; i++) { (*workers)[i] = start + i; }
}

typedef struct {
    size_t log_delay;
    char *text;
} args_t;

size_t delay(size_t log_delay) {
    return log_delay > 1 ? delay(log_delay - 1) + delay(log_delay - 1) : 2;
}

void func(void *buffers[], void *clArgs) {
    args_t *args = clArgs;

    printf("Start %s \n", args->text);
    printf("%zu\n", delay(args->log_delay));
    printf("End %s \n", args->text);
}

void* parallel_func(void *log_delay) {
    printf("Start %s \n", "Parallel func");
    printf("%zu\n", delay((size_t)log_delay));
    printf("End %s \n", "Parallel func");

    return NULL;
}

int main() {
    STARPU_CHECK_RETURN_VALUE(starpu_init(NULL), "starpu_init");

    /* region create codelet */
    struct starpu_codelet cl;
    starpu_codelet_init(&cl);
    cl.cpu_funcs[0] = func;
    cl.nbuffers = 0;
    /* endregion */

    int *workers_0_5;
    create_worker_array(&workers_0_5, 0, 5);
    unsigned ctx_0_5 = starpu_sched_ctx_create(workers_0_5, 6, "ctx_0_5", STARPU_SCHED_CTX_POLICY_NAME, "", 0);

    int *workers_6_11;
    create_worker_array(&workers_6_11, 6, 11);
    unsigned ctx_6_11 = starpu_sched_ctx_create(workers_6_11, 6, "ctx_6_11", STARPU_SCHED_CTX_POLICY_NAME, "", 0);

    struct starpu_task *task_0_5 = starpu_task_create();
    task_0_5->cl = &cl;
    args_t arg_0_5;
    arg_0_5.log_delay = 32;
    arg_0_5.text = "task 0-5";
    task_0_5->cl_arg = &arg_0_5;
    task_0_5->cl_arg_size = sizeof(args_t);
    task_0_5->cl_arg_free = 0;
    STARPU_CHECK_RETURN_VALUE(starpu_task_submit_to_ctx(task_0_5, ctx_0_5), "starput_task_submit_to_ctx");

    struct starpu_task *task_main = starpu_task_create();
    task_main->cl = &cl;
    args_t arg_main;
    arg_main.log_delay = 32;
    arg_main.text = "task main";
    task_main->cl_arg = &arg_main;
    task_main->cl_arg_size = sizeof(args_t);
    task_main->cl_arg_free = 0;
    STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task_main), "starput_task_submit_to_ctx");

    struct starpu_task *task_6_11 = starpu_task_create();
    task_6_11->cl = &cl;
    args_t arg_6_11;
    arg_6_11.log_delay = 32;
    arg_6_11.text = "task 6-11";
    task_6_11->cl_arg = &arg_6_11;
    task_6_11->cl_arg_size = sizeof(args_t);
    task_6_11->cl_arg_free = 0;
    STARPU_CHECK_RETURN_VALUE(starpu_task_submit_to_ctx(task_6_11, ctx_6_11), "starput_task_submit_to_ctx");

    starpu_sched_ctx_exec_parallel_code(parallel_func, (void*)33, ctx_0_5);
    starpu_shutdown();

    return 0;
}