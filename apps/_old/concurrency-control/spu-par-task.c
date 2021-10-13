//
// Created by sm108 on 1/13/21.
//

#include <starpu.h>

#define DELAY 32

size_t delay(size_t log_delay) {
    return log_delay > 1 ? delay(log_delay - 1) + delay(log_delay - 1) : 2;
}

void spu_cpu_kernel(void *buffers[], void *clArgs) {
    printf("%d\n", starpu_combined_worker_get_size());
    #pragma omp parallel default(none)
    delay(32);
}

int main() {
    struct starpu_conf conf;
    starpu_conf_init(&conf);
    conf.single_combined_worker = 0;
    conf.sched_policy_name = "peager";
    STARPU_CHECK_RETURN_VALUE(starpu_init(&conf), "starpu_init");

    struct starpu_codelet cl[3];
    for (int i = 0; i < 3; i++) {
        starpu_codelet_init(&cl[i]);
        cl[i].cpu_funcs[0] = spu_cpu_kernel;
        cl[i].type = STARPU_FORKJOIN;
        switch (i) {
            case 0:
                cl[i].max_parallelism = 1;
                break;
            case 1:
                cl[i].max_parallelism = 6;
                break;
            case 2:
                cl[i].max_parallelism = 12;
                break;
        }
    }


    for (int i = 0; i < 10; i++) {
        struct starpu_task *task = starpu_task_create();

        if (i < 3) {
            task->cl = &cl[2];
        } else if (i < 8) {
            task->cl = &cl[1];
        } else {
            task->cl = &cl[0];
        }
        STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit");
    }

    starpu_task_wait_for_all();


    return 0;
}