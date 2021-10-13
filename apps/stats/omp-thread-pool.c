//
// Created by sm108 on 2/26/21.
//

#define _GNU_SOURCE

#include <sched.h>
#include <omp.h>
#include <unistd.h>
#include <syscall.h>
#include <stdio.h>
#include <starpu_config.h>
#include <pthread.h>
#include <starpu.h>

struct thread_data {
    int num_threads;
    int niter;
    size_t log_work;
};

size_t work(size_t log_work) {
    return log_work > 1 ? work(log_work - 1) + work(log_work - 1) : 2;
}

void ompfun(const int num_threads, const size_t log_work) {
    int movedPreWork = 0;
    int movedPostWork = 0;

    pid_t x = syscall(__NR_gettid);
    printf("spu worker thread ID: %d\n", x);

#pragma omp parallel num_threads(num_threads) default(none) reduction(+:movedPreWork)
    {
        int cpuid = sched_getcpu();
        if (cpuid != omp_get_thread_num()) {
            movedPreWork++;
        }

        pid_t x = syscall(__NR_gettid);
//        printf("%d %d\n", x, cpuid);

        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(omp_get_thread_num(), &mask);
        int result = sched_setaffinity(0, sizeof(mask), &mask);
    }

#pragma omp parallel num_threads(num_threads) default(none) shared(log_work)
    {
        work(log_work);
    }

#pragma omp parallel num_threads(num_threads) default(none) reduction(+:movedPostWork)
    {
        int cpuid = sched_getcpu();
        if (cpuid != omp_get_thread_num()) {
            movedPostWork++;
        }

        pid_t x = syscall(__NR_gettid);
//        printf("%d %d\n", x, cpuid);

        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(omp_get_thread_num(), &mask);
        int result = sched_setaffinity(0, sizeof(mask), &mask);
    }

    printf("%d %d\n", movedPreWork, movedPostWork);
}

void *thread_fn(void *data) {
    struct thread_data *tdata = ((struct thread_data *) data);
    for (int i = 0; i < tdata->niter; i++) {
        ompfun(tdata->num_threads, tdata->log_work);
    }
    return NULL;
}

void starpu_fn(void *buffers[], void *clArgs) {
    struct thread_data *tdata = ((struct thread_data *) clArgs);
    ompfun(tdata->num_threads, tdata->log_work);
}

void starpu_submit(struct thread_data *data) {
    /* region init codelet */
    struct starpu_codelet cl;
    starpu_codelet_init(&cl);
    cl.cpu_funcs[0] = starpu_fn;
    cl.nbuffers = 0;
    /* endregion */

    for (int i = 0; i < data->niter; i++) {
        struct starpu_task *task = starpu_task_create();
        task->cl = &cl;
        task->cl_arg = data;
        task->cl_arg_size = sizeof(struct thread_data);
        task->cl_arg_free = 0;
        STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), "starpu_task_submit");
    }
    starpu_task_wait_for_all();
}

int main() {
    struct thread_data data = {.num_threads = 12, .niter = 10, .log_work = 30};

//    thread_fn(&data);
//
//    pthread_t pthread;
//    pthread_create(&pthread, NULL, thread_fn, &data);
//    pthread_join(pthread, NULL);


    STARPU_CHECK_RETURN_VALUE(starpu_init(NULL), "starpu_init");
    starpu_submit(&data);
    starpu_shutdown();

    return 0;
}
