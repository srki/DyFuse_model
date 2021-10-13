/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */
#define _GNU_SOURCE
#include <sched.h>
#undef _GNU_SOURCE

#include <starpu.h>
#include <omp.h>


#ifdef STARPU_QUICK_CHECK
#define NTASKS 8
#else
#define NTASKS 3
#endif
#define SIZE 4000

size_t delay(size_t log_delay) {
    return log_delay > 1 ? delay(log_delay - 1) + delay(log_delay - 1) : 2;
}

/* Codelet SUM */
static void sum_cpu(void *descr[], void *cl_arg) {
    delay(32);
    #pragma omp parallel default(none)
    {
//        #pragma omp single
        printf("A Thread %d/%d; CPU %u\n", omp_get_thread_num(), omp_get_num_threads(), sched_getcpu());
        delay(32);
    };

//    int size;
//    starpu_codelet_unpack_args(cl_arg, &size);
////    fprintf(stderr, "sum_cpu\n");
//    int i, k;
//#pragma omp parallel
////    fprintf(stderr, "hello from the task %d\n", omp_get_thread_num());
//    for (k=0;k<10;k++)
//    {
//#pragma omp parallel for
//        for (i=0; i<size; i++)
//        {
//            v_dst[i]+=v_src0[i]+v_src1[i];
//        }
//    }
}

static struct starpu_codelet sum_cl =
        {
                .cpu_funcs = {sum_cpu, NULL},
                .nbuffers = 0
        };

int main(void) {
//    int ntasks = NTASKS;
//    int ret, i;
//    struct starpu_cluster_machine *clusters;
//
//    setenv("STARPU_NMIC", "0", 1);
//    setenv("STARPU_NMPI_MS", "0", 1);
//
//    ret = starpu_init(NULL);
//    if (ret == -ENODEV)
//        return 77;
//    STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");
//
//    /* We regroup resources under each sockets into a cluster. We express a partition
//     * of one socket to create two internal clusters */
//    clusters = starpu_cluster_machine(HWLOC_OBJ_SOCKET,
//                                      STARPU_CLUSTER_PARTITION_ONE,
//                                      STARPU_CLUSTER_NEW,
////					  STARPU_CLUSTER_TYPE, STARPU_CLUSTER_OPENMP,
////					  STARPU_CLUSTER_TYPE, STARPU_CLUSTER_INTEL_OPENMP_MKL,
//                                      STARPU_CLUSTER_NB, 2,
//                                      STARPU_CLUSTER_NCORES, 4,
//
//                                      STARPU_CLUSTER_NEW,
//                                      STARPU_CLUSTER_NB, 2,
//                                      STARPU_CLUSTER_NCORES, 6,
//
//                                      STARPU_CLUSTER_NEW,
//                                      STARPU_CLUSTER_NB, 2,
//                                      STARPU_CLUSTER_NCORES, 6,
//                                      0);
//    starpu_cluster_print(clusters);
//
//    int size = SIZE;
//
//    for (i = 0; i < ntasks; i++) {
//        ret = starpu_task_insert(&sum_cl,
//                                 STARPU_VALUE, &size, sizeof(int),
//
//                /* For two tasks, try out the case when the task isn't parallel and expect
//                   the configuration to be sequential due to this, then automatically changed
//                   back to the parallel one */
//                                 STARPU_POSSIBLY_PARALLEL, (i <= 4 || i > 6) ? 1 : 0,
//
//                /* Note that this mode requires that you put a prologue callback managing
//                   this on all tasks to be taken into account. */
//                                 STARPU_PROLOGUE_CALLBACK_POP, &starpu_openmp_prologue,
//
//                                 0);
//
//        if (ret == -ENODEV)
//            goto out;
//        STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
//    }
//
//
//    out:
//    /* wait for all tasks at the end*/
//    starpu_task_wait_for_all();
//
//    starpu_uncluster_machine(clusters);
//
//    starpu_shutdown();
    return 0;
}
