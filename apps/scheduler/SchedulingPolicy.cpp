#include "SchedulingPolicy.h"

#include <starpu.h>
#include <starpu_sched_component.h>
#include "ConcurrencyControlComponent.h"

void initScheduler(unsigned ctxId) {
    assert(starpu_combined_worker_get_count() == 0);

    starpu_sched_ctx_create_worker_collection(ctxId, STARPU_WORKER_LIST);

    starpu_sched_tree *tree = starpu_sched_tree_create(ctxId);

    starpu_sched_component *fifoComponent = starpu_sched_component_fifo_create(tree, nullptr);
    tree->root = fifoComponent;

    ConcurrencyControlArgs concurrencyControlArgs{.numWorkers = static_cast<int>(starpu_worker_get_count())};

    auto concurrencyControlComponent = createConcurrencyControlComponent(tree, &concurrencyControlArgs);
    tree->root = concurrencyControlComponent;

    starpu_sched_component_fifo_data fifoData = {
            .ntasks_threshold = 0,
            .exp_len_threshold = 1e10
    };

    for (unsigned i = 0; i < starpu_worker_get_count(); ++i) {
        starpu_sched_component *workerComponent = starpu_sched_component_worker_new(ctxId, int(i));
        starpu_sched_component *workerFifoComponent = starpu_sched_component_fifo_create(tree, &fifoData);
        starpu_sched_component_connect(workerFifoComponent, workerComponent);

        starpu_sched_component_connect(concurrencyControlComponent, workerFifoComponent);
    }

    starpu_sched_tree_update_workers(tree);
    starpu_sched_ctx_set_policy_data(ctxId, tree);
}

void deinitScheduler(unsigned ctxId) {
    auto *tree = static_cast<starpu_sched_tree *>(starpu_sched_ctx_get_policy_data(ctxId));
    starpu_sched_tree_destroy(tree);
//    starpu_sched_ctx_delete_worker_collection(ctxId);
}

starpu_sched_policy schedulingPolicy =
        {
                .init_sched = initScheduler,
                .deinit_sched = deinitScheduler,
                .push_task = starpu_sched_tree_push_task,
                .pop_task = starpu_sched_tree_pop_task,
                .pop_every_task = nullptr,
                .pre_exec_hook = starpu_sched_component_worker_pre_exec_hook,
                .post_exec_hook = starpu_sched_component_worker_post_exec_hook,
                .add_workers = starpu_sched_tree_add_workers,
                .remove_workers = starpu_sched_tree_remove_workers,
                .policy_name = "tree-eager-prefetching",
                .policy_description = "eager with prefetching tree policy"
        };