#include "ConcurrencyControlComponent.h"

#include <iostream>
#include "ControlTask.h"
#include "SchedulerArgs.h"

int isConcurrencyControlComponent(struct starpu_sched_component *component);


static int pushTask(struct starpu_sched_component *component, struct starpu_task *task) {
    STARPU_ASSERT(component && task && isConcurrencyControlComponent(component));
    STARPU_ASSERT(starpu_sched_component_can_execute_task(component, task));

    auto data = static_cast<ConcurrencyControlData *>(component->data);
    struct starpu_sched_component *target;

    auto schedulerData = reinterpret_cast<SchedulerArgs *>(task->sched_data);

    int workerId = 0;
    if (schedulerData) {
        workerId = schedulerData->workerId;

        switch (schedulerData->taskType) {
            case SchedulerArgs::EnumTaskType::Uninitialized: {
                assert(false);
                break;
            }

            case SchedulerArgs::EnumTaskType::UserTask: {
//                std::cout << "UserTask" << std::endl;
                break;
            }

            case SchedulerArgs::EnumTaskType::ResumeTasks: {
                auto args = static_cast<ControlTaskArgs *>(task->cl_arg);
                args->concurrencyControlData = data;
                break;
            }

            case SchedulerArgs::EnumTaskType::SuspendTask: {
                auto args = static_cast<ControlTaskArgs *>(task->cl_arg);
                args->concurrencyControlData = data;
                break;
            }

            case SchedulerArgs::EnumTaskType::SignalTaskSplit:
            case SchedulerArgs::EnumTaskType::SignalTaskMerge: {
                auto args = static_cast<SignalTaskArgs *>(task->cl_arg);
                args->concurrencyControlData = data;
                break;
            }

            case SchedulerArgs::EnumTaskType::PinTask: {
                break;
            }
        }
    }

    std::lock_guard lck(data->workerQueueLocks[workerId]);
    data->workerQueues[workerId].push(task);

    return 0;
}


static int canPush(struct starpu_sched_component *component, struct starpu_sched_component *to) {
    auto data = static_cast<ConcurrencyControlData *>(component->data);

    int workerId = starpu_bitmap_first(to->workers);

    std::lock_guard lck(data->workerQueueLocks[workerId]);
    if (data->workerQueues[workerId].empty() || data->workerStates[workerId] == 0) { return 1; }

//    std::cout << "A " << starpu_bitmap_first(to->workers) << std::endl;
    auto task = data->workerQueues[workerId].front();

    if (task->sched_data != nullptr) {
        auto schedulerData = static_cast<SchedulerArgs *>(task->sched_data);
        switch (schedulerData->taskType) {
            case SchedulerArgs::EnumTaskType::Uninitialized: {
                assert(false);
                break;
            }

            case SchedulerArgs::EnumTaskType::UserTask: {
//                std::cout << "UserTask" << std::endl;
                break;
            }

            case SchedulerArgs::EnumTaskType::PinTask: {
                break;
            }

            case SchedulerArgs::EnumTaskType::ResumeTasks:
            case SchedulerArgs::EnumTaskType::SuspendTask: {
                auto args = static_cast<ControlTaskArgs *>(task->cl_arg);
                if (args->signalVar->load() != 0) { return 1; }
                delete args->signalVar;
                break;
            }

            case SchedulerArgs::EnumTaskType::SignalTaskMerge: {
                auto args = static_cast<SignalTaskArgs *>(task->cl_arg);
                assert(data->workerStates[args->workerId] == 1);
                data->workerStates[args->workerId] = 0;
                break;
            }

            case SchedulerArgs::EnumTaskType::SignalTaskSplit: {
                auto args = static_cast<SignalTaskArgs *>(task->cl_arg);
//                data->workerStates[args->workerId] = 0;
                break;
            }
        }

        free(schedulerData);
    }

    int ret = starpu_sched_component_push_task(component, to, task);
    data->workerQueues[workerId].pop();

    return ret;
}


static void deinitData(struct starpu_sched_component *component) {
    STARPU_ASSERT(isConcurrencyControlComponent(component));
    delete static_cast<ConcurrencyControlData *>(component->data);
}

int isConcurrencyControlComponent(struct starpu_sched_component *component) {
    return component->push_task == pushTask;
}

starpu_sched_component *createConcurrencyControlComponent(struct starpu_sched_tree *tree, void *arg) {
    starpu_sched_component *component = starpu_sched_component_create(tree, "eager");

    auto &args = *static_cast<ConcurrencyControlArgs *>(arg);

    auto data = new ConcurrencyControlData{args.numWorkers};

    component->data = data;
    component->push_task = pushTask;
    component->can_push = canPush;
    component->can_pull = starpu_sched_component_can_pull_all;
    component->deinit_data = deinitData;

    return component;
}
