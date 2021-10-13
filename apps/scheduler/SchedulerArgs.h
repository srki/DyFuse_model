#ifndef GRB_FUSION_SCHEDULER_ARGS_H
#define GRB_FUSION_SCHEDULER_ARGS_H

struct SchedulerArgs {
    enum class EnumTaskType : int {
        Uninitialized,
        UserTask,
        PinTask,
        ResumeTasks,
        SuspendTask,
        SignalTaskMerge,
        SignalTaskSplit
    };

    int workerId;
    EnumTaskType taskType;
};

#endif //GRB_FUSION_SCHEDULER_ARGS_H
