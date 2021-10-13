/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_TASK_UTIL_H
#define GRB_FUSION_TASK_UTIL_H

#include <starpu.h>

namespace grb::detail {

    template<size_t idx = 0, class Object, class ...Objects>
    inline void initHandles(starpu_task *task, const Object &obj, size_t blockIdx, const Objects &... objs) {
        if constexpr (std::is_same_v<VectorData<void, void>, Object> ||
                      std::is_same_v<MatrixData<void, void>, Object>) {
            if constexpr (sizeof...(Objects) > 0) { initHandles<idx>(task, objs...); }
        } else {
            task->handles[idx] = obj->getHandle(blockIdx);
            task->modes[idx] = task->cl->modes[idx];

            if constexpr (sizeof...(Objects) > 0) { initHandles<idx + 1>(task, objs...); }
        }
    }

    template<class Arg, class ...Objects>
    inline starpu_task *createTask(starpu_codelet *cl, Arg &arg, const Objects &... objs) {
        auto task = starpu_task_create();
        task->cl = cl;
        task->cl_arg = &arg;
        task->cl_arg_size = sizeof(Arg);
        task->cl_arg_free = 0;

        if constexpr (sizeof...(objs) != 0) { initHandles(task, objs...); }

        return task;
    }

    /* endregion */

}

#endif //GRB_FUSION_TASK_UTIL_H
