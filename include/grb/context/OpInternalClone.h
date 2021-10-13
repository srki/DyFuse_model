/* LICENSE PLACEHOLDER */

/* TODO: move file */

#ifndef GRB_FUSION_OP_INTERNAL_CLONE_H
#define GRB_FUSION_OP_INTERNAL_CLONE_H

#include <grb/context/Operation.h>

namespace grb::detail {
    class OpInternalClone : public Operation {
    public:
        explicit OpInternalClone(const std::string &name) : Operation(OperationType::INTERNAL_CLONE) {
            setName(name);
        }

        void release() override {}
    };
}

#endif //GRB_FUSION_OP_INTERNAL_CLONE_H
