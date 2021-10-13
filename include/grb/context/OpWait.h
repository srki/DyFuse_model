/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_WAIT_H
#define GRB_FUSION_WAIT_H

#include "Operation.h"

namespace grb::detail {

    class OpWait : public Operation {
    public:
        OpWait(const std::vector<Operation*> &pendingOps) : Operation(OperationType::WAIT) {
            initDependencies(pendingOps);
        }

        void release() override {}

    private:
        void initDependencies(const std::vector<Operation*> &pendingOps) {
            for (auto op : pendingOps) {
                if (op->getOutputDependencies().empty()) {
                    addInputDependency(op, DependencyType::WAIT);
                }
            }
        }
    };

}

#endif //GRB_FUSION_WAIT_H
