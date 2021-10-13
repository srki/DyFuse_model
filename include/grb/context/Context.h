/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_CONTEXT_H
#define GRB_FUSION_CONTEXT_H

#include <vector>
#include <cstdint>

/* region forward declarations */
namespace grb::detail {
    class Operation;
}

namespace grb::detail {

    class Context {
    public:
        static Context& getDefaultContext();

        const std::vector<Operation*> &getOperations();

        Context& addOperation(Operation *op);

        Context& addOperationAndWait(Operation *op);

        Context& wait();

        void releaseOps();

    private:
        std::vector<Operation*> _operations;
//        std::vector<Operation*> _pendingOperations;
//        OpWait *_lastWait;

    public:
        uint32_t _currentGroupId{0};

    };

}

#endif //GRB_FUSION_CONTEXT_H
