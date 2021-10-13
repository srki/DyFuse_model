/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_OP_CTOR_H
#define GRB_FUSION_OP_CTOR_H

#include <starpu.h>

#include "grb/context/Operation.h"

namespace grb::detail {

    template <bool IsMatrix>
    class OpCtor : public Operation {

    public:
        OpCtor() : Operation(IsMatrix ? OperationType::MATRIX_OP : OperationType::VECTOR_OP) {}

        void release() override {}

    private:

    };

}

#endif //GRB_FUSION_OP_CTOR_H
