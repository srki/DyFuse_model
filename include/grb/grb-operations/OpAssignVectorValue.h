/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_OP_ASSIGN_VECTOR_VALUE_H
#define GRB_FUSION_OP_ASSIGN_VECTOR_VALUE_H

#include <starpu.h>


namespace grb::detail {

    template<class WVectorT, class MaskT, class ValueT, class IndexT, class NIndicesT>
    class OpAssignVectorValue : public Operation {
    public:
        OpAssignVectorValue(WVectorT &w, MaskT &mask, GrB_BinaryOp accum, ValueT val,
                            IndexT *indices, NIndicesT nindices, GrB_Descriptor desc)
                : Operation{OperationType::ASSIGN_VECTOR_VALUE}, _w{w.getData()}, _mask{mask.getData()},
                  _args{accum, val, indices, nindices, desc} {
            initDependencies(w, mask);
            initCodelet();
            initTasks();
        }

        void initTasks() {
            for (size_t i = 0; i < _w->getNumBlocks(); i++) {
                addTask(createTask(&_codelet, _args, _w, i, _mask, i),
                        "assignVectorValue:" + std::to_string(i));
            }
        }

        void release() override {
            _w.reset();
            _mask.reset();
        }

    private:
        const static bool USE_MASK = !std::is_same_v<MaskT, VectorImpl < void, void>>
        ;

        DataPtr_t <WVectorT> _w;
        DataPtr_t <MaskT> _mask;

        std::tuple<GrB_BinaryOp, ValueT, IndexT *, NIndicesT, GrB_Descriptor> _args;

        starpu_codelet _codelet;

        void initDependencies(WVectorT &w, MaskT &mask) {
            addInputDependency(mask.getOp(), DependencyType::READ);
            for (const auto dep : w.getOp()->getOutputDependencies()) {
                if (dep.second == DependencyType::REUSE) { continue; }
            }

            addInputDependency(w.getOp(), DependencyType::WRITE);
            w.setOp(this);
        }

        void initCodelet() {
            starpu_codelet_init(&_codelet);
            _codelet.cpu_funcs[0] = [](void *buffers[], void *clArgs) {
                auto w = *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[0]);
                auto mask = USE_MASK ? *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[1]) : GrB_NULL;


                auto[accum, val, indices, nIndices, desc] = *static_cast<decltype(_args) *>(clArgs);
                grbTry(CInterface<ValueT>::vectorAssignValue(w, mask, accum, val, indices, nIndices, desc));
            };
            _codelet.nbuffers = 1 + (USE_MASK ? 1 : 0);
            _codelet.modes[0] = std::get<0>(_args) == GrB_NULL ? STARPU_W : STARPU_RW; // w
            _codelet.modes[1] = STARPU_R; // mask
        }
    };

}

#endif //GRB_FUSION_OP_ASSIGN_VECTOR_VALUE_H
