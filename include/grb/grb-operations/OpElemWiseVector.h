/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_OP_ELEM_WISE_VECTOR_H
#define GRB_FUSION_OP_ELEM_WISE_VECTOR_H

#include <starpu.h>

#include <grb/grb-operations/OpElemWiseType.h>

namespace grb::detail {


    template<OpElemWiseType OpType, class WVectorT, class MaskT, class OpT, class UVectorT, class VVectorT>
    class OpElemWiseVector : public Operation {
        static_assert(std::is_same_v<OpT, GrB_Semiring> ||
                      std::is_same_v<OpT, GrB_Monoid> ||
                      std::is_same_v<OpT, GrB_BinaryOp>);

    public:

        OpElemWiseVector(WVectorT &w, MaskT &mask, GrB_BinaryOp accum, OpT op, UVectorT &u, VVectorT &v,
                         GrB_Descriptor desc) : Operation{
                OpType == OpElemWiseType::MULTIPLY ? OperationType::eWISE_MULTIPLICATION : OperationType::eWISE_ADD},
                                                _w{w.getData()}, _mask{mask.getData()},
                                                _u{u.getData()}, _v{v.getData()}, _args{accum, op, desc} {
            /* Optimize: determine if w should be cloned */
            initDependencies(w, mask, u, v);
            initCodelet();
            initTasks();
        }

        void initTasks() {
            assert(_w->getNumBlocks() == _u->getNumBlocks() && _w->getNumBlocks() == _v->getNumBlocks());
            for (size_t i = 0; i < _w->getNumBlocks(); i++) {
                addTask(createTask(&_codelet, _args, _w, i, _u, i, _v, i, _mask, i));
            }
        }


        void release() override {
            _w.reset();
            _mask.reset();
            _u.reset();
            _v.reset();
        }


    private:
        const static bool USE_MASK = !std::is_same_v<MaskT, VectorImpl<void, void>>;

        DataPtr_t<WVectorT> _w;
        DataPtr_t<MaskT> _mask;
        DataPtr_t<UVectorT> _u;
        DataPtr_t<VVectorT> _v;

        std::tuple<GrB_BinaryOp, OpT, GrB_Descriptor> _args;

        starpu_codelet _codelet;

        void initDependencies(WVectorT &w, MaskT &mask, UVectorT &u, VVectorT &v) {
            addInputDependency(mask.getOp(), DependencyType::READ);
            addInputDependency(u.getOp(), DependencyType::READ);
            addInputDependency(v.getOp(), DependencyType::READ);

            for (const auto dep : w.getOp()->getOutputDependencies()) {
                if (dep.second == DependencyType::REUSE) { continue; }
                addInputDependency(dep.first, DependencyType::REUSE);
            }

            /* If w has output dependencies, this dependency will be transitive */
            addInputDependency(w.getOp(), DependencyType::WRITE);

            w.setOp(this);
        }

        void initCodelet() {
            starpu_codelet_init(&_codelet);
            _codelet.cpu_funcs[0] = [](void *buffers[], void *clArgs) {
                auto w = *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[0]);
                auto u = *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[1]);
                auto v = *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[2]);
                auto mask = USE_MASK ? *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[3]) : GrB_NULL;

                auto[accum, op, desc] = *static_cast<decltype(_args) *>(clArgs);

                switch (OpType) {
                    case OpElemWiseType::ADD:
                        if constexpr (std::is_same_v<OpT, GrB_Semiring>) {
                            grbTry(GrB_Vector_eWiseAdd_Semiring(w, mask, accum, op, u, v, desc));
                        } else if constexpr (std::is_same_v<OpT, GrB_Monoid>) {
                            grbTry(GrB_Vector_eWiseAdd_Monoid(w, mask, accum, op, u, v, desc));
                        } else if constexpr (std::is_same_v<OpT, GrB_BinaryOp>) {
                            grbTry(GrB_Vector_eWiseAdd_BinaryOp(w, mask, accum, op, u, v, desc));
                        }
                        break;

                    case OpElemWiseType::MULTIPLY:
                        if constexpr (std::is_same_v<OpT, GrB_Semiring>) {
                            grbTry(GrB_Vector_eWiseMult_Semiring(w, mask, accum, op, u, v, desc));
                        } else if constexpr (std::is_same_v<OpT, GrB_Monoid>) {
                            grbTry(GrB_Vector_eWiseMult_Monoid(w, mask, accum, op, u, v, desc));
                        } else if constexpr (std::is_same_v<OpT, GrB_BinaryOp>) {
                            grbTry(GrB_Vector_eWiseMult_BinaryOp(w, mask, accum, op, u, v, desc));
                        }
                        break;
                }
            };
            _codelet.nbuffers = 3 + (USE_MASK ? 1 : 0);
            _codelet.modes[0] = std::get<0>(_args) == GrB_NULL ? STARPU_W : STARPU_RW;; // w
            _codelet.modes[1] = STARPU_R; // u
            _codelet.modes[2] = STARPU_R; // v
            _codelet.modes[3] = STARPU_R; // mask
        }
    };

    template<OpElemWiseType OpType, class WVectorT, class MaskT, class OpT, class UVectorT, class VVectorT>
    auto newOpElemWiseVector(WVectorT &w, MaskT &mask, GrB_BinaryOp accum, OpT op, UVectorT &u, VVectorT &v,
                             GrB_Descriptor desc) {
        return new OpElemWiseVector<OpType, WVectorT, MaskT, OpT, UVectorT, VVectorT>(w, mask, accum, op, u, v, desc);
    }
}

#endif //GRB_FUSION_OP_ELEM_WISE_VECTOR_H
