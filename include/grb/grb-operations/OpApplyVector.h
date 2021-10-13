/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_OP_APPLY_VECTOR_H
#define GRB_FUSION_OP_APPLY_VECTOR_H

#include <starpu.h>
#include <grb/grb-operations/tasking/TaskUtil.h>

namespace grb::detail {

    template<class WVectorT, class MaskT, class UVectorT>
    class OpApplyVector : public Operation {
    public:
        OpApplyVector(WVectorT &w, MaskT &mask, GrB_BinaryOp accum, GrB_UnaryOp op, UVectorT u, GrB_Descriptor desc)
                : Operation{OperationType::APPLY_MATRIX}, _w{w.getData()}, _mask{mask.getData()}, _u{u.getData()},
                  _args(accum, op, desc) {
            initDependencies(w, mask, u);
            initCodelet();
            initTasks();
        }

        void initTasks() {
            for (size_t i = 0; i < _w->getNumBlocks(); i++) {
                addTask(createTask(&_codelet, _args, _w, i, _u, i, _mask, i));
            }
        }


        void release() override {
            _w.reset();
            _mask.reset();
            _u.reset();
        }


    private:
        const static bool USE_MASK = !std::is_same_v<MaskT, VectorImpl < void, void>>
        ;

        DataPtr_t <WVectorT> _w;
        DataPtr_t <MaskT> _mask;
        DataPtr_t <UVectorT> _u;

        std::tuple<GrB_BinaryOp, GrB_UnaryOp, GrB_Descriptor> _args;

        starpu_codelet _codelet;

        void initDependencies(WVectorT &w, MaskT &mask, UVectorT &u) {
            addInputDependency(mask.getOp(), DependencyType::READ);
            addInputDependency(u.getOp(), DependencyType::READ);

            for (const auto dep : w.getOp()->getOutputDependencies()) {
                if (dep.second == DependencyType::REUSE) { continue; }
                addInputDependency(dep.first, DependencyType::REUSE);
            }

            addInputDependency(w.getOp(), DependencyType::WRITE);
            w.setOp(this);
        }

        void initCodelet() {
            starpu_codelet_init(&_codelet);
            _codelet.cpu_funcs[0] = [](void *buffers[], void *clArgs) {
                auto w = *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[0]);
                auto u = *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[1]);
                auto mask = USE_MASK ? *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[2]) : GrB_NULL;


                auto[accum, op, desc] = *static_cast<decltype(_args) *>(clArgs);
                grbTry(GrB_Vector_apply(w, mask, accum, op, u, desc));
            };
            _codelet.nbuffers = 2 + (USE_MASK ? 1 : 0);
            _codelet.modes[0] = std::get<0>(_args) == GrB_NULL ? STARPU_W : STARPU_RW;; // w
            _codelet.modes[1] = STARPU_R; // u
            _codelet.modes[2] = STARPU_R; // mask
        }
    };

}

#endif //GRB_FUSION_OP_APPLY_VECTOR_H
