/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_OP_VXM_H
#define GRB_FUSION_OP_VXM_H

#include <starpu.h>

#include <grb/util/Util.h>
#include <grb/objects/VectorImpl.h>
#include <grb/objects/traits.h>
#include <grb/grb-operations/tasking/TaskUtil.h>

namespace grb::detail {

    template<class WVectorT, class MaskT, class UVectorT, class AMatrixT>
    class OpVxM : public Operation {
    public:
        OpVxM(WVectorT &w, MaskT &mask, GrB_BinaryOp accum, GrB_Semiring op, UVectorT &u, AMatrixT &A,
              GrB_Descriptor desc) : Operation{OperationType::VXM}, _w{nullptr}, _mask{mask.getData()},
                                     _u{u.getData()}, _mA{A.getData()}, _args{accum, op, desc} {
            GrB_Desc_Value outp;
            grbTry(GxB_Descriptor_get(&outp, std::get<2>(_args), GrB_OUTP));
            bool fullClone = outp != GrB_REPLACE || accum != GrB_NULL;

            if ((w == u || w == mask) && w.getData()->getNumBlocks() > 1) {
                w.clone(fullClone);
            } else {
                /* Optimize: Clone is not be required. */
                if (!w.getOp()->getOutputDependencies().empty()) {
                    w.clone(fullClone);
                }
            }

            _w = w.getData();

            initDependencies(w, mask, u, A);
            initCodelets();
            initTasks();
        }

        void initTasks() {
            /* Check whether A is transposed */
            GrB_Desc_Value transposed;
            grbTry(GxB_Descriptor_get(&transposed, std::get<2>(_args), GrB_INP1));

            size_t istride = transposed == GrB_TRAN ? 1 : _mA->getNumColsBlocked();
            size_t jstride = transposed == GrB_TRAN ? _mA->getNumColsBlocked() : 1;

//            assert(_u->getNumBlocks() == _mA->getNumRowsBlocked());
//            assert(_w->getNumBlocks() == _mA->getNumColsBlocked());

            /* Submit tasks */
            for (size_t j = 0; j < _w->getNumBlocks(); j++) {
                for (size_t i = 0; i < _u->getNumBlocks(); i++) {
//                    std::cout << j << " += " << i << " * " << i * blockInc + j << std::endl;
                    auto codelet = &(i == 0 ? _codeletA : _codeletB);
                    auto blockIdx = i * istride + j * jstride;
                    addTask(createTask(codelet, _args, _w, j, _u, i, _mA, blockIdx, _mask, j),
                            "vxm:" + std::to_string(i) + ":" + std::to_string(j));
                }
            }
        }

        void release() override {
            _w.reset();
            _mask.reset();
            _u.reset();
            _mA.reset();
        }

    private:
        const static bool USE_MASK = !std::is_same_v<MaskT, VectorImpl<void, void>>;

        DataPtr_t<WVectorT> _w;
        DataPtr_t<MaskT> _mask;
        DataPtr_t<UVectorT> _u;
        DataPtr_t<AMatrixT> _mA;

        std::tuple<GrB_BinaryOp, GrB_Semiring, GrB_Descriptor> _args;

        starpu_codelet _codeletA;
        starpu_codelet _codeletB;

        void initDependencies(WVectorT &w, MaskT &mask, UVectorT &u, AMatrixT &A) {
            addInputDependency(mask.getOp(), DependencyType::READ);
            addInputDependency(u.getOp(), DependencyType::READ);
            addInputDependency(A.getOp(), DependencyType::READ);

            for (const auto dep : w.getOp()->getOutputDependencies()) {
                if (dep.second == DependencyType::REUSE) { continue; }
                addInputDependency(dep.first, DependencyType::REUSE);
            }
            addInputDependency(w.getOp(), DependencyType::WRITE);
            w.setOp(this);
        }

        void initCodelets() {
            starpu_codelet_init(&_codeletA);
            _codeletA.cpu_funcs[0] = [](void *buffers[], void *clArgs) {
                auto w = *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[0]);
                auto u = *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[1]);
                auto A = *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[2]);
                auto mask = USE_MASK ? *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[3]) : GrB_NULL;

                auto[accum, op, desc] = *static_cast<decltype(_args) *>(clArgs);

                grbTry(GrB_vxm(w, mask, accum, op, u, A, desc));
            };
            _codeletA.nbuffers = 3 + (USE_MASK ? 1 : 0);
            _codeletA.modes[0] = std::get<0>(_args) == GrB_NULL ? STARPU_W : STARPU_RW;; // w
            _codeletA.modes[1] = STARPU_R; // u
            _codeletA.modes[2] = STARPU_R; // A
            _codeletA.modes[3] = STARPU_R; // mask

            starpu_codelet_init(&_codeletB);
            _codeletB.cpu_funcs[0] = [](void *buffers[], void *clArgs) {
                auto w = *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[0]);
                auto u = *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[1]);
                auto A = *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[2]);
                auto mask = USE_MASK ? *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[3]) : GrB_NULL;

                auto[accum, op, desc] = *static_cast<decltype(_args) *>(clArgs);

                GrB_Monoid monoid;
                GxB_Semiring_add(&monoid, op);
                GxB_Monoid_operator(&accum, monoid);
                grbTry(GrB_vxm(w, mask, accum, op, u, A, desc));
            };
            _codeletB.nbuffers = 3 + (USE_MASK ? 1 : 0);
            _codeletB.modes[0] = static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE); // w
            _codeletB.modes[1] = STARPU_R; // u
            _codeletB.modes[2] = STARPU_R; // A
            _codeletB.modes[3] = STARPU_R; // mask

        }
    };

}

#endif //GRB_FUSION_OP_VXM_H
