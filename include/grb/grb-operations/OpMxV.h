/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_OP_MXV_H
#define GRB_FUSION_OP_MXV_H

#include <starpu.h>

#include <grb/util/GraphBLASImpl.h>
#include <grb/objects/traits.h>
#include "grb/context/Operation.h"

namespace grb::detail {

    template<class WVectorT, class MaskT, class AMatrixT, class UVectorT>
    class OpMxV : public Operation {
    public:
        OpMxV(WVectorT &w, MaskT &mask, GrB_BinaryOp accum, GrB_Semiring op, AMatrixT &A, UVectorT &u,
              GrB_Descriptor desc) : Operation{OperationType::MXV}, _w{w.getData()}, _mask{mask.getData()},
                                     _mA{A.getData()}, _u{u.getData()}, _args{accum, op, desc} {
            initDependencies(w, mask, u, A);
            initCodelet();
            initTasks();
        }

        void initTasks() {
            /* Check whether A is transposed */
            GrB_Desc_Value transposed;
            grbTry(GxB_Descriptor_get(&transposed, std::get<2>(_args), GrB_INP0));

            size_t istride = transposed == GrB_TRAN ? 1 : _mA->getNumColsBlocked();
            size_t jstride = transposed == GrB_TRAN ? _mA->getNumColsBlocked() : 1;

            for (size_t i = 0; i < _mA->getNumRowsBlocked(); i++) {
                for (size_t j = 0; j < _mA->getNumColsBlocked(); j++) {
                    auto codelet = &(j == 0 ? _codeletA : _codeletB);
                    auto blockIdx = i * istride + j * jstride;
                    addTask(createTask(codelet, _args, _w, i, _mA, blockIdx, _u, j, _mask, i));
                }
            }
        }


        void release() override {
            _w.reset();
            _mask.reset();
            _mA.reset();
            _u.reset();
        }


    private:
        const static bool USE_MASK = !std::is_same_v<MaskT, VectorImpl < void, void>>
        ;

        DataPtr_t <WVectorT> _w;
        DataPtr_t <MaskT> _mask;
        DataPtr_t <AMatrixT> _mA;
        DataPtr_t <UVectorT> _u;

        std::tuple<GrB_BinaryOp, GrB_Semiring, GrB_Descriptor> _args;

        starpu_codelet _codeletA;
        starpu_codelet _codeletB;

        void initDependencies(WVectorT &w, MaskT &mask, UVectorT &u, AMatrixT &A) {
            addInputDependency(mask.getOp(), DependencyType::READ);
            addInputDependency(A.getOp(), DependencyType::READ);
            addInputDependency(u.getOp(), DependencyType::READ);

            for (const auto dep : w.getOp()->getOutputDependencies()) {
                if (dep.second == DependencyType::REUSE) { continue; }
                addInputDependency(dep.first, DependencyType::REUSE);
            }

            addInputDependency(w.getOp(), DependencyType::WRITE);
            w.setOp(this);
        }

        void initCodelet() {
            starpu_codelet_init(&_codeletA);
            _codeletA.cpu_funcs[0] = [](void *buffers[], void *clArgs) {
                auto w = *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[0]);
                auto A = *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[1]);
                auto u = *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[2]);
                auto mask = USE_MASK ? *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[3]) : GrB_NULL;
                auto[accum, op, desc] = *static_cast<decltype(_args) *>(clArgs);

                grbTry(GrB_mxv(w, mask, accum, op, A, u, desc));
            };
            _codeletA.nbuffers = 3 + (USE_MASK ? 1 : 0);
            _codeletA.modes[0] = std::get<0>(_args) == GrB_NULL ? STARPU_W : STARPU_RW;; // w
            _codeletA.modes[1] = STARPU_R; // A
            _codeletA.modes[2] = STARPU_R; // u
            _codeletA.modes[3] = STARPU_R; // mask

            starpu_codelet_init(&_codeletB);
            _codeletB.cpu_funcs[0] = [](void *buffers[], void *clArgs) {
                auto w = *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[0]);
                auto A = *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[1]);
                auto u = *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[2]);
                auto mask = USE_MASK ? *(GrB_Vector *) STARPU_VARIABLE_GET_PTR(buffers[3]) : GrB_NULL;

                auto[accum, op, desc] = *static_cast<decltype(_args) *>(clArgs);

                GrB_Monoid monoid;
                GxB_Semiring_add(&monoid, op);
                GxB_Monoid_operator(&accum, monoid);
                grbTry(GrB_mxv(w, mask, accum, op, A, u, desc));
            };
            _codeletB.nbuffers = 3 + (USE_MASK ? 1 : 0);
            _codeletB.modes[0] = static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE); // w
            _codeletB.modes[1] = STARPU_R; // A
            _codeletB.modes[2] = STARPU_R; // u
            _codeletB.modes[3] = STARPU_R; // mask
        }
    };

}

#endif //GRB_FUSION_OP_MXV_H
