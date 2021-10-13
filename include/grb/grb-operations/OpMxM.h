/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_OP_MXM_H
#define GRB_FUSION_OP_MXM_H

#include <starpu.h>

#include <grb/util/GraphBLASImpl.h>
#include <grb/objects/traits.h>
#include "grb/context/Operation.h"

namespace grb::detail {

    template<class CMatrixT, class MaskT, class AMatrixT, class BMatrixT>
    class OpMxM : public Operation {
    public:
        OpMxM(CMatrixT &C, MaskT &mask, GrB_BinaryOp accum, GrB_Semiring op, AMatrixT &A, BMatrixT &B,
              GrB_Descriptor desc) : Operation{OperationType::MXM}, _mC{C.getData()}, _mask{mask.getData()},
                                     _mA{A.getData()}, _mB{B.getData()}, _args(accum, op, desc) {
            GrB_Desc_Value outp;
            grbTry(GxB_Descriptor_get(&outp, std::get<2>(_args), GrB_OUTP));
            bool fullClone = outp != GrB_REPLACE || accum != GrB_NULL;

            if ((C == mask || C == A || C == B) && C.getData()->getNumBlocks() > 1) {
                C.clone(fullClone);
            } else {
                /* Optimize: Clone is not required. */
                if (!C.getOp()->getOutputDependencies().empty()) {
                    C.clone(fullClone);
                }
            }

            _mC = C.getData();

            initDependencies(C, mask, A, B);
            initCodelet();
            initTasks();
        }

        void initTasks() {
            GrB_Desc_Value transposed;
            /* Check whether A is transposed */
            size_t ARowStride = _mA->getNumColsBlocked();
            size_t AColStride = 1;
            grbTry(GxB_Descriptor_get(&transposed, std::get<2>(_args), GrB_INP0));
            if (transposed == GrB_TRAN) { std::swap(ARowStride, AColStride); }

            /* Check whether B is transposed */
            size_t BRowStride = _mB->getNumColsBlocked();
            size_t BColStride = 1;
            grbTry(GxB_Descriptor_get(&transposed, std::get<2>(_args), GrB_INP1));
            if (transposed == GrB_TRAN) { std::swap(BRowStride, BColStride); }

            auto kEnd = transposed == GrB_TRAN ? _mB->getNumColsBlocked() : _mB->getNumRowsBlocked();

            /* Submit tasks */
            for (size_t i = 0; i < _mC->getNumRowsBlocked(); i++) {
                for (size_t j = 0; j < _mC->getNumColsBlocked(); j++) {
                    for (size_t k = 0; k < kEnd; k++) {
                        auto codelet = &(k == 0 ? _codeletA : _codeletB);

                        auto CBlockIdx = i * _mC->getNumColsBlocked() + j;
                        auto ABlockIdx = i * ARowStride + k * AColStride;
                        auto BBlockIdx = k * BRowStride + j * BColStride;

                        addTask(createTask(codelet, _args,
                                           _mC, CBlockIdx, _mA, ABlockIdx, _mB, BBlockIdx, _mask, CBlockIdx),
                                "mxm:" + std::to_string(i) + ":" + std::to_string(j) + ":" + std::to_string(k));
                    }
                }
            }
        }


        void release() override {
            _mC.reset();
            _mask.reset();
            _mA.reset();
            _mB.reset();
        }

    private:
        const static bool USE_MASK = !std::is_same_v<MaskT, MatrixImpl<void, void>>;

        DataPtr_t<CMatrixT> _mC;
        DataPtr_t<MaskT> _mask;
        DataPtr_t<AMatrixT> _mA;
        DataPtr_t<BMatrixT> _mB;

        std::tuple<GrB_BinaryOp, GrB_Semiring, GrB_Descriptor> _args;

        starpu_codelet _codeletA;
        starpu_codelet _codeletB;

        void initDependencies(CMatrixT &C, MaskT &mask, AMatrixT &A, BMatrixT &B) {
            addInputDependency(mask.getOp(), DependencyType::READ);
            addInputDependency(A.getOp(), DependencyType::READ);
            addInputDependency(B.getOp(), DependencyType::READ);

            for (const auto dep : C.getOp()->getOutputDependencies()) {
                if (dep.second == DependencyType::REUSE) { continue; }
                addInputDependency(dep.first, DependencyType::REUSE);
            }

            addInputDependency(C.getOp(), DependencyType::WRITE);
            C.setOp(this);
        }

        void initCodelet() {
            starpu_codelet_init(&_codeletA);
            _codeletA.cpu_funcs[0] = [](void *buffers[], void *clArgs) {
                auto C = *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[0]);
                auto A = *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[1]);
                auto B = *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[2]);
                auto mask = USE_MASK ? *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[3]) : GrB_NULL;

                auto[accum, op, desc] = *static_cast<decltype(_args) *>(clArgs);

                grbTry(GrB_mxm(C, mask, accum, op, A, B, desc));
            };
            _codeletA.nbuffers = 3 + (USE_MASK ? 1 : 0);
            _codeletA.modes[0] = std::get<0>(_args) == GrB_NULL ? STARPU_W : STARPU_RW;; // C
            _codeletA.modes[1] = STARPU_R; // A
            _codeletA.modes[2] = STARPU_R; // B
            _codeletA.modes[3] = STARPU_R; // mask

            starpu_codelet_init(&_codeletB);
            _codeletB.cpu_funcs[0] = [](void *buffers[], void *clArgs) {
                auto C = *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[0]);
                auto A = *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[1]);
                auto B = *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[2]);
                auto mask = USE_MASK ? *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[3]) : GrB_NULL;

                auto[accum, op, desc] = *static_cast<decltype(_args) *>(clArgs);
                GrB_Monoid monoid;
                GxB_Semiring_add(&monoid, op);
                GxB_Monoid_operator(&accum, monoid);

                grbTry(GrB_mxm(C, mask, accum, op, A, B, desc));
            };
            _codeletB.nbuffers = 3 + (USE_MASK ? 1 : 0);
            _codeletB.modes[0] = static_cast<starpu_data_access_mode>(STARPU_RW | STARPU_COMMUTE); // w
            _codeletB.modes[1] = STARPU_R; // A
            _codeletB.modes[2] = STARPU_R; // B
            _codeletB.modes[3] = STARPU_R; // mask
        }
    };

}

#endif //GRB_FUSION_OP_MXM_H
