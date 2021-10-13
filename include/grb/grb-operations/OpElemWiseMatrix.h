/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_OP_ELEM_WISE_MATRIX_H
#define GRB_FUSION_OP_ELEM_WISE_MATRIX_H

#include <starpu.h>

#include <grb/grb-operations/OpElemWiseType.h>

namespace grb::detail {

    template<OpElemWiseType OpType, class CMatrixT, class MaskT, class OpT, class AMatrixT, class BMatrixT>
    class OpElemWiseMatrix : public Operation {
        static_assert(std::is_same_v<OpT, GrB_Semiring> ||
                      std::is_same_v<OpT, GrB_Monoid> ||
                      std::is_same_v<OpT, GrB_BinaryOp>);

    public:
        OpElemWiseMatrix(CMatrixT &C, MaskT &mask, GrB_BinaryOp accum, OpT op, AMatrixT &A, BMatrixT &B,
                         GrB_Descriptor desc) : Operation{
                OpType == OpElemWiseType::MULTIPLY ? OperationType::eWISE_MULTIPLICATION : OperationType::eWISE_ADD},
                                                _mC{C.getData()}, _mask{mask.getData()}, _mA{A.getData()},
                                                _mB{B.getData()}, _args(accum, op, desc) {
            /* Optimize: determine if w should be cloned */
            initDependencies(C, mask, A, B);
            initCodelet();
            initTasks();
        }

        void initTasks() {
            assert(_mC->getNumBlocks() == _mA->getNumBlocks() && _mC->getNumBlocks() == _mB->getNumBlocks());
            for (size_t i = 0; i < _mC->getNumBlocks(); i++) {
                addTask(createTask(&_codelet, _args, _mC, i, _mA, i, _mB, i, _mask, i));
            }
        }


        void release() override {
            _mC.reset();
            _mask.reset();
            _mA.reset();
            _mB.reset();
        }


    private:
        const static bool USE_MASK = !std::is_same_v<MaskT, MatrixImpl < void, void>>
        ;

        DataPtr_t <CMatrixT> _mC;
        DataPtr_t <MaskT> _mask;
        DataPtr_t <AMatrixT> _mA;
        DataPtr_t <BMatrixT> _mB;

        std::tuple<GrB_BinaryOp, OpT, GrB_Descriptor> _args;

        starpu_codelet _codelet;

        void initDependencies(CMatrixT &C, MaskT &mask, AMatrixT &A, BMatrixT &B) {
            addInputDependency(mask.getOp(), DependencyType::READ);
            addInputDependency(A.getOp(), DependencyType::READ);
            addInputDependency(B.getOp(), DependencyType::READ);

            for (const auto dep : C.getOp()->getOutputDependencies()) {
                if (dep.second == DependencyType::REUSE) { continue; }
                addInputDependency(dep.first, DependencyType::REUSE);
            }

            /* If C has output dependencies, this dependency will be transitive */
            addInputDependency(C.getOp(), DependencyType::WRITE);

            C.setOp(this);
        }

        void initCodelet() {
            starpu_codelet_init(&_codelet);
            _codelet.cpu_funcs[0] = [](void *buffers[], void *clArgs) {
                auto C = *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[0]);
                auto A = *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[1]);
                auto B = *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[2]);
                auto mask = USE_MASK ? *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[3]) : GrB_NULL;

                auto[accum, op, desc] = *static_cast<decltype(_args) *>(clArgs);

                switch (OpType) {
                    case OpElemWiseType::ADD:
                        if constexpr (std::is_same_v<OpT, GrB_Semiring>) {
                            grbTry(GrB_Matrix_eWiseAdd_Semiring(C, mask, accum, op, A, B, desc));
                        } else if constexpr (std::is_same_v<OpT, GrB_Monoid>) {
                            grbTry(GrB_Matrix_eWiseAdd_Monoid(C, mask, accum, op, A, B, desc));
                        } else if constexpr (std::is_same_v<OpT, GrB_BinaryOp>) {
                            grbTry(GrB_Matrix_eWiseAdd_BinaryOp(C, mask, accum, op, A, B, desc));
                        }
                        break;

                    case OpElemWiseType::MULTIPLY:
                        if constexpr (std::is_same_v<OpT, GrB_Semiring>) {
                            grbTry(GrB_Matrix_eWiseMult_Semiring(C, mask, accum, op, A, B, desc));
                        } else if constexpr (std::is_same_v<OpT, GrB_Monoid>) {
                            grbTry(GrB_Matrix_eWiseMult_Monoid(C, mask, accum, op, A, B, desc));
                        } else if constexpr (std::is_same_v<OpT, GrB_BinaryOp>) {
                            grbTry(GrB_Matrix_eWiseMult_BinaryOp(C, mask, accum, op, A, B, desc));
                        }
                        break;
                }
            };
            _codelet.nbuffers = 3 + (USE_MASK ? 1 : 0);
            _codelet.modes[0] = std::get<0>(_args) == GrB_NULL ? STARPU_W : STARPU_RW;; // C
            _codelet.modes[1] = STARPU_R; // A
            _codelet.modes[2] = STARPU_R; // B
            _codelet.modes[3] = STARPU_R; // mask
        }
    };

    template<OpElemWiseType OpType, class CMatrixT, class MaskT, class OpT, class AMatrixT, class BMatrixT>
    auto newOpElemWiseMatrix(CMatrixT &C, MaskT &mask, GrB_BinaryOp accum, OpT op, AMatrixT &A, BMatrixT &B,
                             GrB_Descriptor desc) {
        return new OpElemWiseMatrix<OpType, CMatrixT, MaskT, OpT, AMatrixT, BMatrixT>(C, mask, accum, op, A, B, desc);
    }

}

#endif //GRB_FUSION_OP_ELEM_WISE_MATRIX_H
