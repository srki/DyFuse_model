/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_OP_APPLY_MATRIX_H
#define GRB_FUSION_OP_APPLY_MATRIX_H

#include <starpu.h>
#include <grb/grb-operations/tasking/TaskUtil.h>

namespace grb::detail {

    template<class CMatrixT, class MaskT, class AMatrixT>
    class OpApplyMatrix : public Operation {
    public:
        OpApplyMatrix(CMatrixT &C, MaskT &mask, GrB_BinaryOp accum, GrB_UnaryOp op, AMatrixT A, GrB_Descriptor desc)
                : Operation(OperationType::APPLY_MATRIX), _mC{C.getData()}, _mask{mask.getData()}, _mA{A.getData()},
                  _args(accum, op, desc) {
            initDependencies(C, mask, A);
            initCodelet();
            initTasks();
        }

        void initTasks() {
            /* Check whether A is transposed */
            GrB_Desc_Value transposed;
            grbTry(GxB_Descriptor_get(&transposed, std::get<2>(_args), GrB_INP0));

            size_t istride = transposed == GrB_TRAN ? 1 : _mA->getNumColsBlocked();
            size_t jstride = transposed == GrB_TRAN ? _mA->getNumColsBlocked() : 1;

            for (size_t i = 0; i < _mC->getNumRowsBlocked(); i++) {
                for (size_t j = 0; j < _mC->getNumColsBlocked(); j++) {
                    auto CBlockIdx = i * _mC->getNumColsBlocked() + j;
                    auto ABlockIdx = i * istride + j * jstride;
                    addTask(createTask(&_codelet, _args, _mC, CBlockIdx, _mA, ABlockIdx, _mask, CBlockIdx));
                }
            }
        }

        void release() override {
            _mC.reset();
            _mask.reset();
            _mA.reset();
        }

    private:
        const static bool USE_MASK = !std::is_same_v<MaskT, MatrixImpl < void, void>>
        ;

        DataPtr_t <CMatrixT> _mC;
        DataPtr_t <MaskT> _mask;
        DataPtr_t <AMatrixT> _mA;

        std::tuple<GrB_BinaryOp, GrB_UnaryOp, GrB_Descriptor> _args;

        starpu_codelet _codelet;

        void initDependencies(CMatrixT &C, MaskT &mask, AMatrixT &A) {
            addInputDependency(mask.getOp(), DependencyType::READ);
            addInputDependency(A.getOp(), DependencyType::READ);

            for (const auto dep : C.getOp()->getOutputDependencies()) {
                if (dep.second == DependencyType::REUSE) { continue; }
                addInputDependency(dep.first, DependencyType::REUSE);
            }

            addInputDependency(C.getOp(), DependencyType::WRITE);
            C.setOp(this);
        }

        void initCodelet() {
            starpu_codelet_init(&_codelet);
            _codelet.cpu_funcs[0] = [](void *buffers[], void *clArgs) {
                auto C = *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[0]);
                auto A = *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[1]);
                auto mask = USE_MASK ? *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[2]) : GrB_NULL;

                auto[accum, op, desc] = *static_cast<decltype(_args) *>(clArgs);
                grbTry(GrB_Matrix_apply(C, mask, accum, op, A, desc));
            };
            _codelet.nbuffers = 2 + (USE_MASK ? 1 : 0);
            _codelet.modes[0] = std::get<0>(_args) == GrB_NULL ? STARPU_W : STARPU_RW; // C
            _codelet.modes[1] = STARPU_R; // A
            _codelet.modes[2] = STARPU_R; // mask
        }

    };

}

#endif //GRB_FUSION_OP_APPLY_MATRIX_H
