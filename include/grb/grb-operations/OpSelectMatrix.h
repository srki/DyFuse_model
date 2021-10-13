/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_OP_SELECT_MATRIX_H
#define GRB_FUSION_OP_SELECT_MATRIX_H

#include <starpu.h>
#include <grb/grb-operations/tasking/TaskUtil.h>

namespace grb::detail {

    template<class CMatrixT, class MaskT, class AMatrixT, class ScalarT>
    class OpSelectMatrix : public Operation {
    public:
        OpSelectMatrix(CMatrixT &C, MaskT &mask, GrB_BinaryOp accum, GxB_SelectOp op, AMatrixT &A, ScalarT Thunk,
                       GrB_Descriptor desc) : Operation{OperationType::SELECT_MATRIX}, _mC{C.getData()},
                                              _mask{mask.getData()}, _mA{A.getData()}, _args(accum, op, Thunk, desc) {
            if (op != GxB_TRIL && op != GxB_TRIU) {
                fprintf(stderr, "OpSelect supports only GxB_TRIL and GxB_TRIU");
                exit(-1);
            }

            initDependencies(C, mask, A);
            initCodelet();
            initBlockArgs();
            initTasks();
        }

        void initTasks() {
            /* Check whether A is transposed */
            GrB_Desc_Value transposed;
            grbTry(GxB_Descriptor_get(&transposed, std::get<3>(_args), GrB_INP0));

            if (transposed == GrB_TRAN) {
                fprintf(stderr, "transpose in not supported for OpSelect");
                exit(-1);
            }

            size_t istride = transposed == GrB_TRAN ? 1 : _mA->getNumColsBlocked();
            size_t jstride = transposed == GrB_TRAN ? _mA->getNumColsBlocked() : 1;

            for (size_t i = 0; i < _mC->getNumRowsBlocked(); i++) {
                for (size_t j = 0; j < _mC->getNumColsBlocked(); j++) {
                    auto CBlockIdx = i * _mC->getNumColsBlocked() + j;
                    auto ABlockIdx = i * istride + j * jstride;
                    addTask(createTask(&_codelet, _blockArgs[i * _mC->getNumColsBlocked() + j],
                                       _mC, CBlockIdx, _mA, ABlockIdx, _mask, CBlockIdx));
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

        std::tuple<GrB_BinaryOp, GxB_SelectOp, ScalarT, GrB_Descriptor> _args;
        std::unique_ptr<std::tuple<GrB_BinaryOp, GxB_SelectOp, ScalarT, GrB_Descriptor>[]> _blockArgs;

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

                auto[accum, op, scalar, desc] = *static_cast<decltype(_args) *>(clArgs);

                if (op == GxB_TRIL || op == GxB_TRIU) {
                    GxB_Scalar thunk;
                    grbTry(GxB_Scalar_new(&thunk, CInterface<ScalarT>::type()));
                    CInterface<ScalarT>::scalarSetElement(thunk, scalar);
                    grbTry(GxB_Matrix_select(C, mask, accum, op, A, thunk, desc));
                    grbTry(GxB_Scalar_free(&thunk));
                }
            };
            _codelet.nbuffers = 2 + (USE_MASK ? 1 : 0);
            _codelet.modes[0] = std::get<0>(_args) == GrB_NULL ? STARPU_W : STARPU_RW; // C
            _codelet.modes[1] = STARPU_R; // A
            _codelet.modes[2] = STARPU_R; // mask
        }

        void initBlockArgs() {
            auto nrowsBlocked = _mC->getNumRowsBlocked();
            auto ncolsBlocked = _mC->getNumColsBlocked();

            _blockArgs = std::make_unique<std::tuple<GrB_BinaryOp, GxB_SelectOp, ScalarT, GrB_Descriptor>[]>(
                    nrowsBlocked * ncolsBlocked);

            auto nrowsPerBlock = _mC->getNumRowsPerBlock();
            auto ncolsPerBlock = _mC->getNumColsPerBlock();

            for (size_t i = 0; i < nrowsBlocked; i++) {
                for (size_t j = 0; j < ncolsBlocked; j++) {
                    std::get<0>(_blockArgs[i * ncolsBlocked + j]) = std::get<0>(_args);
                    std::get<1>(_blockArgs[i * ncolsBlocked + j]) = std::get<1>(_args);
                    std::get<2>(_blockArgs[i * ncolsBlocked + j])
                            = std::get<2>(_args) + ssize_t(nrowsPerBlock * i - ncolsPerBlock * j);
                    std::get<3>(_blockArgs[i * ncolsBlocked + j]) = std::get<3>(_args);
                }
            }

        }

    };

}


#endif //GRB_FUSION_OP_SELECT_MATRIX_H
