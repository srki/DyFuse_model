/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_OP_ASSIGN_MATRIX_VALUE_H
#define GRB_FUSION_OP_ASSIGN_MATRIX_VALUE_H

#include <starpu.h>


namespace grb::detail {

    template<class CMatrixT, class MaskT, class ValueT, class IndexT, class NIndicesT>
    class OpAssignMatrixValue : public Operation {
    public:
        OpAssignMatrixValue(CMatrixT &C, MaskT &mask, GrB_BinaryOp accum, ValueT val, IndexT *rowIndices,
                            NIndicesT nrows, IndexT *colIndices, NIndicesT ncols, GrB_Descriptor desc)
                : Operation{OperationType::ASSIGN_MATRIX_VALUE}, _mC{C.getData()}, _mask{mask.getData()},
                  _args{accum, val, rowIndices, nrows, colIndices, ncols, desc} {
            initDependencies(C, mask);
            initCodelet();
            initTasks();
        }

        void initTasks() {
            for (size_t blockIdx = 0; blockIdx < _mC->getNumBlocks(); blockIdx++) {
                addTask(createTask(&_codelet, _args, _mC, blockIdx, _mask, blockIdx));
            }
        }


        void release() override {
            _mC.reset();
            _mask.reset();
        }


    private:
        const static bool USE_MASK = !std::is_same_v<MaskT, MatrixImpl < void, void>>
        ;

        DataPtr_t <CMatrixT> _mC;
        DataPtr_t <MaskT> _mask;

        std::tuple<GrB_BinaryOp, ValueT, IndexT *, NIndicesT, IndexT *, NIndicesT, GrB_Descriptor> _args;

        starpu_codelet _codelet;

        void initDependencies(CMatrixT &C, MaskT &mask) {
            addInputDependency(mask.getOp(), DependencyType::READ);
            for (const auto dep : C.getOp()->getOutputDependencies()) {
                if (dep.second == DependencyType::REUSE) { continue; }
            }

            addInputDependency(C.getOp(), DependencyType::WRITE);
            C.setOp(this);
        }

        void initCodelet() {
            starpu_codelet_init(&_codelet);
            _codelet.cpu_funcs[0] = [](void *buffers[], void *clArgs) {
                auto C = *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[0]);
                auto mask = USE_MASK ? *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[1]) : GrB_NULL;

                auto[accum, val, rowIndices, nrows, colIndices, ncols, desc] = *static_cast<decltype(_args) *>(clArgs);

                grbTry(CInterface<ValueT>::matrixAssignValue(C, mask, accum, val,
                                                             rowIndices, nrows, colIndices, ncols, desc));
            };
            _codelet.nbuffers = 1 + (USE_MASK ? 1 : 0);
            _codelet.modes[0] = std::get<0>(_args) == GrB_NULL ? STARPU_W : STARPU_RW;; // C
            _codelet.modes[1] = STARPU_R; // mask
        }
    };

}

#endif //GRB_FUSION_OP_ASSIGN_MATRIX_VALUE_H
