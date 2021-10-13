/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_OP_EXTRACT_MATRIX_H
#define GRB_FUSION_OP_EXTRACT_MATRIX_H

#include <starpu.h>


namespace grb::detail {

    template<class CMatrixT, class MaskT, class AMatrixT, class RowIndexT, class NRowIndicesT,
            class ColIndexT, class NColIndicesT>
    class OpExtractMatrix : public Operation {
    public:
        OpExtractMatrix(CMatrixT &C, MaskT &mask, GrB_BinaryOp accum, AMatrixT &A, RowIndexT *rowIndices,
                        NRowIndicesT nrows, ColIndexT *colIndices, NColIndicesT ncols, GrB_Descriptor desc)
                : Operation{OperationType::EXTRACT_MATRIX}, _mC{C.getData()}, _mask{mask.getData()}, _mA{A.getData()},
                  _args{accum, rowIndices, nrows, colIndices, ncols, desc} {
            initDependencies(C, mask, A);
            initCodelet();

            /* Optimize: this is extremely inefficient */
            /* Create temporary GrB matrix from Matrix A */
            GrB_Matrix ATmp;
            if (_mA->getNumBlocks() > 1) {
                using ScalarT = typename MatrixInfo<AMatrixT>::scalar_t;
                using IndexT = typename MatrixInfo<AMatrixT>::index_t;

                auto nvals = A.nvals();
                auto I = new IndexT[nvals];
                auto J = new IndexT[nvals];
                auto X = new ScalarT[nvals];
                A.extractTuples(I, J, X, &nvals);

                grbTry(GrB_Matrix_new(&ATmp, CInterface<ScalarT>::type(), A.nrows(), A.ncols()));
                grbTry(CInterface<ScalarT>::matrixBuild(ATmp, I, J, X, nvals, CInterface<ScalarT>::plus()));

                delete[] I;
                delete[] J;
                delete[] X;
            } else {
                ATmp = _mA->getBlock(0);
            }

            /* Extract Mask */
            GrB_Matrix maskTmp = GrB_NULL;
            if constexpr (USE_MASK) {
                if (_mA->getNumBlocks() > 1) {
                    using ScalarT = typename MatrixInfo<MaskT>::scalar_t;
                    using IndexT = typename MatrixInfo<MaskT>::index_t;

                    auto nvals = mask.nvals();
                    auto I = new IndexT[nvals];
                    auto J = new IndexT[nvals];
                    auto X = new ScalarT[nvals];
                    mask.extractTuples(I, J, X, &nvals);

                    grbTry(GrB_Matrix_new(&maskTmp, CInterface<ScalarT>::type(), mask.nrows(), mask.ncols()));
                    grbTry(CInterface<ScalarT>::matrixBuild(maskTmp, I, J, X, nvals, CInterface<ScalarT>::plus()));

                    delete[] I;
                    delete[] J;
                    delete[] X;
                } else {
                    maskTmp = _mask->getBlock(0);
                }
            }

            /* Create Matrix C form Matrix A */
            {
                GrB_Matrix CTmp;
                using ScalarT = typename MatrixInfo<CMatrixT>::scalar_t;
                using IndexT = typename MatrixInfo<CMatrixT>::index_t;

                grbTry(GrB_Matrix_new(&CTmp, CInterface<ScalarT>::type(), nrows, ncols));
                grbTry(GrB_Matrix_extract(CTmp, maskTmp, accum, ATmp, rowIndices, nrows, colIndices, ncols, desc));

                GrB_Index nvals;
                grbTry(GrB_Matrix_nvals(&nvals, CTmp));
                auto I = new IndexT[nvals];
                auto J = new IndexT[nvals];
                auto X = new ScalarT[nvals];

                grbTry(CInterface<int>::matrixExtractTuples(I, J, X, &nvals, CTmp));
                C.build(I, J, X, nvals, CInterface<ScalarT>::plus());

                delete[] I;
                delete[] J;
                delete[] X;
                GrB_Matrix_free(&CTmp);
            }

            if (_mA->getNumBlocks() > 1) { grbTry(GrB_Matrix_free(&ATmp)); }
            if constexpr (USE_MASK) { if (_mA->getNumBlocks() > 1) { grbTry(GrB_Matrix_free(&maskTmp)); }}
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

        std::tuple<GrB_BinaryOp, RowIndexT *, NRowIndicesT, ColIndexT *, NColIndicesT, GrB_Descriptor> _args;

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
                GrB_Matrix mask = GrB_NULL;
                if (USE_MASK) {
                    mask = *(GrB_Matrix *) STARPU_VARIABLE_GET_PTR(buffers[2]);
                }
                auto[accum, rowIndices, nrows, colIndices, ncols, desc] = *static_cast<decltype(_args) *>(clArgs);

                GrB_Index nrowsC, ncolsC, nrowsA, ncolsA;
                GrB_Matrix_nrows(&nrowsA, A);
                GrB_Matrix_ncols(&ncolsA, A);
                GrB_Matrix_nrows(&nrowsC, C);
                GrB_Matrix_ncols(&ncolsC, C);


                grbTry(GrB_Matrix_extract(C, mask, accum, A, rowIndices, nrows, colIndices, ncols, desc));
            };
            _codelet.nbuffers = 2 + (USE_MASK ? 1 : 0);
            _codelet.modes[0] = std::get<0>(_args) == GrB_NULL ? STARPU_W : STARPU_RW;; // C
            _codelet.modes[1] = STARPU_R; // A
            _codelet.modes[2] = STARPU_R; // mask
        }

    };

}

#endif //GRB_FUSION_OP_EXTRACT_MATRIX_H
