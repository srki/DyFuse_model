#ifndef GRB_FUSION_MATRIX_IMPL_H
#define GRB_FUSION_MATRIX_IMPL_H

#include <cassert>

#include <starpu.h>

#include <grb/context/Operation.h>
#include <grb/context/Context.h>
#include <grb/context/OpInternalClone.h>
#include <grb/grb-operations/OpCtor.h>
#include <grb/util/Util.h>
#include <grb/util/GraphBLASImpl.h>
#include "MatrixData.h"

namespace grb::detail {

    template<class ScalarT, class IndexT>
    class MatrixImpl {
        static_assert(std::is_same_v<IndexT, GrB_Index>, "IndexT must be same as GrB_Index.");

    public:
        MatrixImpl() = default;

        MatrixImpl(IndexT nrows, IndexT ncols, IndexT nrowsPerBlock, IndexT ncolsPerBlock)
                : _data{nrows * ncols > 0 ? std::make_shared<MatrixData<ScalarT, IndexT>>(nrows, ncols, nrowsPerBlock,
                                                                                          ncolsPerBlock) : nullptr},
                  _op{new OpCtor<true>{}} {
            Context::getDefaultContext().addOperation(_op);
        }

        MatrixImpl(const MatrixImpl &other) : _data{other._data}, _op{other._op}, _name{other._name} {};

        MatrixImpl(MatrixImpl &&other) noexcept {
            *this = std::move(other);
        }

        MatrixImpl &operator=(const MatrixImpl &other) {
            if (this == &other) { return *this; }
            _data = other._data;
            _op = other._op;
            _name = other._name;
            return *this;
        };

        MatrixImpl &operator=(MatrixImpl &&other) noexcept {
            if (this == &other) { return *this; }
            release();
            swap(*this, other);
            return *this;
        };

        ~MatrixImpl() {
            release();
        }

        friend void swap(MatrixImpl &lhs, MatrixImpl &rhs) {
            std::swap(lhs._data, rhs._data);
            std::swap(lhs._op, rhs._op);
            std::swap(lhs._name, rhs._name);
        }

        void release() {
            _op = nullptr;
            _data = nullptr;
            _name.clear();
        }

        /* region Public API */

        void resize(IndexT nrows, IndexT ncols) {
            std::cerr << "Not implemented: " << __PRETTY_FUNCTION__ << std::endl;
            exit(1);
        }

        void clear() {
            /* TODO: check whether _data was not modified or if there are pending ops */
            _data.reset(new MatrixData<ScalarT, IndexT>{_data->getNumRows(), _data->getNumCols(),
                                                        _data->getNumRowsPerBlock(), _data->getNumColsPerBlock()});
            _op->setName(_name);
        }

        [[nodiscard]] IndexT nrows() { return _data->getNumRows(); }

        [[nodiscard]] IndexT ncols() { return _data->getNumCols(); }

        [[nodiscard]] IndexT nvals() {
            _op->wait();
            return _data->nvals();
        }

        void build(const IndexT *rowIndices, const IndexT *colIndices, const ScalarT *values, GrB_Index n,
                   GrB_BinaryOp accum) {
            _data->build(rowIndices, colIndices, values, n, accum);
        }

        void setElement(ScalarT val, IndexT rowIndex, IndexT colIndex) {
            wait(); // TODO: maybe create an Operation class instead
            _data->setElement(val, rowIndex, colIndex);
        }

        void removeElement(IndexT rowIndex, IndexT colIndex) {
            wait(); // TODO: maybe create an Operation class instead
            _data->removeElement(rowIndex, colIndex);
        }

        ScalarT extractElement(IndexT rowIndex, IndexT colIndex) {
            wait();
            return _data->extractElement(rowIndex, colIndex);
        }

        void extractTuples(IndexT *rowIndices, IndexT *colIndices, ScalarT *values, IndexT *n) {
            wait();
            assert(*n >= this->nvals());
            _data->extractTuples(rowIndices, colIndices, values, n);
        }

        /* endregion */

        /* region Extended API */

        void wait() {
            _op->wait();
        }

        void print() {
            Context::getDefaultContext().wait();

            auto nvals = this->nvals();

            auto I = std::make_unique<IndexT[]>(nvals);
            auto J = std::make_unique<IndexT[]>(nvals);
            auto X = std::make_unique<ScalarT[]>(nvals);
            extractTuples(I.get(), J.get(), X.get(), &nvals);

            GrB_Matrix t;
            GrB_Matrix_new(&t, CInterface<ScalarT>::type(), nrows(), ncols());
            grbTry(CInterface<ScalarT>::matrixBuild(t, I.get(), J.get(), X.get(), nvals, CInterface<ScalarT>::plus()));

            GxB_Matrix_fprint(t, _name.c_str(), GxB_COMPLETE, stdout);
            GrB_Matrix_free(&t);
        }

        void printDense() {
            Context::getDefaultContext().wait();

            auto nvals = this->nvals();
            auto nrows = this->nrows();
            auto ncols = this->ncols();

            auto I = std::make_unique<IndexT[]>(nvals);
            auto J = std::make_unique<IndexT[]>(nvals);
            auto X = std::make_unique<ScalarT[]>(nvals);
            extractTuples(I.get(), J.get(), X.get(), &nvals);

            try {
                auto denseMatrix = std::make_unique<ScalarT[]>(nrows * ncols);
                for (IndexT i = 0; i < nvals; i++) {
                    denseMatrix[I[i] * ncols + J[i]] = X[i];
                }

                for (IndexT i = 0; i < nrows; i++) {
                    for (IndexT j = 0; j < ncols; j++) {
                        std::cout << denseMatrix[i * ncols + j] << " ";
                    }
                    std::cout << std::endl;
                }

            } catch (std::bad_alloc &) {
                fprintf(stdout, "Not enough memory for dense print.");
            }
        }

        void block(IndexT nrowsPerBlock, IndexT ncolsPerBlock) {
            wait();

            auto nrows = _data->getNumRows();
            auto ncols = _data->getNumCols();
            auto nvals = _data->nvals();
            auto I = std::make_unique<IndexT[]>(nvals);
            auto J = std::make_unique<IndexT[]>(nvals);
            auto X = std::make_unique<ScalarT[]>(nvals);

            _data->extractTuples(I.get(), J.get(), X.get(), &nvals);
            _data.reset(new MatrixData<ScalarT, IndexT>{nrows, ncols, nrowsPerBlock, ncolsPerBlock});
            _data->build(I.get(), J.get(), X.get(), nvals, CInterface<ScalarT>::plus());
        }

        /* endregion */

        /* region Internal API */

        [[nodiscard]] Operation *getOp() { return _op; }

        [[nodiscard]] std::shared_ptr<MatrixData<ScalarT, IndexT>> &getData() { return _data; }

        void setOp(Operation *op) { _op = op; }

        void setName(const std::string &name) {
            _name = name;
            if (_op->getType() == OperationType::MATRIX_OP) {
                _op->setName(name);
            }
        }

        friend bool operator==(const MatrixImpl &lhs, const MatrixImpl &rhs) {
            auto eq = lhs._data.get() == rhs._data.get();

            if (eq) { assert(lhs._op == rhs._op && lhs._name == rhs._name); }
            else { assert(lhs._op != rhs._op); }

            return eq;
        }

        void clone(bool fullClone) {
            _data.reset(_data->clone(fullClone));
            auto cloneOp = new OpInternalClone{_name + "-clone"};
            cloneOp->addInputDependency(_op, DependencyType::WRITE);
            _op = cloneOp;
        }

        /* endregion */

    private:
        std::shared_ptr<MatrixData<ScalarT, IndexT>> _data;
        Operation *_op{nullptr};
        std::string _name{};
    };

    template<>
    class MatrixImpl<void, void> {
    public:
        [[nodiscard]] static MatrixImpl &getInstance() {
            static MatrixImpl instance;
            return const_cast<MatrixImpl &>(instance);
        }

        [[nodiscard]] Operation *getOp() const { return nullptr; }

        [[nodiscard]] MatrixData<void, void> getData() const { return MatrixData<void, void>(); }

        friend bool operator==(const MatrixImpl &lhs, const MatrixImpl &rhs) = delete;

    };

    template<class ScalarT1, class IndexT1, class ScalarT2, class IndexT2>
    bool operator==(const MatrixImpl<ScalarT1, IndexT1> &lhs, const MatrixImpl<ScalarT2, IndexT2> &rhs) {
        static_assert(!std::is_same_v<ScalarT1, ScalarT2> || !std::is_same_v<IndexT1, IndexT2>);
        return false;
    }

}

#endif //GRB_FUSION_MATRIX_IMPL_H
