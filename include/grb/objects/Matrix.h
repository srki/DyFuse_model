/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_MATRIX_H
#define GRB_FUSION_MATRIX_H

#include <memory>
#include <grb/objects/MatrixImpl.h>

namespace grb {

    template<class ScalarT, class IndexT = uint64_t>
    class Matrix {
        using Impl = detail::MatrixImpl<ScalarT, IndexT>;

    public:
        Matrix(IndexT nrows, IndexT ncols, IndexT nrowsPerBlock = 0, IndexT ncolsPerBlock = 0)
                : _impl{nrows, ncols, nrowsPerBlock, ncolsPerBlock} {}

        Matrix() : Matrix(0, 0) {}

        Matrix(const Matrix &other) : _impl{other._impl} {}

        Matrix(Matrix &&other) noexcept {
            *this = std::move(other);
        }

        Matrix &operator=(const Matrix &other) {
            if (this == &other) { return *this; }
            _impl = other._impl;
            return *this;
        }

        Matrix &operator=(Matrix &&other) noexcept {
            if (this == &other) { return *this; }
            _impl = std::move(other._impl);
            return *this;
        }

        /* region public API */

        void resize(IndexT nrows, IndexT ncols) {
            _impl.resize(nrows, ncols);
        }

        void clear() {
            _impl.clear();
        }

        [[nodiscard]] IndexT nrows() {
            return _impl.nrows();
        }

        [[nodiscard]] IndexT ncols() {
            return _impl.ncols();
        }

        [[nodiscard]] IndexT nvals() {
            return _impl.nvals();
        }

        void build(const IndexT *rowIndices, const IndexT *colIndices, const ScalarT *values, GrB_Index n,
                   GrB_BinaryOp accum) {
            _impl.build(rowIndices, colIndices, values, n, accum);
        }

        void setElement(ScalarT val, IndexT row_index, IndexT col_index) {
            _impl.setElement(val, row_index, col_index);
        }

        void remoteElement(IndexT row_index, IndexT col_index) {
            _impl.removeElement(row_index, col_index);
        }

        [[nodiscard]] ScalarT extractElement(IndexT row_index, IndexT col_index) {
            return _impl.extractElement(row_index, col_index);
        }

        void extractTuples(IndexT *rowIndices, IndexT *colIndices, ScalarT *values, IndexT *n) {
            _impl.extractTuples(rowIndices, colIndices, values, n);
        }

        /* endregion */

        /* region API extensions */

        void wait() {
            _impl.wait();
        }

        void print() {
            _impl.print();
        }

        void printDense() {
            _impl.printDense();
        }

        auto &getImpl() {
            return _impl;
        }

        const auto &getImpl() const {
            return _impl;
        }

        void block(IndexT nrowsPerBlock, IndexT ncolsPerBlock) {
            _impl.block(nrowsPerBlock, ncolsPerBlock);
        }

        /* endregion */

    private:
        Impl _impl;

    public:
        Matrix(IndexT nrows, IndexT ncols, const std::string &name, IndexT nrowsPerBlock = 0, IndexT ncolsPerBlock = 0)
                : Matrix(nrows, ncols, nrowsPerBlock, ncolsPerBlock) {
            setName(name);
        }

        void setName(const std::string &name) {
            _impl.setName(name);
        }

    };

}

#endif //GRB_FUSION_MATRIX_H
