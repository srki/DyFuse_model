/* LICENSE PLACEHOLDER */


#ifndef GRB_FUSION_MATRIX_DATA_H
#define GRB_FUSION_MATRIX_DATA_H

#include <vector>
#include <numeric>
#include <algorithm>

#include <grb/util/Util.h>


namespace grb::detail {

    template<class ScalarT, class IndexT>
    class MatrixData {
    public:
        MatrixData(IndexT nrows, IndexT ncols, IndexT nrowsPerBlock, IndexT ncolsPerBlock)
                : MatrixData(nrows, ncols, nrowsPerBlock, ncolsPerBlock, true) {}

        MatrixData(const MatrixData &other)
                : MatrixData(other._nrows, other._ncols, other._nrowsPerBlock, other._ncolsPerBlock, false) {
            for (IndexT i = 0; i < _numBlocks; i++) {
                grbTry(GrB_Matrix_dup(&_blocks[i], other._blocks[i]));
                starpu_variable_data_register(&_handles[i], STARPU_MAIN_RAM, (uintptr_t) &_blocks[i],
                                              sizeof(GrB_Matrix));
            }
        }

        ~MatrixData() {
            for (IndexT blockIdx = 0; blockIdx < _numBlocks; blockIdx++) {
                grbTry(GrB_Matrix_free(&_blocks[blockIdx]));
                starpu_data_unregister(_handles[blockIdx]);
            }

            delete[] _blocks;
            delete[] _handles;
        }

        /* region Public API */

        [[nodiscard]] IndexT nvals() {
            return std::accumulate(_blocks, _blocks + _numBlocks, IndexT(0),
                                   [](IndexT cnt, GrB_Matrix v) {
                                       IndexT nvals = 0;
                                       GrB_Matrix_nvals(&nvals, v);
                                       return cnt + nvals;
                                   });
        }

        void build(const IndexT *rowIndices, const IndexT *colIndices, const ScalarT *values, GrB_Index n,
                   GrB_BinaryOp accum) {
            /* If there is only one block there is no need to block the tuples */
            if (_numBlocks == 1) {
                grbTry(CInterface<ScalarT>::matrixBuild(_blocks[0], rowIndices, colIndices, values, n, accum));
                return;
            }

            /* Calculate the number of elements per block */
            auto nvalsBlock = std::make_unique<IndexT[]>(_numBlocks);
            for (IndexT i = 0; i < n; i++) {
                auto blockIdx = (rowIndices[i]) / _nrowsPerBlock * _ncolsBlocked + (colIndices[i]) / _ncolsPerBlock;
                nvalsBlock[blockIdx]++;
            }

            /* Create tuple arrays for blocks */
            auto IBlock = std::vector<std::unique_ptr<IndexT[]>>{};
            auto JBlock = std::vector<std::unique_ptr<IndexT[]>>{};
            auto XBlock = std::vector<std::unique_ptr<ScalarT[]>>{};
            IBlock.reserve(_numBlocks);
            JBlock.reserve(_numBlocks);
            XBlock.reserve(_numBlocks);
            for (IndexT i = 0; i < _numBlocks; i++) {
                IBlock.emplace_back(std::make_unique<IndexT[]>(nvalsBlock[i]));
                JBlock.emplace_back(std::make_unique<IndexT[]>(nvalsBlock[i]));
                XBlock.emplace_back(std::make_unique<ScalarT[]>(nvalsBlock[i]));
            }

            /* Fill the tuples arrays */
            std::fill(nvalsBlock.get(), nvalsBlock.get() + _numBlocks, 0);
            for (IndexT i = 0; i < n; i++) {
                auto[r, c, rb, cb, blockIdx] = getBlockIndex(rowIndices[i], colIndices[i]);
                auto idx = nvalsBlock[blockIdx]++;
                IBlock[blockIdx][idx] = rb;
                JBlock[blockIdx][idx] = cb;
                XBlock[blockIdx][idx] = values[i];
            }

            /* Build blocks */
            for (IndexT i = 0; i < _numBlocks; i++) {
                grbTry(CInterface<ScalarT>::matrixBuild(_blocks[i], IBlock[i].get(), JBlock[i].get(), XBlock[i].get(),
                                                        nvalsBlock[i], CInterface<ScalarT>::plus()));
            }
        }

        void setElement(ScalarT val, IndexT rowIndex, IndexT colIndex) {
            auto[r, c, rb, cb, blockIdx] = getBlockIndex(rowIndex, colIndex);
            grbTry(CInterface<ScalarT>::matrixSetElement(_blocks[blockIdx], val, rb, cb));
        }

        void removeElement(IndexT rowIndex, IndexT colIndex) {
            auto[r, c, rb, cb, blockIdx] = getBlockIndex(rowIndex, colIndex);
            grbTry(GrB_Matrix_removeElement(_blocks[blockIdx], rb, cb));
        }

        [[nodiscard]] ScalarT extractElement(IndexT rowIndex, IndexT colIndex) {
            ScalarT val;
            auto[r, c, rb, cb, blockIdx] = getBlockIndex(rowIndex, colIndex);
            grbTry(CInterface<ScalarT>::matrixExtractElement(&val, _blocks[blockIdx], rb, cb));
            return val;
        }

        void extractTuples(IndexT *rowIndices, IndexT *colIndices, ScalarT *values, IndexT *n) {
            auto remainingSize = *n;
            IndexT nextPos = 0;
            for (IndexT i = 0; i < _nrowsBlocked; i++) {
                for (IndexT j = 0; j < _ncolsBlocked; j++) {
                    auto size = *n - nextPos;
                    auto idx = i * _ncolsBlocked + j;

                    grbTry(CInterface<ScalarT>::matrixExtractTuples(rowIndices + nextPos, colIndices + nextPos,
                                                                    values + nextPos, &size, _blocks[idx]));
                    shiftTupleIndices(i * _nrowsPerBlock, rowIndices + nextPos, size);
                    shiftTupleIndices(j * _ncolsPerBlock, colIndices + nextPos, size);
                    nextPos += size;
                }
            }
            *n = nextPos;
        }

        /* endregion */

        /* region Implementation API */

        MatrixData *clone(bool fullClone) {
            return fullClone ? new MatrixData{*this} : new MatrixData{_nrows, _ncols, _nrowsPerBlock, _ncolsPerBlock};
        }

        MatrixData *emptyCone() {
            return new MatrixData{_nrows, _ncols, _nrowsPerBlock, _ncolsPerBlock};
        }

        [[nodiscard]] auto getBlockIndex(IndexT rowIndex, IndexT colIndex) const {
            auto i = rowIndex / _nrowsPerBlock;
            auto j = colIndex / _ncolsPerBlock;
            auto ib = rowIndex % _nrowsPerBlock;
            auto jb = colIndex % _ncolsPerBlock;
            auto blockIdx = i * _ncolsBlocked + j;

            assert(i < _nrowsBlocked);
            assert(j < _ncolsBlocked);

            return std::make_tuple(i, j, ib, jb, blockIdx);
        }

        [[nodiscard]] IndexT getNumRows() const { return _nrows; }

        [[nodiscard]] IndexT getNumCols() const { return _ncols; }

        [[nodiscard]] IndexT getNumRowsPerBlock() const { return _nrowsPerBlock; }

        [[nodiscard]] IndexT getNumColsPerBlock() const { return _ncolsPerBlock; }

        [[nodiscard]] IndexT getNumRowsBlocked() const { return _nrowsBlocked; }

        [[nodiscard]] IndexT getNumColsBlocked() const { return _ncolsBlocked; }

        [[nodiscard]] IndexT getNumBlocks() const { return _numBlocks; }

        [[nodiscard]] starpu_data_handle_t &getHandle(IndexT rowIdx, IndexT colIdx) {
            assert(rowIdx < _nrowsBlocked);
            assert(colIdx < _ncolsBlocked);
            return _handles[rowIdx * _ncolsBlocked + colIdx];
        }

        [[nodiscard]] starpu_data_handle_t &getHandle(IndexT blockIdx) {
            assert(blockIdx < _numBlocks);
            return _handles[blockIdx];
        }

        [[nodiscard]] GrB_Matrix getBlock(IndexT rowIdx, IndexT colIdx) {
            assert(rowIdx < _nrowsBlocked);
            assert(colIdx < _ncolsBlocked);
            return _blocks[rowIdx * _ncolsBlocked + colIdx];
        }

        [[nodiscard]] GrB_Matrix getBlock(IndexT blockIdx) {
            assert(blockIdx < _numBlocks);
            return _blocks[blockIdx];
        }

        /* endregion */

    private:
        const IndexT _nrows;
        const IndexT _ncols;
        const IndexT _nrowsPerBlock;
        const IndexT _ncolsPerBlock;
        const IndexT _nrowsBlocked;
        const IndexT _ncolsBlocked;
        const IndexT _numBlocks;
        GrB_Matrix *const _blocks;
        starpu_data_handle_t *const _handles;

        MatrixData(IndexT nrows, IndexT ncols, IndexT nrowsPerBlock, IndexT ncolsPerBlock, bool initBlocks)
                : _nrows{nrows}, _ncols{ncols},
                  _nrowsPerBlock{nrowsPerBlock > 0 ? nrowsPerBlock : nrows},
                  _ncolsPerBlock{ncolsPerBlock > 0 ? ncolsPerBlock : ncols},
                  _nrowsBlocked{(_nrows + _nrowsPerBlock - 1) / _nrowsPerBlock},
                  _ncolsBlocked{(_ncols + _ncolsPerBlock - 1) / _ncolsPerBlock},
                  _numBlocks{_nrowsBlocked * _ncolsBlocked},
                  _blocks{new GrB_Matrix[_nrowsBlocked * _ncolsBlocked]},
                  _handles{new starpu_data_handle_t[_nrowsBlocked * _ncolsBlocked]} {
            if (initBlocks) {
                for (IndexT i = 0; i < _nrowsBlocked; i++) {
                    for (IndexT j = 0; j < _ncolsBlocked; j++) {
                        auto blockIdx = i * _ncolsBlocked + j;
                        grbTry(GrB_Matrix_new(&_blocks[blockIdx], CInterface<ScalarT>::type(),
                                              std::min(_nrowsPerBlock, _nrows - i * _nrowsPerBlock),
                                              std::min(_ncolsPerBlock, _ncols - j * _ncolsPerBlock)));
                        starpu_variable_data_register(&_handles[blockIdx], STARPU_MAIN_RAM,
                                                      (uintptr_t) &_blocks[blockIdx],
                                                      sizeof(GrB_Matrix));
                    }
                }
            }
        }

        void shiftTupleIndices(IndexT offset, IndexT *indices, size_t size) {
            std::transform(indices, indices + size, indices, std::bind1st(std::plus<IndexT>{}, offset));
        }
    };

    template<>
    class MatrixData<void, void> {
    public:
        void reset() {}
    };

}

#endif //GRB_FUSION_MATRIX_DATA_H
