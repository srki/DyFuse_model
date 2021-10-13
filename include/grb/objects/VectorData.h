/* LICENSE PLACEHOLDER */

#include <iostream>

#ifndef GRB_FUSION_VECTOR_DATA_H
#define GRB_FUSION_VECTOR_DATA_H

#include <vector>
#include <numeric>
#include <algorithm>

#include <grb/util/Util.h>

namespace grb::detail {

    template<class ScalarT, class IndexT>
    class VectorData {
    public:
        VectorData(IndexT size, IndexT blockSize) : VectorData(size, blockSize, true) {}

        VectorData(const VectorData &other) : VectorData(other._size, other._blockSize, false) {
            for (IndexT i = 0; i < _numBlocks; i++) {
                grbTry(GrB_Vector_dup(&_blocks[i], other._blocks[i]));
                starpu_variable_data_register(&_handles[i], STARPU_MAIN_RAM, (uintptr_t) &_blocks[i],
                                              sizeof(GrB_Vector));
            }
        }

        ~VectorData() {
            for (IndexT i = 0; i < _numBlocks; i++) {
                grbTry(GrB_Vector_free(&_blocks[i]));
                starpu_data_unregister(_handles[i]);
            }

            delete[] _blocks;
            delete[] _handles;
        }

        /* region Public API */

        [[nodiscard]] IndexT nvals() {
            return std::accumulate(_blocks, _blocks + _numBlocks, IndexT(0),
                                   [](IndexT cnt, GrB_Vector v) {
                                       IndexT nvals = 0;
                                       GrB_Vector_nvals(&nvals, v);
                                       return cnt + nvals;
                                   });
        }

        void build(const IndexT *indices, const ScalarT *values, GrB_Index n, GrB_BinaryOp accum) {
            /* If there is only one block there is no need to block the tuples */
            if (_numBlocks == 1) {
                grbTry(CInterface<ScalarT>::vectorBuild(_blocks[0], indices, values, n, accum));
                return;
            }

            /* Calculate the number of elements per block */
            auto nvalsBlock = std::make_unique<IndexT[]>(_numBlocks);
            for (IndexT i = 0; i < n; i++) {
                nvalsBlock[indices[i] / _blockSize]++;
            }

            /* Create tuple arrays for blocks */
            auto IBlock = std::vector<std::unique_ptr<IndexT[]>>{};
            auto XBlock = std::vector<std::unique_ptr<ScalarT[]>>{};
            IBlock.reserve(_numBlocks);
            XBlock.reserve(_numBlocks);
            for (IndexT i = 0; i < _numBlocks; i++) {
                IBlock.emplace_back(std::make_unique<IndexT[]>(nvalsBlock[i]));
                XBlock.emplace_back(std::make_unique<ScalarT[]>(nvalsBlock[i]));
            }

            /* Fill the tuples arrays */
            std::fill(nvalsBlock.get(), nvalsBlock.get() + _numBlocks, 0);
            for (IndexT i = 0; i < n; i++) {
                auto blockIdx = indices[i] / _blockSize;
                auto idx = nvalsBlock[blockIdx]++;
                IBlock[blockIdx][idx] = indices[i] % _blockSize;
                XBlock[blockIdx][idx] = values[i];
            }

            /* Build blocks */
            for (IndexT i = 0; i < _numBlocks; i++) {
                grbTry(CInterface<ScalarT>::vectorBuild(_blocks[i], IBlock[i].get(), XBlock[i].get(),
                                                        nvalsBlock[i], CInterface<ScalarT>::plus()));
            }
        }

        void setElement(ScalarT val, IndexT index) {
            auto[i, ib] = getBlockIndex(index);
            grbTry(CInterface<ScalarT>::vectorSetElement(_blocks[i], val, ib));
        }

        void removeElement(IndexT index) {
            auto[i, ib] = getBlockIndex(index);
            grbTry(GrB_Vector_removeElement(_blocks[i], ib));
        }

        [[nodiscard]] ScalarT extractElement(IndexT index) {
            ScalarT val;
            auto[i, ib] = getBlockIndex(index);
            grbTry(CInterface<ScalarT>::vectorExtractElement(&val, _blocks[i], ib));
            return val;
        }

        void extractTuples(IndexT *indices, ScalarT *values, IndexT *n) {
            auto remainingSize = *n;
            IndexT nextPos = 0;
            for (IndexT i = 0; i < _numBlocks; i++) {
                IndexT size = *n - nextPos;
                grbTry(CInterface<ScalarT>::vectorExtractTuples(indices + nextPos, values + nextPos, &size,
                                                                _blocks[i]));
                shiftTupleIndices(i * _blockSize, indices + nextPos, size);
                nextPos += size;
            }
            *n = nextPos;
        }

        /* endregion */

        /* region implementation API */

        VectorData *clone(bool fullClone) {
            return fullClone ? new VectorData{*this} : new VectorData{_size, _blockSize};
        }

        [[nodiscard]] auto getBlockIndex(IndexT idx) const {
            auto i = idx / _blockSize;
//            auto ib = idx;
            auto ib = idx % _blockSize;
            assert(i < _numBlocks);

            return std::make_tuple(i, ib);
        }

        [[nodiscard]] IndexT getSize() const { return _size; }

        [[nodiscard]] IndexT getBlockSize() const { return _blockSize; }

        [[nodiscard]] IndexT getNumBlocks() const { return _numBlocks; }

        [[nodiscard]] starpu_data_handle_t &getHandle(IndexT idx) const {
            assert(idx < _numBlocks);
            return _handles[idx];
        }

        [[nodiscard]] GrB_Vector getBlock(IndexT blockIdx) {
            assert(blockIdx < _numBlocks);
            return _blocks[blockIdx];
        }

        /* endregion */

    protected:
        VectorData(IndexT size, IndexT blockSize, bool initBlocks)
                : _size(size),
                  _blockSize{blockSize != 0 ? blockSize : size}, _numBlocks{(size + _blockSize - 1) / _blockSize},
                  _blocks{new GrB_Vector[_numBlocks]},
                  _handles{new starpu_data_handle_t[_numBlocks]} {
            if (initBlocks) {
                for (IndexT i = 0; i < _numBlocks; i++) {
                    grbTry(GrB_Vector_new(&_blocks[i], CInterface<ScalarT>::type(),
                                          std::min(_blockSize, _size - i * _blockSize)));
                    starpu_variable_data_register(&_handles[i], STARPU_MAIN_RAM, (uintptr_t) &_blocks[i],
                                                  sizeof(GrB_Vector));
                }
            }
        }

    private:
        const IndexT _size;
        const IndexT _blockSize;
        const IndexT _numBlocks;
        GrB_Vector *const _blocks;
        starpu_data_handle_t *const _handles;

        void shiftTupleIndices(IndexT offset, IndexT *indices, size_t size) {
            std::transform(indices, indices + size, indices, std::bind1st(std::plus<IndexT>{}, offset));
        }
    };

    template<>
    class VectorData<void, void> {
    public:
        void reset() {}
    };

}

#endif //GRB_FUSION_VECTOR_DATA_H
