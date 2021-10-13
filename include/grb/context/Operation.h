/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_OPERATION_H
#define GRB_FUSION_OPERATION_H

#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <typeinfo>

#include <starpu.h>

namespace grb::detail {
    enum class DependencyType : uint8_t {
        NONE = 0,
        READ,
        WRITE,
        REUSE,
        WAIT
    };

    enum class OperationType : uint8_t {
        NONE = 0,
        INTERNAL_CLONE,
        BUILD,
        VECTOR_OP,
        MATRIX_OP,
        MXM,
        MXV,
        VXM,
        eWISE_MULTIPLICATION,
        eWISE_ADD,
        EXTRACT_VECTOR,
        EXTRACT_MATRIX,
        EXTRACT_MATRIX_COLUMN,
        ASSIGN_VECTOR_VALUE,
        ASSIGN_MATRIX_VALUE,
        APPLY_VECTOR,
        APPLY_VECTOR_BIN_OP,
        APPLY_MATRIX,
        APPLY_MATRIX_BIN_OP,
        REDUCE_MATRIX_VECTOR,
        REDUCE_VECTOR_SCALAR,
        REDUCE_MATRIX_SCALAR,
        SELECT_MATRIX,
        SELECT_VECTOR,
        WAIT
    };

    class Operation {
    public:
        explicit Operation(OperationType opType) : _opType(opType) {}

        Operation(OperationType opType, std::string name) : _opType(opType), _name(std::move(name)) {}

        virtual ~Operation() = default;


        virtual void release() = 0;

        bool submit() {
            if (_submitted) { return false; }
            _submitted = true;

            for (auto &inputDependency : _inputDependencies) {
                inputDependency.first->submit();
            }
//            std::cout << "S " << getName() << std::endl;

            for (auto task : _tasks) {
                task->detach = 0;
                STARPU_CHECK_RETURN_VALUE(starpu_task_submit(task), __FILE__);
            }

            return true;
        }

        virtual bool wait() {
            if (_completed) { return false; }

            submit();

            for (auto task : _tasks) {
                /* Name ptr is saved because the task will be deallocated after starpu_task_wait */
                auto name = task->name;
                STARPU_CHECK_RETURN_VALUE(starpu_task_wait(task), "starpu_task_wait");
                delete[] name;
            }

            release();
            _completed = true;

            return true;
        }


        void addInputDependency(Operation *dependency, DependencyType type) {
            if (dependency == nullptr) { return; }
            _inputDependencies.emplace_back(dependency, type);
            dependency->_outputDependencies.emplace_back(this, type);
        }

        [[nodiscard]] const auto &getInputDependencies() const {
            return _inputDependencies;
        }

        [[nodiscard]] const auto &getOutputDependencies() const {
            return _outputDependencies;
        }

        void clearDependencies() {
            clearInputDependencies();
            clearOutputDependencies();
        }

        void clearOutputDependencies() {
            _outputDependencies.clear();
        }

        void clearInputDependencies() {
            _inputDependencies.clear();
        }

        [[nodiscard]] std::string getName() const {
            return _name.empty() ? opTypeToStr(_opType) : _name;
        }

        void setName(const std::string &name) {
            _name = name;
        }

        [[nodiscard]] OperationType getType() const {
            return _opType;
        }

    protected:
        void addTask(starpu_task *task, const std::string &taskName) {
            task->name = strcpy(new char[taskName.length() + 1], taskName.c_str());
            addTask(task);
        }

        void addTask(starpu_task *task) {
            _tasks.emplace_back(task);
        }

    private:
        bool _completed{false};
        bool _submitted{false};
        std::vector<starpu_task *> _tasks;

        std::vector<std::pair<Operation *, DependencyType>> _inputDependencies;
        std::vector<std::pair<Operation *, DependencyType>> _outputDependencies;

        std::string _name;
        const OperationType _opType;

        static std::string opTypeToStr(OperationType type) {
            switch (type) {
                case OperationType::NONE:
                    return "None";
                case OperationType::INTERNAL_CLONE:
                    return "InternalClone";
                case OperationType::BUILD:
                    return "Build";
                case OperationType::VECTOR_OP:
                    return "VectorOp";
                case OperationType::MATRIX_OP:
                    return "MatrixOp";
                case OperationType::MXM:
                    return "mxm";
                case OperationType::MXV:
                    return "mxv";
                case OperationType::VXM:
                    return "vxm";
                case OperationType::eWISE_MULTIPLICATION:
                    return "eWiseMult";
                case OperationType::eWISE_ADD:
                    return "eWiseAdd";
                case OperationType::EXTRACT_VECTOR:
                    return "extractVector";
                case OperationType::EXTRACT_MATRIX:
                    return "extractMatrix";
                case OperationType::EXTRACT_MATRIX_COLUMN:
                    return "extractMatrixColumn";
                case OperationType::ASSIGN_VECTOR_VALUE:
                    return "assignVectorValue";
                case OperationType::ASSIGN_MATRIX_VALUE:
                    return "assignMatrixValue";
                case OperationType::APPLY_VECTOR:
                    return "applyVector";
                case OperationType::APPLY_VECTOR_BIN_OP:
                    return "applyVectorBinOp";
                case OperationType::APPLY_MATRIX:
                    return "applyMatrix";
                case OperationType::APPLY_MATRIX_BIN_OP:
                    return "applyMatrixBinOp";
                case OperationType::REDUCE_MATRIX_VECTOR:
                    return "reduceMatrixVector";
                case OperationType::REDUCE_VECTOR_SCALAR:
                    return "reduceVectorScalar";
                case OperationType::REDUCE_MATRIX_SCALAR:
                    return "reduceMatrixScalar";
                case OperationType::SELECT_VECTOR:
                    return "selectVector";
                case OperationType::SELECT_MATRIX:
                    return "selectMatrix";
                case OperationType::WAIT:
                    return "wait";
                default:
                    return "error";
            }
        }

    public:
        uint32_t _groupId{};

        [[nodiscard]] uint32_t getGroupId() const {
            return _groupId;
        }

        void setGroupId(uint32_t groupId) {
            _groupId = groupId;
        }
    };

}

#endif //GRB_FUSION_OPERATION_H
