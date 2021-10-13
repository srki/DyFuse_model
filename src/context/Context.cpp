/* LICENSE PLACEHOLDER */

#include "grb/context/Context.h"
#include <grb/context/Operation.h>
#include <starpu.h>

namespace grb::detail {
    Context &Context::getDefaultContext() {
        static Context ctx;
        return ctx;
    }

    const std::vector<Operation *> &Context::getOperations() {
        return _operations;
    }

    Context &Context::addOperation(Operation *op) {
        _operations.push_back(op);
        op->setGroupId(_currentGroupId);
//        op->submit();
//            _pendingOperations.push_back(op);
//            op->addInputDependency(_lastWait, DependencyType::WAIT);
        return *this;
    }

    Context &Context::addOperationAndWait(Operation *op) {
        addOperation(op);
        op->wait();
        return *this;
    }

    Context &Context::wait() {
        _currentGroupId++;
//            _lastWait = new OpWait{_pendingOperations};
//            _operations.push_back(_lastWait);
//            _pendingOperations.clear();
        for (auto op : _operations) {
            op->submit();
        }

        starpu_task_wait_for_all();

        for (auto op : _operations) {
            op->wait();
        }

        return *this;
    }

    void Context::releaseOps() {
        for (auto op : _operations) {
            delete op;
        }
    }

}
