/* LICENSE PLACEHOLDER */

#ifndef GRB_FUSION_GRAPH_VIZ_UTIL_H
#define GRB_FUSION_GRAPH_VIZ_UTIL_H

#include <grb/context/Context.h>
#include <grb/context/Operation.h>
#include <cassert>
#include <iostream>
#include <fstream>

namespace grb::util {

    inline void printGraphViz(std::ostream &os) {
        os << "digraph G {" << std::endl;

        for (auto op : detail::Context::getDefaultContext().getOperations()) {
            if (op->getOutputDependencies().empty() && op->getInputDependencies().empty()) { continue; }
            os << "\tsubgraph cluster" << op->getGroupId()
               << " { style=filled; color=lightgrey; " << size_t(op)
               << " [label=\"" << op->getName()
               << "\"]; }" << std::endl;
//            os << "\t" << size_t(op) << " [label=\"" << op->getName() << "\"];" << std::endl;
        }

        for (auto op : detail::Context::getDefaultContext().getOperations()) {
            auto inDeps = op->getInputDependencies();
            if (inDeps.empty()) { continue; }

            for (const auto &inDep : inDeps) {
                os << "\t" << size_t(inDep.first) << " -> " << size_t(op) << "[";

                switch (inDep.second) {
                    case detail::DependencyType::READ:
                        os << "color=\"blue\"";
                        break;
                    case detail::DependencyType::WRITE:
                        os << "color=\"red\"";
                        break;
                    case detail::DependencyType::REUSE:
                        os << "style=dotted";
                        break;
                    case detail::DependencyType::WAIT:
                        os << "color=\"black\"";
                        break;
                    default:
                        std::cerr << __FILE__ << ": " << __LINE__ << std::endl;
                        assert(false);
                }

                os << "]" << std::endl;
            }
        }

        os << "}" << std::endl;
    }

    inline void printGraphViz() {
        printGraphViz(std::cout);
    }

    inline void printGraphVizEnv() {
        std::string path;
        if (auto p = std::getenv("GRB_GRAPHVIZ")) {
            path = std::string{p};
        } else {
            return;
        }

        std::ofstream f{path};
        printGraphViz(f);
    }

}

#endif //GRB_FUSION_GRAPH_VIZ_UTIL_H
