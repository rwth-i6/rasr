/** Copyright 2020 RWTH Aachen University. All rights reserved.
 *
 *  Licensed under the RWTH ASR License (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.hltpr.rwth-aachen.de/rwth-asr/rwth-asr-license.html
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
#ifndef TREEWALKERS_HH
#define TREEWALKERS_HH

#include <unordered_set>

#include "TreeStructure.hh"

namespace Search {
/**
 * Inherit this, and then in the given base, implement nodeVisited(TreeNodeIndex, token),
 * visitExit(uint exit, token), and visitNode(TreeNodeIndex, token)
 */
template<class Token, class Base>
class SubTreeWalker : public Base {
public:
    SubTreeWalker(HMMStateNetwork& _tree)
            : tree(_tree) {
    }

    ///Visits the nodes and all its followers, in the correct order
    void visit(StateId node, Token token) {
        bool hadToken = token;

        if (token)
            token = this->visitNode(node, token);

        if (token) {
            for (HMMStateNetwork::SuccessorIterator target = tree.successors(node); target; ++target) {
                if (target.isLabel()) {
                    this->visitExit(target.label(), token);
                }
                else {
                    visit(*target, token);
                }
            }
        }

        if (hadToken)
            this->nodeVisited(node, token);
    }

    HMMStateNetwork& tree;
};

struct CountSizeTreeWalkerBackend {
    CountSizeTreeWalkerBackend()
            : totalVisited(0), stopAtVisited(false), visitedFinalOutputs(0) {
    }

    void nodeVisited(StateId /*node*/, int /*token*/) {
    }

    int visitNode(StateId node, int token) {
        std::unordered_set<Search::StateId>::iterator it = visited.find(node);
        if (it == visited.end()) {
            visited.insert(node);
        }
        else {
            if (stopAtVisited)
                return 0;
        }

        ++totalVisited;

        return token + 1;
    }

    void visitExit(u32 /*exit*/, int /*token*/) {
        ++visitedFinalOutputs;
    }

    std::unordered_set<StateId>   visited;
    u32                           totalVisited;
    bool                          stopAtVisited;
    u32                           visitedFinalOutputs;
};

typedef Search::SubTreeWalker<int, CountSizeTreeWalkerBackend> CountSizeTreeWalker;
}  // namespace Search

#endif  // TREEWALKERS_HH
