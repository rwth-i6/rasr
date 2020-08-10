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
#include "TreeStructure.hh"
#include "TreeWalker.hh"

namespace Search {
HMMStateNetwork::HMMStateNetwork()
        : subTreeManager_(subTreeListBatches_, states_), edgeTargetManager_(edgeTargetBatches_, states_) {
    //The zero index is reserved as "invalid", so push one dummy item into all arrays
    trees_.push_back(Tree());
    states_.push_back(HMMState());
    subTreeListBatches_.push_back(0);
    edgeTargetBatches_.push_back(0);
    edgeTargetLists_.push_back(0);
    verify(sizeof(HMMState) % sizeof(u32) == 0);
}

TreeIndex HMMStateNetwork::allocateTree() {
    trees_.push_back(Tree());
    return TreeIndex(trees_.size() - 1);
}

StateId HMMStateNetwork::allocateTreeNode(TreeIndex parent) {
    verify(parent != EmptyTreeIndex);
    StateId ret = subTreeManager_.appendOne(tree(parent).nodes, HMMState());
    return ret;
}

void HMMStateNetwork::clearOutputEdges(StateId node) {
    state(node).successors = InvalidBatchId;
}

HMMStateNetwork::ChangePlan HMMStateNetwork::change(StateId node) {
    return ChangePlan(*this, node);
}

void HMMStateNetwork::ChangePlan::apply() {
    if (remove.empty() && add.empty())
        return;

    SuccessorBatchId& list = structure->state(node).successors;

    std::set<StateId> targets;
    std::set<u32>     outputs;

    for (HMMStateNetwork::SuccessorIterator targetIt = structure->batchSuccessors(list); targetIt; ++targetIt) {
        if (remove.find(*targetIt) != remove.end())
            continue;
        if (targetIt.isLabel()) {
            outputs.insert(targetIt.label());
        }
        else {
            targets.insert(*targetIt);
        }
    }

    for (std::set<int>::iterator it = add.begin(); it != add.end(); ++it) {
        if (IS_LABEL(*it))
            outputs.insert(LABEL_FROM_ID(*it));
        else
            targets.insert(*it);
    }

    structure->clearOutputEdges(node);

    for (std::set<StateId>::const_iterator it = targets.begin(); it != targets.end(); ++it)
        structure->addNodeToEdge(list, *it);

    for (std::set<u32>::iterator it = outputs.begin(); it != outputs.end(); ++it) {
        verify(not IS_LABEL(*it));
        structure->addOutputToEdge(list, ID_FROM_LABEL(*it));
    }
}

void HMMStateNetwork::removeTargetFromNode(StateId node, StateId remove) {
    ChangePlan plan = change(node);
    plan.removeSuccessor(remove);
    plan.apply();
}

void HMMStateNetwork::removeOutputFromNode(StateId node, u32 remove) {
    ChangePlan plan = change(node);
    plan.removeSuccessorLabel(remove);
    plan.apply();
}

void HMMStateNetwork::addNodeToEdge(SuccessorBatchId& list, StateId target) {
    addTargetToEdge(list, target);
}

void HMMStateNetwork::addTargetToEdge(SuccessorBatchId& batch, u32 target) {
    verify(target >= 0);

    //Special case for only one item
    edgeTargetManager_.appendToBatch(batch, target, target + 1);

    verify(batch != InvalidBatchId);
}

void HMMStateNetwork::addOutputToEdge(SuccessorBatchId& list, u32 outputIndex) {
    addTargetToEdge(list, ID_FROM_LABEL(outputIndex));
}

u32 HMMStateNetwork::treeCount() const {
    return trees_.size();
}

u32 HMMStateNetwork::stateCount() const {
    return states_.size();
}

bool HMMStateNetwork::write(Core::MappedArchiveWriter writer) {
    u32 version = DiskFormatVersion;
    writer << version << subTreeListBatches_ << states_ << edgeTargetLists_ << edgeTargetBatches_ << trees_;
    return writer.good();
}

bool HMMStateNetwork::read(Core::MappedArchiveReader reader) {
    u32 version;
    reader >> version;
    if (version != DiskFormatVersion)
        return false;

    reader >> subTreeListBatches_ >> states_ >> edgeTargetLists_ >> edgeTargetBatches_ >> trees_;

    return reader.good();
}

u32 HMMStateNetwork::countReachableEnds(std::vector<u32>& counts, StateId node) const {
    if (counts[node] == Core::Type<u32>::max) {
        counts[node] = 0;

        for (HMMStateNetwork::SuccessorIterator targetIt = successors(node); targetIt; ++targetIt) {
            if (targetIt.isLabel())
                ++counts[node];
            else
                counts[node] += countReachableEnds(counts, *targetIt);
        }
    }

    return counts[node];
}

HMMStateNetwork::CleanupResult HMMStateNetwork::cleanup(std::list<Search::StateId> startNodes, Search::TreeIndex masterTree, bool clearDeadEnds, bool onlyBatches) {
    if (clearDeadEnds && !onlyBatches) {
        u32              deadEndNodes = 0;
        std::vector<u32> reachableEnds(states_.size(), Core::Type<u32>::max);
        for (StateId node = 1; node < states_.size(); ++node) {
            countReachableEnds(reachableEnds, node);
            if (reachableEnds[node] == 0) {
                ++deadEndNodes;
                clearOutputEdges(node);
            }
        }

        // Eventually remove dead nodes from predecessor batches
        // Thereby they become unreachable and will be removed
        u32 cleared = 0;
        for (StateId node = 1; node < states_.size(); ++node) {
            bool removed = false;
            for (HMMStateNetwork::SuccessorIterator targetIt = successors(node); targetIt; ++targetIt) {
                if (not targetIt.isLabel()) {
                    if (not successors(*targetIt)) {
                        ++cleared;
                        removeTargetFromNode(node, *targetIt);
                        removed = true;
                        break;
                    }
                }
            }
            if (removed)
                --node;  //Process the same node again, as more targets may get removed
        }

        Core::Application::us()->log() << "cleared " << cleared << " dead-end nodes";
    }

    Core::Application::us()->log() << "total nodes before cleanup: " << states_.size();

    CleanupResult       ret;
    CountSizeTreeWalker counter(*this);
    if (onlyBatches) {
        for (TreeIndex tree = 1; tree < trees_.size(); ++tree)
            counter.visitedTrees.insert(tree);
        for (StateId node = 1; node < states_.size(); ++node)
            counter.visited.insert(node);
    }
    else {
        counter.visitedTrees.insert(masterTree);
        counter.stopAtVisited = true;

        ///Mark all reachable trees and nodes
        Core::Application::us()->log() << "calculating reachable nodes and trees";
        for (std::list<StateId>::const_iterator it = startNodes.begin(); it != startNodes.end(); ++it)
            counter.visit(*it, 1);
    }

    {
        std::vector<Tree>             newTrees;
        std::vector<StateId>          newSubTreeListBatches;
        std::vector<HMMState>         newNodes;
        std::vector<SuccessorBatchId> newEdgeTargetLists;

        //Must be created before adding the initial items, since it clears the lists
        Tools::BatchManager<SubTreeListId, StateId, HMMState, true, InvalidBatchId> newSubTreeManager(newSubTreeListBatches, newNodes);

        newTrees.push_back(Tree());
        newEdgeTargetLists.push_back(0);
        newSubTreeListBatches.push_back(0);
        newNodes.push_back(HMMState());

        std::vector<u32> orderBehind(states_.size(), 0);
        std::vector<u32> follow(states_.size(), 0);

        std::vector<std::vector<StateId>> orderedPerTree;
        orderedPerTree.push_back(std::vector<StateId>());

        for (u32 tree = 1; tree < trees_.size(); ++tree) {
            /// @todo Build a topology and order the nodes in a stable way based on that
            //Build the order so that the second-order batches are continuous
            Tools::BatchManager<SubTreeListId, StateId, HMMState, true, Search::InvalidBatchId>::Iterator it = subTreeManager_.getIterator(trees_[tree].nodes);
            for (; it; ++it) {
                StateId node = *it;
                if (counter.visited.count(node) == 0)
                    continue;
                // 2nd order
                u32 previousSkipTarget = 0;

                // first order
                u32 previousTarget = 0;

                for (HMMStateNetwork::SuccessorIterator targetIt = successors(node); targetIt; ++targetIt) {
                    if (targetIt.isLabel())
                        break;

                    StateId target = *targetIt;

                    if (!orderBehind[target])
                        orderBehind[target] = previousTarget;
                    if (!follow[previousTarget])
                        follow[previousTarget] = target;
                    previousTarget = target;
                    verify(target < states_.size());

                    for (HMMStateNetwork::SuccessorIterator skipTargetIt = successors(target); skipTargetIt; ++skipTargetIt) {
                        if (skipTargetIt.isLabel())
                            break;
                        StateId skipTarget         = *skipTargetIt;
                        orderBehind[skipTarget]    = previousSkipTarget;
                        follow[previousSkipTarget] = skipTarget;
                        previousSkipTarget         = skipTarget;
                    }
                }
            }
            std::vector<StateId>        ordered;
            std::unordered_set<StateId> had;

            {
                Tools::BatchManager<SubTreeListId, StateId, HMMState, true, Search::InvalidBatchId>::Iterator it = subTreeManager_.getIterator(trees_[tree].nodes);
                for (; it; ++it) {
                    StateId current = *it;

                    if (counter.visited.count(current) == 0)
                        continue;

                    if (onlyBatches) {
                        ordered.push_back(current);
                    }
                    else {
                        while (current) {
                            if (had.count(current))
                                break;

                            ordered.push_back(current);
                            had.insert(current);
                            current = follow[current];
                        }
                    }
                }
            }

            orderedPerTree.push_back(ordered);
        }

        for (u32 tree = 1; tree < trees_.size(); ++tree) {
            if (counter.visitedTrees.find(tree) != counter.visitedTrees.end()) {
                //Build the order so that the second-order batches are continuous
                const std::vector<StateId>& ordered(orderedPerTree[tree]);

                //Transfer network into new list
                ret.treeMap.insert(std::make_pair(tree, newTrees.size()));
                newTrees.push_back(trees_[tree]);
                newTrees.back().nodes = InvalidBatchId;

                //Transfer nodes into new batches

                for (u32 idx = 0; idx < ordered.size(); ++idx) {
                    StateId node = ordered[idx];
                    if (counter.visited.find(node) != counter.visited.end()) {
                        verify(newNodes.size() > 0);
                        StateId newNode = newSubTreeManager.appendOne(newTrees.back().nodes, states_[node]);
                        ret.nodeMap.insert(std::make_pair(node, newNode));
                    }
                }
                //No empty trees
                verify(newTrees.back().nodes != InvalidBatchId);
            }
            else {
                //This network is removed
            }
        }
        Core::Application::us()->log() << "count of new nodes: " << newNodes.size();
        verify(newNodes.size());
        trees_.swap(newTrees);
        states_.swap(newNodes);
        subTreeListBatches_.swap(newSubTreeListBatches);
        edgeTargetLists_.swap(newEdgeTargetLists);
    }
    verify(states_.size());

    std::vector<StateId> newEdgeTargetBatches;

    Tools::BatchManager<SuccessorBatchId, StateId, HMMState, false, InvalidBatchId, SingleSuccessorBatchMask> newEdgeTargetManager(newEdgeTargetBatches, states_);
    //Must be created before adding the initial items, since it clears the lists
    newEdgeTargetBatches.push_back(0);

    //Map the direct node members
    for (u32 node = 1; node < states_.size(); ++node) {
        std::unordered_map<TreeIndex, TreeIndex>::iterator it;

        //Map the edge-targets of single batches
        SuccessorBatchId newBatch = InvalidBatchId;
        SuccessorBatchId oldBatch = states_[node].successors;

        for (Tools::BatchManager<SuccessorBatchId, StateId, HMMState, false, Search::InvalidBatchId, SingleSuccessorBatchMask>::Iterator it = edgeTargetManager_.getIterator(oldBatch); it; ++it) {
            if (IS_LABEL(*it)) {
                //It's an label encoded as negative number
                newEdgeTargetManager.appendToBatch(newBatch, *it, *it + 1);
            }
            else {
                //It's a node
                verify(counter.visited.find(*it) != counter.visited.end());
                std::unordered_map<StateId, StateId>::const_iterator nodeMapIt = ret.nodeMap.find(*it);
                verify(nodeMapIt != ret.nodeMap.end());
                newEdgeTargetManager.appendToBatch(newBatch, (*nodeMapIt).second, (*nodeMapIt).second + 1);
            }
        }

        states_[node].successors = newBatch;
    }

    for (u32 batchNum = 1; batchNum < edgeTargetLists_.size(); ++batchNum) {
        if (edgeTargetLists_[batchNum]) {
            SuccessorBatchId oldBatch  = edgeTargetLists_[batchNum];
            edgeTargetLists_[batchNum] = InvalidBatchId;

            for (Tools::BatchManager<SuccessorBatchId, StateId, HMMState, false, Search::InvalidBatchId, SingleSuccessorBatchMask>::Iterator it = edgeTargetManager_.getIterator(oldBatch); it; ++it) {
                if (IS_LABEL(*it)) {
                    //It's an label encoded as negative number
                    newEdgeTargetManager.appendToBatch(edgeTargetLists_[batchNum], *it, *it + 1);
                }
                else {
                    //It's a node
                    verify(counter.visited.find(*it) != counter.visited.end());
                    std::unordered_map<StateId, StateId>::const_iterator nodeMapIt = ret.nodeMap.find(*it);
                    verify(nodeMapIt != ret.nodeMap.end());
                    newEdgeTargetManager.appendToBatch(edgeTargetLists_[batchNum], (*nodeMapIt).second, (*nodeMapIt).second + 1);
                }
            }
        }
    }

    edgeTargetBatches_.swap(newEdgeTargetBatches);
    verify(states_.size());

    {
        CountSizeTreeWalker counter2(*this);
        counter2.stopAtVisited = true;

        ///Check again and make sure the same count of nodes is reachable
        Core::Application::us()->log() << "re-calculating reachable nodes and trees";
        for (std::list<StateId>::const_iterator it = startNodes.begin(); it != startNodes.end(); ++it) {
            std::unordered_map<StateId, StateId>::const_iterator mapIt = ret.nodeMap.find(*it);
            verify(mapIt != ret.nodeMap.end());

            StateId node = (*mapIt).second;

            counter2.visit(node, 1);
            counter2.visitedTrees.insert(masterTree);
        }
        Core::Application::us()->log() << "previous reachable nodes: " << counter.visited.size() << " new reachable nodes: " << counter2.visited.size() << " new total nodes: " << states_.size();
        Core::Application::us()->log() << "previous trees: " << counter.visitedTrees.size() << " new trees: " << counter2.visitedTrees.size();
        Core::Application::us()->log() << "previous exits: " << counter.visitedFinalOutputs << " new exits: " << counter2.visitedFinalOutputs;
        verify(counter2.visited.size() == counter.visited.size());
        verify(counter2.visitedTrees.size() == counter.visitedTrees.size());
    }

    return ret;
}

std::set<StateId> HMMStateNetwork::CleanupResult::mapNodes(const std::set<StateId>& nodes) const {
    std::set<StateId> ret;
    for (std::set<StateId>::const_iterator it = nodes.begin(); it != nodes.end(); ++it) {
        Core::HashMap<StateId, StateId>::const_iterator mapIt = nodeMap.find(*it);
        verify(mapIt != nodeMap.end());
        ret.insert(mapIt->second);
    }
    return ret;
}
}  // namespace Search
