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
        : subTreeManager_(subTreeListBatches_, states_), edgeTargetManager_(edgeTargetBatches_, states_), tree_() {
    // The zero index is reserved as "invalid", so push one dummy item into all arrays
    states_.push_back(HMMState());
    subTreeListBatches_.push_back(0);
    edgeTargetBatches_.push_back(0);
    edgeTargetLists_.push_back(0);
    verify(sizeof(HMMState) % sizeof(u32) == 0);
}

StateId HMMStateNetwork::allocateTreeNode() {
    StateId ret = subTreeManager_.appendOne(tree_.nodes, HMMState());
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

    // Special case for only one item
    edgeTargetManager_.appendToBatch(batch, target, target + 1);

    verify(batch != InvalidBatchId);
}

void HMMStateNetwork::addOutputToEdge(SuccessorBatchId& list, u32 outputIndex) {
    addTargetToEdge(list, ID_FROM_LABEL(outputIndex));
}

u32 HMMStateNetwork::stateCount() const {
    return states_.size();
}

bool HMMStateNetwork::write(Core::MappedArchiveWriter writer) {
    u32 version = DiskFormatVersionV2;
    // The previous version used a vector of trees, where index 0 represented an invalid tree and index 1 contained the actual master tree.
    // Therefore, to mainain backward compatibility, the tree needs to be written into a similar vector structure, which is then saved to the cache.
    std::vector<Tree> trees = {Tree(), tree_};
    writer << version << subTreeListBatches_ << states_ << edgeTargetLists_ << edgeTargetBatches_ << trees;
    return writer.good();
}

bool HMMStateNetwork::read(Core::MappedArchiveReader reader) {
    u32 version = 0;
    reader >> version;

    if (version == DiskFormatVersionV1) {
        // The previous version used a vector of trees, where index 0 represented an invalid tree and index 1 contained the actual master tree.
        // Therefore, to maintain backward compatibility, the cache needs to be read into a vector again and the tree can then be retrieved from index 1.
        std::vector<Tree>       trees;
        std::vector<HMMStateV1> states;
        reader >> subTreeListBatches_ >> states >> edgeTargetLists_ >> edgeTargetBatches_ >> trees;
        tree_ = trees[1];

        if (!reader.good()) {
            return false;
        }

        states_.clear();
        states_.reserve(states.size());

        std::transform(
                states.begin(),
                states.end(),
                std::back_inserter(states_),
                [](HMMStateV1 s) { return s.toHMMState(); });
    }
    else if (version == DiskFormatVersionV2) {
        // The previous version used a vector of trees, where index 0 represented an invalid tree and index 1 contained the actual master tree.
        // Therefore, to maintain backward compatibility, the cache needs to be read into a vector again and the tree can then be retrieved from index 1.
        std::vector<Tree> trees;
        reader >> subTreeListBatches_ >> states_ >> edgeTargetLists_ >> edgeTargetBatches_ >> trees;
        tree_ = trees[1];
    }
    else {
        return false;
    }

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

HMMStateNetwork::CleanupResult HMMStateNetwork::cleanup(std::list<Search::StateId> startNodes, bool clearDeadEnds, bool onlyBatches) {
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
                --node;  // Process the same node again, as more targets may get removed
        }

        Core::Application::us()->log() << "cleared " << cleared << " dead-end nodes";
    }

    Core::Application::us()->log() << "total nodes before cleanup: " << states_.size();

    CleanupResult       ret;
    CountSizeTreeWalker counter(*this);
    if (onlyBatches) {
        for (StateId node = 1; node < states_.size(); ++node)
            counter.visited.insert(node);
    }
    else {
        counter.stopAtVisited = true;

        /// Mark all reachable nodes
        Core::Application::us()->log() << "calculating reachable nodes";
        for (std::list<StateId>::const_iterator it = startNodes.begin(); it != startNodes.end(); ++it)
            counter.visit(*it, 1);
    }

    {
        Tree                          newTree;
        std::vector<StateId>          newSubTreeListBatches;
        std::vector<HMMState>         newNodes;
        std::vector<SuccessorBatchId> newEdgeTargetLists;

        // Must be created before adding the initial items, since it clears the lists
        Tools::BatchManager<SubTreeListId, StateId, HMMState, true, InvalidBatchId> newSubTreeManager(newSubTreeListBatches, newNodes);

        newEdgeTargetLists.push_back(0);
        newSubTreeListBatches.push_back(0);
        newNodes.push_back(HMMState());

        std::vector<u32> orderBehind(states_.size(), 0);
        std::vector<u32> follow(states_.size(), 0);

        /// @todo Build a topology and order the nodes in a stable way based on that
        // Build the order so that the second-order batches are continuous
        Tools::BatchManager<SubTreeListId, StateId, HMMState, true, Search::InvalidBatchId>::Iterator it = subTreeManager_.getIterator(tree_.nodes);
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
            Tools::BatchManager<SubTreeListId, StateId, HMMState, true, Search::InvalidBatchId>::Iterator it = subTreeManager_.getIterator(tree_.nodes);
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

        // Build the order so that the second-order batches are continuous
        {
            // Transfer network into new list
            newTree       = tree_;
            newTree.nodes = InvalidBatchId;

            // Transfer nodes into new batches
            for (u32 idx = 0; idx < ordered.size(); ++idx) {
                StateId node = ordered[idx];
                if (counter.visited.find(node) != counter.visited.end()) {
                    verify(newNodes.size() > 0);
                    StateId newNode = newSubTreeManager.appendOne(newTree.nodes, states_[node]);
                    ret.nodeMap.insert(std::make_pair(node, newNode));
                }
            }
            // No empty trees
            verify(newTree.nodes != InvalidBatchId);
        }

        Core::Application::us()->log() << "count of new nodes: " << newNodes.size();
        verify(newNodes.size());
        tree_ = newTree;
        states_.swap(newNodes);
        subTreeListBatches_.swap(newSubTreeListBatches);
        edgeTargetLists_.swap(newEdgeTargetLists);
    }
    verify(states_.size());

    std::vector<StateId> newEdgeTargetBatches;

    Tools::BatchManager<SuccessorBatchId, StateId, HMMState, false, InvalidBatchId, SingleSuccessorBatchMask> newEdgeTargetManager(newEdgeTargetBatches, states_);
    // Must be created before adding the initial items, since it clears the lists
    newEdgeTargetBatches.push_back(0);

    // Map the direct node members
    for (u32 node = 1; node < states_.size(); ++node) {
        // Map the edge-targets of single batches
        SuccessorBatchId newBatch = InvalidBatchId;
        SuccessorBatchId oldBatch = states_[node].successors;

        for (Tools::BatchManager<SuccessorBatchId, StateId, HMMState, false, Search::InvalidBatchId, SingleSuccessorBatchMask>::Iterator it = edgeTargetManager_.getIterator(oldBatch); it; ++it) {
            if (IS_LABEL(*it)) {
                // It's an label encoded as negative number
                newEdgeTargetManager.appendToBatch(newBatch, *it, *it + 1);
            }
            else {
                // It's a node
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
                    // It's an label encoded as negative number
                    newEdgeTargetManager.appendToBatch(edgeTargetLists_[batchNum], *it, *it + 1);
                }
                else {
                    // It's a node
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

        /// Check again and make sure the same count of nodes is reachable
        Core::Application::us()->log() << "re-calculating reachable nodes";
        for (std::list<StateId>::const_iterator it = startNodes.begin(); it != startNodes.end(); ++it) {
            std::unordered_map<StateId, StateId>::const_iterator mapIt = ret.nodeMap.find(*it);
            verify(mapIt != ret.nodeMap.end());

            StateId node = (*mapIt).second;

            counter2.visit(node, 1);
        }
        Core::Application::us()->log() << "previous reachable nodes: " << counter.visited.size() << " new reachable nodes: " << counter2.visited.size() << " new total nodes: " << states_.size();
        Core::Application::us()->log() << "previous exits: " << counter.visitedFinalOutputs << " new exits: " << counter2.visitedFinalOutputs;
        verify(counter2.visited.size() == counter.visited.size());
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
