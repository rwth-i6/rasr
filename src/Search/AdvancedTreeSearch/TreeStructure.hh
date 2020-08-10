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
#ifndef SEARCH_TREESTRUCTURE_HH
#define SEARCH_TREESTRUCTURE_HH

#include <Core/MappedArchive.hh>
#include <Search/StateTree.hh>
#include <vector>
#include "BatchManager.hh"

#define inline_ __attribute__((always_inline)) inline

/**
 * This file contains a storable, efficient and dynamic structure that allows representing the state-network
 * in speech recognition flexibly. The structure can dynamically be changed by creating new states and successor-links,
 * and a "cleanup" operation can be used to optimize the structure after significant changes.
 */

///@todo Enable batch-merging (mergeBatches_), and make sure the structure is optimal.
///@todo Allow "freezing" all batches: No follower-gap in the middle (should save 1/3 of space for the batches)

namespace Search {
/// This index represents an arbitrary list of successor-states and labels
typedef u32 SuccessorBatchId;

enum {
    /// When this bitmask set on an SuccessorBatchId, the id represents only one successor (or label),
    /// and the index can be extracted through "id & (~SingleSuccessorBatchMask)"
    SingleSuccessorBatchMask = 1 << 30,
    LabelMask                = 1 << 27
};

///Global index of a tree or subtree
typedef u32 TreeIndex;

///Index of a state or label (see IS_LABEL, ID_FROM_LABEL, and LABEL_FROM_ID)
typedef u32 StateId;

///@todo Maybe this should be zero!
static const StateId invalidTreeNodeIndex = (StateId)-1;

/// Index representing the batch of all nodes that are contained by one tree
typedef u32 SubTreeListId;

/// Returns whether the given edge-successor index represents a label rather than a state
inline_ bool IS_LABEL(const StateId x) {
    return x & LabelMask;
}

/// Returns the representation index of this label as an edge-successor
inline_ StateId ID_FROM_LABEL(const u32 x) {
    return x | LabelMask;
}

/// Encodes the given index as an edge-successor so that it represents a label
inline_ u32 LABEL_FROM_ID(StateId x) {
    return x & (~LabelMask);
}

enum {
    InvalidBatchId = 0
};

/// The standard network state, representing a state and its successor-links
struct HMMState {
    inline_ HMMState()
            : successors(InvalidBatchId) {
    }

    inline_ HMMState& operator=(const HMMState& rhs) {
        stateDesc  = rhs.stateDesc;
        successors = rhs.successors;
        return *this;
    }

    ///This must be initialized explicitly after creating the state
    StateTree::StateDesc stateDesc;

    ///Batch of successor states, managed through a batch-manager in TreeStructure
    SuccessorBatchId successors;

    /// Returns true if the label-edges batch represents only one single successor, which can be handled more efficiently
    inline_ bool hasSingleSuccessor() const {
        return (successors & SingleSuccessorBatchMask) == SingleSuccessorBatchMask;
    }

    /// If hasSingleSuccessor returned true, then this can be used to retrieve the single successor
    inline_ StateId singleSuccessor() const {
        return successors & (~SingleSuccessorBatchMask);
    }
};

struct Tree {
    Tree()
            : nodes(InvalidBatchId) {
    }
    ///All nodes contained by this tree. Managed as a batch by TreeStructure
    SubTreeListId nodes;
};

class HMMStateNetwork {
public:
    enum {
        //Index of the empty network
        //The empty network has no node, and exactly one label that is to be activated directly
        EmptyTreeIndex = 0
    };

    enum {
        DiskFormatVersion = 1
    };

    typedef Tools::BatchIndexIterator<SubTreeListId, StateId, InvalidBatchId, 0> SubTreeIterator;

    struct SuccessorIterator : public Tools::BatchIndexIterator<SuccessorBatchId, StateId, InvalidBatchId, SingleSuccessorBatchMask> {
        SuccessorIterator(SuccessorBatchId batchId, const std::vector<StateId>& batches)
                : Tools::BatchIndexIterator<SuccessorBatchId, StateId, InvalidBatchId, SingleSuccessorBatchMask>(batchId, batches) {
        }
        inline bool isLabel() const {
            return IS_LABEL(**this);
        }
        inline u32 label() const {
            return LABEL_FROM_ID(**this);
        }
    };

    class EfficientSingleTargetIterator {
    public:
        inline EfficientSingleTargetIterator(StateId target)
                : target_(target) {
        }

        inline void operator++() {
            target_ = invalidTreeNodeIndex;
        }

        inline operator bool() const {
            return target_ != invalidTreeNodeIndex;
        }

        inline StateId operator*() const {
            return target_;
        }

        inline bool isOutput() const {
            return IS_LABEL(**this);
        }

        inline u32 label() const {
            return LABEL_FROM_ID(**this);
        }

    private:
        StateId target_;
    };

    class EfficientBatchTargetIterator {
    public:
        inline EfficientBatchTargetIterator(std::pair<StateId, StateId> targets)
                : targets_(targets) {
        }

        inline operator bool() const {
            return targets_.first != targets_.second;
        }

        inline EfficientBatchTargetIterator& operator++() {
            targets_.first += 1;
            return *this;
        }

        inline StateId operator*() const {
            return targets_.first;
        }

        inline bool isOutput() const {
            return IS_LABEL(**this);
        }

        inline u32 label() const {
            return LABEL_FROM_ID(**this);
        }

    private:
        std::pair<StateId, StateId> targets_;
    };

    HMMStateNetwork();

    ///****** STATE MANAGEMENT ******************************************************************************************

    ///Do not keep pointers/references to the returned tree, the address may change
    inline Tree& tree(TreeIndex index) {
        verify_(index > 0 && index < trees_.size());
        return trees_[index];
    }

    ///Do not keep pointers to the returned state, the address may change when the network is manipulated
    inline_ HMMState& state(StateId state) {
        verify_(state > 0 && state < (int)states_.size());
        return states_[state];
    }

    ///Do not keep pointers to the returned state, the address may change when the network is manipulated
    inline const HMMState& state(StateId state) const {
        verify_(state > 0 && state < (int)states_.size());
        return states_[state];
    }

    ///Allocates a new tree
    TreeIndex allocateTree();

    ///Allocates a new subtree, and adds it into the subtree list of the given parent.
    ///As many subtrees for the same parent should be allocated in a row as possible, so batch-merging can happen
    ///Returns a fully valid subtree(With initialized edge-list)
    StateId allocateTreeNode(TreeIndex parent);

    ///Returns the count of nodes contained by the tree
    inline u32 getNodeCount(TreeIndex parent);

    ///Returns the @p number th node contained in the given parent tree
    inline StateId getTreeNode(TreeIndex parent, u32 number);

    ///Returns the number of nodes contained by the given parent tree
    inline u32 getNodeNumber(TreeIndex parent, StateId node);

    ///Much faster version of getNodeNumber, that only works when the structure has been cleaned
    inline u32 getNodeNumberCleanStructure(TreeIndex parent, StateId node) {
        return node - subTreeListBatches_[tree(parent).nodes];
    }

    ///Returns the total number of trees, which is the maximum upper bound for a valid TreeIndex
    u32 treeCount() const;

    ///Returns the total number of nodes, which is the maximum upper bound for a valid TreeNodeIndex
    u32 stateCount() const;

    struct CleanupResult {
        Core::HashMap<StateId, StateId>     nodeMap;
        Core::HashMap<TreeIndex, TreeIndex> treeMap;

        std::set<StateId> mapNodes(const std::set<StateId>& nodes) const;
    };

    ///Completely removes all trees and nodes that are not reachable from the given start-nodes, compressing the structure
    CleanupResult cleanup(std::list<Search::StateId> startNodes, Search::TreeIndex masterTree, bool clearDeadEnds = true, bool onlyBatches = false);

    ///****** EDGE MANAGEMENT *******************************************************************************************

    ///Adds the given target to the list of targets for the given edge. The referenced id will be changed.
    void addNodeToEdge(SuccessorBatchId& list, StateId target);

    ///Adds the given target to the list of targets for the given edge. The referenced id will be changed.
    void addOutputToEdge(SuccessorBatchId& list, u32 outputIndex);

    void addTargetToNode(StateId node, StateId target) {
        addNodeToEdge(state(node).successors, target);
    }

    void addOutputToNode(StateId node, u32 outputIndex) {
        addOutputToEdge(state(node).successors, outputIndex);
    }

    void removeTargetFromNode(StateId node, StateId target);

    void removeOutputFromNode(StateId node, u32 outputIndex);

    ///Clears all connections behind the given node. The memory will be lost unless a cleanup is done afterwards.
    void clearOutputEdges(StateId node);

    u32 getChecksum() const {
        return states_.size() + edgeTargetBatches_.size() + edgeTargetLists_.size() + trees_.size() + subTreeListBatches_.size();
    }

    ///The change is applied when apply() is called
    class ChangePlan {
    public:
        void addSuccessor(StateId state) {
            remove.erase(state);
            add.insert(state);
        }

        void addSuccessorLabel(u32 label) {
            remove.erase(ID_FROM_LABEL(label));
            add.insert(ID_FROM_LABEL(label));
        }

        void removeSuccessor(StateId state) {
            remove.insert(state);
            add.erase(state);
        }

        void removeSuccessorLabel(u32 label) {
            remove.insert(ID_FROM_LABEL(label));
            add.erase(ID_FROM_LABEL(label));
        }

        void apply();

    private:
        ChangePlan(HMMStateNetwork& _structure, StateId _node)
                : node(_node), structure(&_structure) {
        }

        ChangePlan()
                : node(0), structure(0) {
        }

        std::set<int> add;
        std::set<int> remove;
        friend class HMMStateNetwork;
        StateId          node;
        HMMStateNetwork* structure;
    };

    ChangePlan change(StateId node);

    inline SuccessorIterator batchSuccessors(Search::SuccessorBatchId list) const;

    inline SuccessorIterator successors(StateId node) const {
        return batchSuccessors(state(node).successors);
    }

    inline SuccessorIterator successors(const HMMState& node) const {
        return batchSuccessors(node.successors);
    }

    // Calls the given function-object with each target node
    template<class FunctionObject>
    inline void efficientlyIterateTargets(const HMMState& node, FunctionObject& object) const {
        if (node.hasSingleSuccessor()) {
            object(node.singleSuccessor());
            return;
        }

        std::pair<int, int> targets = batchSuccessorsSimple<false>(node.successors);
        if (targets.first != -1) {
            for (; targets.first != targets.second; ++targets.first)
                object(targets.first);
        }
        else {
            for (SuccessorIterator it = this->successors(node); it; ++it)
                object(*it);
        }
    }

    /// Convenience function: Returns the set of target nodes of the given tree node
    std::set<StateId> targetNodeSet(StateId node) const {
        std::set<StateId> ret;
        for (HMMStateNetwork::SuccessorIterator target = successors(node); target; ++target)
            if (!target.isLabel())
                ret.insert(*target);
        return ret;
    }

    /// Convenience function: Returns the set of target outputs of the given tree node
    std::set<uint> targetOutputSet(StateId node) const {
        std::set<uint> ret;
        for (HMMStateNetwork::SuccessorIterator target = successors(node); target; ++target)
            if (target.isLabel())
                ret.insert(target.label());
        return ret;
    }

    /// Convenience function: Returns the set of target nodes and outputs of the given tree node
    /// Outputs are encoded with IS_OUTPUT and OUTPUT_FROM_INDEX
    std::set<StateId> targetSet(StateId node) const {
        std::set<StateId> ret;
        for (HMMStateNetwork::SuccessorIterator target = successors(node); target; ++target)
            ret.insert(*target);
        return ret;
    }

    ///Returns -1, -1 if this simple version does not work. Then "edgeTargets" has to be used.
    template<bool considerOutputs>
    inline std::pair<int, int> batchSuccessorsSimple(SuccessorBatchId list) const;

    ///Does not work with single-batches! Those must be checked before.
    inline std::pair<int, int> batchSuccessorsSimpleIgnoreLabels(SuccessorBatchId list) const;

    ///Reads out the node-range associated to the given batch. Does not verify whether
    ///the batch is a single-batch or has successor-batches, this has to be checked beforehand.
    inline std::pair<int, int> batchNodeRange(SuccessorBatchId batch) const;

    ///*********************************************************************************************************************

    bool write(Core::MappedArchiveWriter writer);
    /// Returns whether reading was successful. Reading will fail on format version mismatch.
    bool read(Core::MappedArchiveReader reader);

    const std::vector<StateId>& edgeTargetBatches() const {
        return edgeTargetBatches_;
    }

private:
    void addTargetToEdge(SuccessorBatchId& batch, u32 target);
    u32  countReachableEnds(std::vector<u32>& counts, StateId node) const;
    //This manager manages lists of sub-trees, one subtree-list for each network
    std::vector<StateId>                                                        subTreeListBatches_;
    std::vector<HMMState>                                                       states_;
    Tools::BatchManager<SubTreeListId, StateId, HMMState, true, InvalidBatchId> subTreeManager_;

    //Contains one SuccessorBatchId for each label of a subtree, as a linear list
    std::vector<SuccessorBatchId> edgeTargetLists_;

    //This manager groups together edge successors, for edges coming from a common source(usually an label of a subtree)
    std::vector<StateId>                                                                                      edgeTargetBatches_;
    Tools::BatchManager<SuccessorBatchId, StateId, HMMState, false, InvalidBatchId, SingleSuccessorBatchMask> edgeTargetManager_;

    std::vector<Tree> trees_;
};

inline HMMStateNetwork::SuccessorIterator HMMStateNetwork::batchSuccessors(Search::SuccessorBatchId list) const {
    return SuccessorIterator(list, edgeTargetBatches_);
}

inline std::pair<int, int> HMMStateNetwork::batchSuccessorsSimpleIgnoreLabels(SuccessorBatchId batch) const {
    if (batch == InvalidBatchId)
        return std::pair<int, int>(-1, -2);

    const Search::StateId start = edgeTargetBatches_[batch];
    const Search::StateId next  = edgeTargetBatches_[batch + 1];
    const Search::StateId end   = edgeTargetBatches_[batch + 2];

    if (next == InvalidBatchId)
        return std::pair<int, int>(start, end);
    else
        return std::pair<int, int>(-1, -3);
}

template<bool considerOutputs>
inline std::pair<int, int> HMMStateNetwork::batchSuccessorsSimple(SuccessorBatchId batch) const {
    if (batch & SingleSuccessorBatchMask) {
        SuccessorBatchId b(batch & (~SingleSuccessorBatchMask));
        if (!considerOutputs && IS_LABEL(b))
            return std::pair<int, int>(0, 0);
        return std::pair<int, int>(b, b + 1);
    }

    if (batch == InvalidBatchId)
        return std::pair<int, int>(-1, -2);

    const Search::StateId next = edgeTargetBatches_[batch + 1];
    if (next == InvalidBatchId || (not considerOutputs && IS_LABEL(edgeTargetBatches_[next]))) {
        const Search::StateId start = edgeTargetBatches_[batch];
        if (not considerOutputs && IS_LABEL(start))
            return std::pair<int, int>(0, 0);
        //Everything ok, this is a simple continous batch without a follower-batch
        return std::pair<int, int>(start, edgeTargetBatches_[batch + 2]);
    }

    return std::pair<int, int>(-1, -3);
}

std::pair<int, int> HMMStateNetwork::batchNodeRange(SuccessorBatchId batch) const {
    return std::make_pair<int, int>((int)edgeTargetBatches_[batch], (int)edgeTargetBatches_[batch + 2]);
}

u32 HMMStateNetwork::getNodeCount(Search::TreeIndex parent) {
    return subTreeManager_.getIterator(tree(parent).nodes).countToEnd();
}

StateId HMMStateNetwork::getTreeNode(Search::TreeIndex parent, u32 number) {
    Tools::BatchManager<SubTreeListId, StateId, HMMState, true, InvalidBatchId>::Iterator it = subTreeManager_.getIterator(tree(parent).nodes);
    it += number;
    return *it;
}

u32 HMMStateNetwork::getNodeNumber(TreeIndex parent, StateId node) {
    Tools::BatchManager<SubTreeListId, StateId, HMMState, true, InvalidBatchId>::Iterator it = subTreeManager_.getIterator(tree(parent).nodes);
    return it.countUntil(node);
}
}  // namespace Search

#endif  // SEARCH_SUBTREESTRUCTURE_HH
