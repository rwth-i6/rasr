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
#include "PersistentStateTree.hh"

#include <sstream>
#include <math.h>

#include <Am/ClassicAcousticModel.hh>
#include <Core/MappedArchive.hh>
#include <Search/StateTree.hh>
#include <Search/AdvancedTreeSearch/BatchManager.hh>
#include <Search/AdvancedTreeSearch/Helpers.hh>

#include "TreeStructure.hh"

using namespace Search;
using namespace Core;

static const Core::ParameterString paramCacheArchive(
        "cache-archive",
        "cache archive in which the persistent state-network should be cached",
        "global-cache");

static u32 formatVersion = 12;

namespace Search {
struct ConvertTree {
    const Search::StateTree*                                             tree;
    HMMStateNetwork&                                                     subtrees;
    StateId                                                              rootSubTree;
    StateId                                                              ciRootNode;
    std::map<StateTree::Exit, u32>                                       exits;  // Maps exits to label-indices @todo Make this a hash_map
    std::vector<PersistentStateTree::Exit>                               exitVector;
    Core::HashMap<StateId, StateTree::StateId>                           statesForNodes;
    Core::HashMap<StateTree::StateId, StateId>                           nodesForStates;
    std::set<StateId>                                                    coarticulatedRootNodes;
    u32                                                                  lostNodeIndices;
    std::map<StateId, std::pair<Bliss::Phoneme::Id, Bliss::Phoneme::Id>> rootTransitDescriptions;

    ConvertTree(const Search::StateTree* _tree, HMMStateNetwork& _subtrees)
            : tree(_tree), subtrees(_subtrees), lostNodeIndices(0) {
    }

    void convert() {
        for (u32 a = 0; a < tree->states_.size(); ++a) {
            StateId created = subtrees.allocateTreeNode();
            verify(a + 1 == created);
        }

        for (int a = ((int)tree->states_.size()) - 1; a >= 0; --a) {
            StateTree::StateId state = a;
            StateId            node  = a + 1;
            convert(state, node);
        }

        std::set<StateTree::StateId> coarticulatedRootStates;
        for (StateTree::StateId state = 0; state < tree->nStates(); ++state) {
            for (std::list<StateTree::Exit>::const_iterator it = tree->state(state).exits.begin(); it != tree->state(state).exits.end(); ++it) {
                if ((*it).transitEntry != tree->root())
                    coarticulatedRootStates.insert((*it).transitEntry);
            }
        }

        /// Make sure a node is created for every single state, so that also the coarticulated roots are respected

        for (std::set<StateTree::StateId>::iterator stateIt = coarticulatedRootStates.begin(); stateIt != coarticulatedRootStates.end(); ++stateIt) {
            StateTree::StateId state = *stateIt;

            verify(nodesForStates.find(state) != nodesForStates.end());
            coarticulatedRootNodes.insert(nodesForStates[state]);
            rootTransitDescriptions[nodesForStates[state]] = tree->describeRootState(state);
        }

        verify(nodesForStates.find(tree->root()) != nodesForStates.end());

        rootTransitDescriptions[nodesForStates[tree->root()]]   = tree->describeRootState(tree->root());
        rootTransitDescriptions[nodesForStates[tree->ciRoot()]] = tree->describeRootState(tree->ciRoot());

        /// Care about outputs

        for (std::unordered_map<StateTree::StateId, StateId>::const_iterator it = nodesForStates.begin(); it != nodesForStates.end(); ++it) {
            StateId            node  = (*it).second;
            StateTree::StateId state = (*it).first;
            verify(node == state + 1);

            std::set<u32>           exitIndices;
            const StateTree::State& realState(tree->state(state));

            for (std::list<StateTree::Exit>::const_iterator it = realState.exits.begin(); it != realState.exits.end(); ++it) {
                std::map<StateTree::Exit, u32>::iterator exitEntry = exits.find(*it);

                if (exitEntry == exits.end()) {
                    PersistentStateTree::Exit                                 e;
                    std::unordered_map<StateTree::StateId, StateId>::iterator nodeIt = nodesForStates.find(it->transitEntry);
                    verify(nodeIt != nodesForStates.end());

                    if (it->pronunciation) {
                        e.pronunciation = it->pronunciation->id();
                        verify(it->pronunciation);
                    }
                    else {
                        e.pronunciation = Bliss::LemmaPronunciation::invalidId;
                    }
                    e.transitState = nodeIt->second;

                    exitVector.push_back(e);
                    exitEntry = exits.insert(std::make_pair(*it, exitVector.size() - 1)).first;
                }
                exitIndices.insert(exitEntry->second);
            }

            // Add connections to the attached outputs/exits
            for (std::set<u32>::iterator it = exitIndices.begin(); it != exitIndices.end(); ++it)
                subtrees.addOutputToEdge(subtrees.state(node).successors, *it);
        }
    }

private:
    void convert(StateTree::StateId stateId, StateId node) {
        if (stateId == tree->root())
            rootSubTree = node;
        if (stateId == tree->ciRoot())
            ciRootNode = node;

        if (nodesForStates.find(stateId) != nodesForStates.end()) {
            verify(nodesForStates[stateId] == node);
            verify(statesForNodes[node] = stateId);
            return;
        }

        verify(stateId + 1 == node);

        nodesForStates[stateId] = node;

        statesForNodes[node] = stateId;

        const StateTree::StateDesc& state = tree->stateDesc(stateId);

        subtrees.state(node).stateDesc = state;

        // Build successor structure
        std::pair<StateTree::SuccessorIterator, StateTree::SuccessorIterator> successors = tree->successors(stateId);

        StateId current = node;  // Just to verify the order

        for (; successors.first != successors.second; ++successors.first) {
            std::unordered_map<StateTree::StateId, StateId>::iterator nodeIt = nodesForStates.find(*successors.first);
            verify(nodeIt != nodesForStates.end());
            verify(nodeIt->second > current);
            current = nodeIt->second;

            subtrees.addNodeToEdge(subtrees.state(node).successors, current);
        }
    }
};

PersistentStateTree::PersistentStateTree(Core::Configuration config, Core::Ref<const Am::AcousticModel> acousticModel, Bliss::LexiconRef lexicon, TreeBuilderFactory treeBuilderFactory)
        : rootState(0),
          ciRootState(0),
          archive_(paramCacheArchive(Core::Configuration(config, "search-network"))),
          acousticModel_(acousticModel),
          lexicon_(lexicon),
          config_(config),
          treeBuilderFactory_(treeBuilderFactory) {
    if (acousticModel_.get() && lexicon_.get()) {
        const Am::ClassicAcousticModel* am = required_cast(const Am::ClassicAcousticModel*, acousticModel.get());
        Core::DependencySet             d;
        am->stateModel()->hmmTopologySet().getDependencies(d);
        am->stateTying()->getDependencies(d);
        dependencies_.add("acoustic model", d);
        dependencies_.add("lexicon", lexicon->getDependency());
    }
}

u32 PersistentStateTree::getChecksum() const {
    return dependencies_.getChecksum() + structure.getChecksum() + exits.size();
}

std::string PersistentStateTree::archiveEntry() const {
    return isBackwardRecognition(config_) ? "backward-state-network-image" : "state-network-image";
}

bool PersistentStateTree::read(int transformation) {
    MappedArchiveReader in = Core::Application::us()->getCacheArchiveReader(archive_, archiveEntry());

    if (!in.good())
        return false;

    int storedTransformation = 0;
    in >> storedTransformation;
    if (storedTransformation != transformation) {
        Core::Application::us()->log() << "failed reading state network because of transformation mismatch: " << storedTransformation
                                       << " vs requested " << transformation;
        return false;
    }

    bool ret = read(in);

    if (ret)
        Core::Application::us()->log() << "reading ready";
    else
        Core::Application::us()->log() << "reading failed";

    return ret;
}

void PersistentStateTree::build() {
    Core::Application::us()->log() << "retrieving classical state network";

    StateTree* tree = new StateTree(Core::Configuration(config_, "state-network"), lexicon_, acousticModel_);

    Core::Application::us()->log() << "converting from classical state network";

    ConvertTree convert(tree, structure);

    convert.convert();
    exits = convert.exitVector;

    rootState                       = convert.rootSubTree;
    ciRootState                     = convert.ciRootNode;
    coarticulatedRootStates         = convert.coarticulatedRootNodes;
    unpushedCoarticulatedRootStates = coarticulatedRootStates;

    rootTransitDescriptions = convert.rootTransitDescriptions;

    delete tree;

    Core::Application::us()->log() << "network conversion ready";
}

bool PersistentStateTree::write(int transformation) {
    if (archive_.empty())
        return false;

    std::string filename = archive_;

    Core::Application::us()->log() << "writing state network into " << archive_;

    MappedArchiveWriter writer = Core::Application::us()->getCacheArchiveWriter(archive_, archiveEntry());

    if (!writer.good())
        return false;

    writer << transformation;

    write(writer);

    return writer.good();
}

template<class T>
MappedArchiveReader& operator>>(MappedArchiveReader& reader, std::set<T>& target) {
    target.clear();
    std::vector<T> vec;
    reader >> vec;
    target.insert(vec.begin(), vec.end());
    return reader;
}

template<class T>
MappedArchiveWriter& operator<<(MappedArchiveWriter& writer, const std::set<T>& set) {
    writer << std::vector<T>(set.begin(), set.end());
    return writer;
}

template<class T, class T2>
MappedArchiveReader& operator>>(MappedArchiveReader& reader, std::map<T, T2>& target) {
    target.clear();
    std::vector<std::pair<T, T2>> vec;
    reader >> vec;
    target.insert(vec.begin(), vec.end());
    return reader;
}

template<class T, class T2>
MappedArchiveWriter& operator<<(MappedArchiveWriter& writer, const std::map<T, T2>& set) {
    writer << std::vector<std::pair<T, T2>>(set.begin(), set.end());
    return writer;
}

void PersistentStateTree::write(Core::MappedArchiveWriter out) {
    // In the previous version, a master tree was used and the index was saved in the cache.
    // For backward compatibility, a dummy index is now written instead.
    // This index is not used further and has no effect on functionality.
    u32 dummyIndex = 1;
    out << formatVersion << dummyIndex << (u32)dependencies_.getChecksum();

    structure.write(out);
    out << exits;

    out << coarticulatedRootStates << unpushedCoarticulatedRootStates;
    out << rootTransitDescriptions << pushedWordEndNodes << uncoarticulatedWordEndStates;
    out << rootState << ciRootState;
}

bool PersistentStateTree::read(Core::MappedArchiveReader in) {
    u32 v;
    in >> v;

    /// @todo Eventually do memory-mapping

    if (v != formatVersion) {
        Core::Application::us()->log() << "Wrong compressed network format, need " << formatVersion << " got " << v;
        return false;
    }

    Core::Application::us()->log() << "Loading persistent network format version " << formatVersion;

    u32 dependenciesChecksum = 0;

    // In the previous version, a master tree was used and the index was saved in the cache.
    // For backward compatibility, read this into a dummy index.
    // This index is not used further and has no effect on functionality.
    u32 dummyIndex;
    in >> dummyIndex >> dependenciesChecksum;

    if (dependenciesChecksum != dependencies_.getChecksum()) {
        Core::Application::us()->log() << "dependencies of the network image don't equal the required dependencies with checksum " << dependenciesChecksum;
        return false;
    }

    if (!structure.read(in))
        return false;

    in >> exits;

    in >> coarticulatedRootStates >> unpushedCoarticulatedRootStates >> rootTransitDescriptions;
    in >> pushedWordEndNodes >> uncoarticulatedWordEndStates;

    in >> rootState >> ciRootState;

    return in.good();
}

void PersistentStateTree::removeOutputs() {
    Core::Application::us()->log() << "removing outputs from the search network";

    for (StateId node = 1; node < structure.stateCount(); ++node) {
        HMMStateNetwork::ChangePlan change = structure.change(node);

        for (HMMStateNetwork::SuccessorIterator it = structure.successors(node); it; ++it)
            if (it.isLabel())
                change.removeSuccessor(*it);
        ;

        change.apply();
    }

    std::list<StateId> rootsList;

    {
        std::set<StateId> roots = coarticulatedRootStates;
        roots.insert(rootState);
        roots.insert(ciRootState);

        // Also collect all transition-successors as coarticulated roots
        for (StateId node = 1; node < structure.stateCount(); ++node) {
            for (HMMStateNetwork::SuccessorIterator target = structure.successors(node); target; ++target)
                if (target.isLabel())
                    roots.insert(exits[target.label()].transitState);
        }

        for (std::set<StateId>::iterator it = roots.begin(); it != roots.end(); ++it)
            rootsList.push_back(*it);
    }

    HMMStateNetwork::CleanupResult cleanupResult = structure.cleanup(rootsList, false, true);

    for (std::unordered_map<StateId, StateId>::const_iterator it = cleanupResult.nodeMap.begin(); it != cleanupResult.nodeMap.end(); ++it) {
        if (it->first != it->second)
            std::cout << "mapped " << it->first << " to " << it->second << std::endl;
        verify(it->first == it->second);
    }
}

HMMStateNetwork::CleanupResult PersistentStateTree::cleanup(bool cleanupExits) {
    Core::Application::us()->log() << "cleaning up the search network";

    if (cleanupExits) {
        std::vector<Exit> newExits;

        for (StateId node = 1; node < structure.stateCount(); ++node) {
            std::set<u32> removeList;
            std::set<u32> addList;

            for (HMMStateNetwork::SuccessorIterator it = structure.successors(node); it; ++it) {
                if (it.isLabel()) {
                    removeList.insert(it.label());
                    addList.insert(newExits.size());
                    newExits.push_back(exits[it.label()]);
                }
            }

            for (std::set<u32>::iterator it = removeList.begin(); it != removeList.end(); ++it)
                structure.removeOutputFromNode(node, *it);

            for (std::set<u32>::iterator it = addList.begin(); it != addList.end(); ++it)
                structure.addOutputToEdge(structure.state(node).successors, *it);
        }

        Core::Application::us()->log() << "changed number of exits from " << exits.size() << " to " << newExits.size();
        exits.swap(newExits);
    }

    std::list<StateId> rootsList;

    for (std::set<StateId>::iterator it = unpushedCoarticulatedRootStates.begin(); it != unpushedCoarticulatedRootStates.end(); ++it)
        verify(coarticulatedRootStates.count(*it));

    {
        std::set<StateId> roots = coarticulatedRootStates;
        roots.insert(rootState);
        roots.insert(ciRootState);
        for (StateId s : otherRootStates) {
            roots.insert(s);
        }

        // Also collect all transition-successors as coarticulated roots
        for (StateId node = 1; node < structure.stateCount(); ++node) {
            for (HMMStateNetwork::SuccessorIterator target = structure.successors(node); target; ++target)
                if (target.isLabel())
                    roots.insert(exits[target.label()].transitState);
        }

        for (std::set<StateId>::iterator it = roots.begin(); it != roots.end(); ++it)
            rootsList.push_back(*it);
    }

    ///@todo Go through the search tree, and collect the required coarticulated root nodes

    HMMStateNetwork::CleanupResult cleanupResult = structure.cleanup(rootsList);

    Core::HashMap<StateId, StateId>::const_iterator targetNodeIt;
    if (rootState) {
        verify(cleanupResult.nodeMap.find(rootState) != cleanupResult.nodeMap.end());  // Root-node must stay unchanged
        verify(cleanupResult.nodeMap.find(rootState)->second == rootState);
        targetNodeIt = cleanupResult.nodeMap.find(rootState);
        verify(targetNodeIt != cleanupResult.nodeMap.end());
        rootState = (*targetNodeIt).second;

        targetNodeIt = cleanupResult.nodeMap.find(ciRootState);
        verify(targetNodeIt != cleanupResult.nodeMap.end());
        ciRootState = (*targetNodeIt).second;
    }

    std::set<StateId>       tempCoarticulatedRootNodes;
    RootTransitDescriptions tempRootTransitDescriptions;
    for (std::set<StateId>::iterator it = coarticulatedRootStates.begin(); it != coarticulatedRootStates.end(); ++it) {
        targetNodeIt = cleanupResult.nodeMap.find(*it);

        if (targetNodeIt != cleanupResult.nodeMap.end()) {
            tempCoarticulatedRootNodes.insert((*targetNodeIt).second);

            std::map<StateId, std::pair<Bliss::Phoneme::Id, Bliss::Phoneme::Id>>::iterator oldIt = rootTransitDescriptions.find(*it);
            verify(oldIt != rootTransitDescriptions.end());

            verify(tempRootTransitDescriptions.find((*targetNodeIt).second) == tempRootTransitDescriptions.end());

            tempRootTransitDescriptions.insert(std::make_pair((*targetNodeIt).second, (*oldIt).second));
        }
    }

    if (rootState)
        tempRootTransitDescriptions.insert(*rootTransitDescriptions.find(rootState));  ///@todo Also add the overall root node to coarticulatedRootNodes, then this is not needed

    Core::Application::us()->log() << "deleted " << tempCoarticulatedRootNodes.size() - coarticulatedRootStates.size() << " coarticulated root nodes";

    verify(coarticulatedRootStates.size() == tempCoarticulatedRootNodes.size());
    verify(rootTransitDescriptions.size() == tempRootTransitDescriptions.size());

    coarticulatedRootStates.swap(tempCoarticulatedRootNodes);
    rootTransitDescriptions.swap(tempRootTransitDescriptions);

    for (std::vector<Exit>::iterator exitIt = exits.begin(); exitIt != exits.end(); ++exitIt) {
        targetNodeIt = cleanupResult.nodeMap.find(exitIt->transitState);
        verify(targetNodeIt != cleanupResult.nodeMap.end());
        exitIt->transitState = (*targetNodeIt).second;
    }

    //   pushedWordEndNodes = cleanupResult.mapNodes( pushedWordEndNodes );
    uncoarticulatedWordEndStates = cleanupResult.mapNodes(uncoarticulatedWordEndStates);
    //   uncoarticulatedPushedWordEndNodes = cleanupResult.mapNodes( uncoarticulatedPushedWordEndNodes );
    unpushedCoarticulatedRootStates = cleanupResult.mapNodes(unpushedCoarticulatedRootStates);

    return cleanupResult;
}

void PersistentStateTree::dumpDotGraph(std::string file, const std::vector<int>& nodeDepths) {
    std::ofstream os(file.c_str());
    verify(os.good());

    std::string name = "search network";

    os << "digraph \"" << name << "\" {" << std::endl
       << "ranksep = 1.5" << std::endl
       << "rankdir = LR" << std::endl
       << "node [fontname=\"Helvetica\"]" << std::endl
       << "edge [fontname=\"Helvetica\"]" << std::endl;

    for (StateId node = 1; node < structure.stateCount(); ++node) {
        if (!nodeDepths.empty()) {
            int depth = nodeDepths[node];
            os << Core::form("n%d [label=\"%d\\nd=%d\\nm=%d\\nt=%d", node, node, depth, structure.state(node).stateDesc.acousticModel, structure.state(node).stateDesc.transitionModelIndex);
        }
        else {
            os << Core::form("n%d [label=\"%d\\nm=%d\\nt=%d", node, node, structure.state(node).stateDesc.acousticModel, structure.state(node).stateDesc.transitionModelIndex);
        }

        for (HMMStateNetwork::SuccessorIterator target = structure.successors(node); target; ++target)
            if (target.isLabel() && exits[target.label()].pronunciation != Bliss::LemmaPronunciation::invalidId)
                os << "\\n"
                   << lexicon_->lemmaPronunciation(exits[target.label()].pronunciation)->lemma()->preferredOrthographicForm()
                   << Core::form(" tr=%d", exits[target.label()].transitState);

        os << "\"";
        bool is_other_root = std::find(otherRootStates.begin(), otherRootStates.end(), node) != otherRootStates.end();
        if (node == rootState || node == ciRootState || uncoarticulatedWordEndStates.count(node) || is_other_root)
            os << ",shape=box";
        os << "]" << std::endl;

        for (HMMStateNetwork::SuccessorIterator target = structure.successors(node); target; ++target)
            if (!target.isLabel())
                os << Core::form("n%d -> n%d\n", node, *target);

        for (HMMStateNetwork::SuccessorIterator target = structure.successors(node); target; ++target)
            if (target.isLabel() && exits[target.label()].pronunciation == Bliss::LemmaPronunciation::invalidId)
                os << Core::form("n%d -> n%d [style=dashed]\n", node, exits[target.label()].transitState);
    }

    os << "}" << std::endl;
}
}  // namespace Search
