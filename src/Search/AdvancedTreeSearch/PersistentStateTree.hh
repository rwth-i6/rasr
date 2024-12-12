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
#ifndef PERSISTENT_STATE_TREE_H
#define PERSISTENT_STATE_TREE_H

#include <Core/MappedArchive.hh>
#include "TreeStructure.hh"

template<class Key>
struct MyStandardValueHash {
    inline u32 operator()(Key a) const {
        // a = (a+0x7ed55d16) + (a<<12);
        a = (a ^ 0xc761c23c) ^ (a >> 19);
        // a = (a+0x165667b1) + (a<<5);
        // a = (a+0xd3a2646c) ^ (a<<9);
        a = (a + 0xfd7046c5) + (a << 3);
        // a = (a^0xb55a4f09) ^ (a>>16);
        return a;
    }
};

class AbstractTreeBuilder;

namespace Search {
class HMMStateNetwork;
class StateTree;

class PersistentStateTree {
public:
    using TreeBuilderFactory = std::function<std::unique_ptr<AbstractTreeBuilder>(Core::Configuration, const Bliss::Lexicon&, const Am::AcousticModel&, PersistentStateTree&, bool)>;

    ///@param lexicon This must be given if the resulting exits are supposed to be functional
    PersistentStateTree(Core::Configuration config, Core::Ref<const Am::AcousticModel> acousticModel, Bliss::LexiconRef lexicon, TreeBuilderFactory treeBuilderFactory);

    /// Builds this state tree.
    void build();

    /// Writes the current state of the state tree into the file,
    /// Returns whether writing was successful
    bool write(int transformation = 0);

    /// Reads the state tree from the file.
    ///@return Whether the reading was successful.
    bool read(int transformation = 0);

    /// Cleans up the structure, saving memory and allowing a more efficient iteration.
    /// Node and tree IDs may be changed.
    ///@return An object that contains a mapping representing the index changes.
    HMMStateNetwork::CleanupResult cleanup(bool cleanupExits = true);

    /// Removes all outputs from the network
    /// Also performs a cleanup, so the search network must already be clean
    /// for indices to stay equal
    void removeOutputs();

    u32 getChecksum() const;

    /// Dump the search network as a dot graph into the given file
    void dumpDotGraph(std::string file, const std::vector<int>& nodeDepths);

    struct Exit {
        Bliss::LemmaPronunciation::Id pronunciation;
        StateId                       transitState;
        bool                          operator==(const Exit& rhs) const {
            return pronunciation == rhs.pronunciation && transitState == rhs.transitState;
        }
        struct Hash {
            u32 operator()(const Exit& exit) const {
                return MyStandardValueHash<u32>()(exit.pronunciation + MyStandardValueHash<u32>()(exit.transitState));
            }
        };
        bool operator<(const Exit& rhs) const {
            return pronunciation < rhs.pronunciation || (pronunciation == rhs.pronunciation && transitState < rhs.transitState);
        }
    };

    bool isRoot(StateId node) const {
        return node == rootState || node == ciRootState || coarticulatedRootStates.count(node);
    }

    typedef std::map<Search::StateId, std::pair<Bliss::Phoneme::Id, Bliss::Phoneme::Id>> RootTransitDescriptions;

    /**  ----- state tree data: ------  */

    // Identity of the main search network
    TreeIndex masterTree;

    // Root node
    StateId rootState;

    // Context-independent root node
    StateId ciRootState;

    // The word-end exits
    std::vector<Exit> exits;

    // The coarticulated root nodes (does not include rootState), including pushed nodes
    std::set<Search::StateId> coarticulatedRootStates;

    // The unpushed coarticulated root nodes (only filled if pushing is used!)
    std::set<Search::StateId> unpushedCoarticulatedRootStates;

    // Nodes in the search network which correspond to pushed word-ends
    std::set<StateId> pushedWordEndNodes;
    // Nodes in the search network which correspond to uncoarticulated physical word-ends,
    // with context-independent right context. May be root nodes as well as normal nodes.
    std::set<StateId> uncoarticulatedWordEndStates;

    // Phoneme transition descriptions for all root nodes (including rootState)
    RootTransitDescriptions rootTransitDescriptions;

    // The network structure (inner states and transitions between states and exits)
    Search::HMMStateNetwork structure;

private:
    std::string archiveEntry() const;

    std::string                        archive_;
    Core::DependencySet                dependencies_;
    Core::Ref<const Am::AcousticModel> acousticModel_;
    Bliss::LexiconRef                  lexicon_;
    Core::Configuration                config_;
    TreeBuilderFactory                 treeBuilderFactory_;

    // Writes the whole state network into the given stream
    void write(Core::MappedArchiveWriter writer);

    // Reads the state network from the given stream.
    //@return Whether the reading was successful.
    bool read(Core::MappedArchiveReader reader);
};
}  // namespace Search
#endif  // STATETREECOMPRESSION_H
