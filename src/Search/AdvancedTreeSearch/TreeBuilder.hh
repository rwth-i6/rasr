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
#ifndef TREEBUILDER_HH
#define TREEBUILDER_HH

#include <Bliss/Phoneme.hh>
#include <Search/StateTree.hh>
#include "Helpers.hh"
#include "LmCache.hh"
#include "PersistentStateTree.hh"

namespace Bliss {
class Lexicon;
}

namespace Am {
class AcousticModel;
};

namespace Core {
class Configuration;
}

class TreeBuilder {
public:
    typedef u32 StateId;

    TreeBuilder(Core::Configuration config, const Bliss::Lexicon& lexicon, const Am::AcousticModel& acousticModel, Search::PersistentStateTree& network, bool initialize = true, bool arcBased = false);
    // Build a new persistent state network.
    void build();
    // Returns a mapping of state-indices. Zero means 'invalid'.
    // If onlyMinimizeBackwards is true, then no forward determinization is performed, but rather only backwards minimization.
    // If allowLost is true, losing states is allowed. Happens if there are unreachable garbage states.
    std::vector<StateId> minimize(bool forceDeterminization = true, bool onlyMinimizeBackwards = false, bool allowLost = false);

    struct HMMSequence {
        HMMSequence()
                : length(0) {}
        enum {
            MaxLength = 12
        };
        s32                                        length;
        inline const Search::StateTree::StateDesc& operator[](u32 index) const {
            return hmm[index];
        }
        Search::StateTree::StateDesc hmm[MaxLength];

        bool operator==(const HMMSequence& rhs) const {
            verify(length < MaxLength);
            return length == rhs.length && std::equal(hmm, hmm + length, rhs.hmm);
        }

        void reverse() {
            for (u32 i = 0; i < length / 2; ++i) {
                Search::StateTree::StateDesc temp(hmm[i]);
                hmm[i]              = hmm[length - 1 - i];
                hmm[length - 1 - i] = temp;
            }
        }

        struct Hash {
            size_t operator()(const HMMSequence& seq) const {
                size_t ret = seq.length;
                for (s32 p = 0; p < seq.length; ++p)
                    ret = StandardValueHash<size_t>()(ret + Search::StateTree::StateDesc::Hash()(seq[p]));
                return ret;
            }
        };
    };

    HMMSequence arcSequence(u32 acousticModelIndex) const;
    std::string arcDesc(u32 acousticModelIndex) const;

    // If this function returns true, then the hmm states are placeholders for hmm sequences which
    // can be acquired through arcSequence(...). The transition model index then contains word boundary information.
    bool arcBased() const {
        return arcBased_;
    }

protected:
    Core::Component::Message log() const;

    struct RootKey {
    public:
        RootKey(Bliss::Phoneme::Id _left = Core::Type<Bliss::Phoneme::Id>::max, Bliss::Phoneme::Id _right = Core::Type<Bliss::Phoneme::Id>::max, int _depth = 0)
                : left(_left),
                  right(_right),
                  depth(_depth),
                  hash(StandardValueHash<Bliss::Phoneme::Id>()(left + StandardValueHash<Bliss::Phoneme::Id>()(right)) + StandardValueHash<Bliss::Phoneme::Id>()(depth)) {
        }

        bool isValid() const {
            return left != Core::Type<Bliss::Phoneme::Id>::max || right != Core::Type<Bliss::Phoneme::Id>::max;
        }

        bool operator==(const RootKey& rhs) const {
            return left == rhs.left && right == rhs.right && depth == rhs.depth;
        }

        struct Hash {
            u32 operator()(const RootKey& key) const {
                return key.hash;
            }
        };

        const Bliss::Phoneme::Id left, right;
        const int                depth;
        const u32                hash;
    };

    struct StatePredecessor {
        StatePredecessor(std::set<StateId> _successors = std::set<StateId>(), Search::StateTree::StateDesc _desc = Search::StateTree::StateDesc(), bool _isWordEnd = false)
                : successors(_successors),
                  desc(_desc),
                  isWordEnd(_isWordEnd),
                  hash(StandardValueHash<Bliss::Phoneme::Id>()(SetHash<StateId>()(successors) + Search::StateTree::StateDesc::Hash()(desc) + (isWordEnd ? 1312 : 0))) {}

        struct Hash {
            u32 operator()(const StatePredecessor& pred) const {
                return pred.hash;
            }
        };

        bool operator==(const StatePredecessor& rhs) const {
            return successors == rhs.successors && desc == rhs.desc && isWordEnd == rhs.isWordEnd;
        }

        const std::set<StateId>            successors;
        const Search::StateTree::StateDesc desc;
        bool                               isWordEnd;
        const u32                          hash;
    };

    void printStats(std::string occasion);
    void buildFanInOutStructure();
    void addCrossWordSkips();
    void skipRootTransitions();
    void propagateExits(StateId state, Search::HMMStateNetwork::ChangePlan change);
    bool isContextDependent(Bliss::Phoneme::Id phone) const;

    Search::StateTree::StateDesc rootDesc() const;

    StateId createSkipRoot(StateId baseRoot);
    StateId createRoot(Bliss::Phoneme::Id left, Bliss::Phoneme::Id right, int depth);
    StateId createState(Search::StateTree::StateDesc desc);
    u32     createExit(Search::PersistentStateTree::Exit exit);
    u32     addExit(StateId                       prePredecessor,
                    StateId                       predecessor,
                    Bliss::Phoneme::Id            leftPhoneme,
                    Bliss::Phoneme::Id            rightPhoneme,
                    int                           depth,
                    Bliss::LemmaPronunciation::Id pron);
    void    hmmFromAllophone(HMMSequence&       ret,
                             Bliss::Phoneme::Id left,
                             Bliss::Phoneme::Id central,
                             Bliss::Phoneme::Id right,
                             u32                boundary         = 0,
                             bool               allowNonStandard = true);

    // Adds the successor as successor of the predecessor, if it isn't in the list yet
    bool                        addSuccessor(StateId predecessor, StateId successor);
    std::pair<StateId, StateId> extendPhone(StateId                                currentState,
                                            u32                                    phoneIndex,
                                            const std::vector<Bliss::Phoneme::Id>& phones,
                                            Bliss::Phoneme::Id                     left  = Bliss::Phoneme::term,
                                            Bliss::Phoneme::Id                     right = Bliss::Phoneme::term);
    StateId                     extendState(StateId predecessor, Search::StateTree::StateDesc desc, RootKey uniqueKey = RootKey());
    StateId                     extendBodyState(StateId state, Bliss::Phoneme::Id first, Bliss::Phoneme::Id second, Search::StateTree::StateDesc desc);
    StateId                     extendFanIn(StateId successor, Search::StateTree::StateDesc desc);
    StateId                     extendFanIn(const std::set<StateId>& successors, Search::StateTree::StateDesc desc);
    void                        minimizeState(StateId state, std::vector<StateId>& minimizeMap);
    void                        minimizeExits(StateId state, const std::vector<u32>& minimizeExitsMap);
    static void                 mapSet(std::set<StateId>& set, const std::vector<StateId>& minimizeMap, bool force);

    std::string describe(std::pair<Bliss::Phoneme::Id, Bliss::Phoneme::Id>);

    const Bliss::Lexicon&        lexicon_;
    const Am::AcousticModel&     acousticModel_;
    Search::PersistentStateTree& network_;
    Core::Configuration          config_;
    s32                          minPhones_;
    bool                         forceExactWordEnds_;
    bool                         keepRoots_;
    bool                         allowCrossWordSkips_;
    bool                         repeatSilence_;
    bool                         reverse_;
    bool                         arcBased_;
    std::set<Bliss::Phoneme::Id> initialPhonemes_, finalPhonemes_;

    // Keys according to which specific states are supposed to be unique
    // Required to omit merging of paths in some critical locations
    Core::HashMap<StateId, RootKey> stateUniqueKeys_;

    typedef Core::HashMap<HMMSequence, u32, HMMSequence::Hash> ArcSequenceHash;
    ArcSequenceHash                                            arcSequencesHash_;
    std::vector<HMMSequence>                                   arcSequences_;
    struct ArcDesc {
        ArcDesc()
                : left(Bliss::Phoneme::term),
                  central(Bliss::Phoneme::term),
                  right(Bliss::Phoneme::term) {
        }
        Bliss::Phoneme::Id left;
        Bliss::Phoneme::Id central;
        Bliss::Phoneme::Id right;
    };
    std::vector<ArcDesc> arcDescs_;

    typedef Core::HashMap<RootKey, StateId, RootKey::Hash> RootHash;
    RootHash                                               roots_;  // Contains roots and joint-states

    typedef Core::HashMap<StateId, StateId> SkipRootsHash;
    SkipRootsHash                           skipRoots_;
    std::set<StateId>                       skipRootSet_;

    typedef Core::HashMap<Search::PersistentStateTree::Exit, u32, Search::PersistentStateTree::Exit::Hash> ExitHash;
    ExitHash                                                                                               exitHash_;

    typedef Core::HashMap<RootKey, std::set<StateId>, RootKey::Hash> CoarticulationJointHash;
    CoarticulationJointHash                                          initialPhoneSuffix_, initialFinalPhoneSuffix_;

    typedef Core::HashMap<StatePredecessor, Search::StateId, StatePredecessor::Hash> PredecessorsHash;
    PredecessorsHash                                                                 predecessors_;
};

#endif
