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

class AbstractTreeBuilder : public Core::Component {
public:
    typedef u32 StateId;

    AbstractTreeBuilder(Core::Configuration config, const Bliss::Lexicon& lexicon, const Am::AcousticModel& acousticModel, Search::PersistentStateTree& network);
    virtual ~AbstractTreeBuilder() = default;

    virtual std::unique_ptr<AbstractTreeBuilder> newInstance(Core::Configuration config, const Bliss::Lexicon& lexicon, const Am::AcousticModel& acousticModel, Search::PersistentStateTree& network, bool initialize = true) = 0;

    // Build a new persistent state network.
    virtual void build() = 0;

protected:
    const Bliss::Lexicon&        lexicon_;
    const Am::AcousticModel&     acousticModel_;
    Search::PersistentStateTree& network_;
};

class MinimizedTreeBuilder : public AbstractTreeBuilder {
public:
    static const Core::ParameterInt  paramMinPhones;
    static const Core::ParameterBool paramAddCiTransitions;
    static const Core::ParameterBool paramUseRootForCiExits;
    static const Core::ParameterBool paramForceExactWordEnds;
    static const Core::ParameterBool paramKeepRoots;
    static const Core::ParameterBool paramAllowCrossWordSkips;
    static const Core::ParameterBool paramRepeatSilence;
    static const Core::ParameterInt  paramMinimizeIterations;
    typedef u32 StateId;

    MinimizedTreeBuilder(Core::Configuration config, const Bliss::Lexicon& lexicon, const Am::AcousticModel& acousticModel, Search::PersistentStateTree& network, bool initialize = true);
    virtual ~MinimizedTreeBuilder() = default;

    virtual std::unique_ptr<AbstractTreeBuilder> newInstance(Core::Configuration config, const Bliss::Lexicon& lexicon, const Am::AcousticModel& acousticModel, Search::PersistentStateTree& network, bool initialize = true);

    virtual void build();

protected:
    struct HMMSequence {
        HMMSequence()
                : length(0) {}
        enum {
            MaxLength = 12
        };

        s32                          length;
        Search::StateTree::StateDesc hmm[MaxLength];

        inline const Search::StateTree::StateDesc& operator[](u32 index) const {
            return hmm[index];
        }

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

        bool operator==(const StatePredecessor& rhs) const {
            return successors == rhs.successors && desc == rhs.desc && isWordEnd == rhs.isWordEnd;
        }

        struct Hash {
            u32 operator()(const StatePredecessor& pred) const {
                return pred.hash;
            }
        };

        const std::set<StateId>            successors;
        const Search::StateTree::StateDesc desc;
        bool                               isWordEnd;
        const u32                          hash;
    };

    typedef std::set<Bliss::Phoneme::Id>                                                                   PhonemeIdSet;
    typedef Core::HashMap<RootKey, StateId, RootKey::Hash>                                                 RootHash;
    typedef Core::HashMap<StateId, StateId>                                                                SkipRootsHash;
    typedef Core::HashMap<Search::PersistentStateTree::Exit, u32, Search::PersistentStateTree::Exit::Hash> ExitHash;
    typedef Core::HashMap<RootKey, std::set<StateId>, RootKey::Hash>                                       CoarticulationJointHash;
    typedef Core::HashMap<StatePredecessor, Search::StateId, StatePredecessor::Hash>                       PredecessorsHash;

    s32  minPhones_;
    bool addCiTransitions_;
    bool useRootForCiExits_;
    bool forceExactWordEnds_;
    bool keepRoots_;
    bool allowCrossWordSkips_;
    bool repeatSilence_;
    u32  minimizeIterations_;
    bool reverse_;

    PhonemeIdSet initialPhonemes_;
    PhonemeIdSet finalPhonemes_;

    // Keys according to which specific states are supposed to be unique
    // Required to omit merging of paths in some critical locations
    Core::HashMap<StateId, RootKey> stateUniqueKeys_;

    RootHash                roots_;  // Contains roots and joint-states
    SkipRootsHash           skipRoots_;
    std::set<StateId>       skipRootSet_;
    ExitHash                exitHash_;
    CoarticulationJointHash initialPhoneSuffix_;
    CoarticulationJointHash initialFinalPhoneSuffix_;
    PredecessorsHash        predecessors_;

    void        printStats(std::string occasion);
    std::string describe(std::pair<Bliss::Phoneme::Id, Bliss::Phoneme::Id>);

    bool isContextDependent(Bliss::Phoneme::Id phone) const;

    void buildBody();
    void buildFanInOutStructure();
    void addCrossWordSkips();
    void skipRootTransitions(StateId start = 1);

    Search::StateTree::StateDesc rootDesc() const;

    StateId createSkipRoot(StateId baseRoot);
    StateId createRoot(Bliss::Phoneme::Id left, Bliss::Phoneme::Id right, int depth);
    StateId createState(Search::StateTree::StateDesc desc);
    u32     createExit(Search::PersistentStateTree::Exit exit);
    u32     addExit(StateId                       predecessor,
                    Bliss::Phoneme::Id            leftPhoneme,
                    Bliss::Phoneme::Id            rightPhoneme,
                    int                           depth,
                    Bliss::LemmaPronunciation::Id pron);

    void hmmFromAllophone(HMMSequence&       ret,
                          Bliss::Phoneme::Id left,
                          Bliss::Phoneme::Id central,
                          Bliss::Phoneme::Id right,
                          u32                boundary = 0);

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

    // Returns a mapping of state-indices. Zero means 'invalid'.
    // If onlyMinimizeBackwards is true, then no forward determinization is performed, but rather only backwards minimization.
    // If allowLost is true, losing states is allowed. Happens if there are unreachable garbage states.
    std::vector<StateId> minimize(bool forceDeterminization = true, bool onlyMinimizeBackwards = false, bool allowLost = false);
    void                 minimizeState(StateId state, std::vector<StateId>& minimizeMap);
    void                 minimizeExits(StateId state, const std::vector<u32>& minimizeExitsMap);
    static void          mapSet(std::set<StateId>& set, const std::vector<StateId>& minimizeMap, bool force);

    void updateHashFromMap(const std::vector<StateId>& map, const std::vector<u32>& exitMap);
    void mapCoarticulationJointHash(CoarticulationJointHash& hash, const std::vector<StateId>& map, const std::vector<u32>& exitMap);
    void mapSuccessors(const std::set<StateId>&, std::set<StateId>&, const std::vector<StateId>&, const std::vector<u32>&);
};

#endif
