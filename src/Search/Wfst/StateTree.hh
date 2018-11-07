/** Copyright 2018 RWTH Aachen University. All rights reserved.
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
#ifndef _SEARCH_WFST_STATE_TREE_HH
#define _SEARCH_WFST_STATE_TREE_HH

#include <Am/AcousticModel.hh>
#include <Bliss/Lexicon.hh>
#include <OpenFst/Types.hh>
#include <Search/StateTree.hh>

namespace Search { namespace Wfst {

class StateSequenceList;
class TiedStateSequenceMap;

class StateTreeConverter : public Core::Component
{
    enum CompressionType { CompressionNone, CompressionFactorized, CompressionHmmLabel };
    static const Core::Choice choiceCompression;
    static const Core::ParameterChoice paramCompression;
    static const Core::ParameterBool paramEpsilonArcs;
    static const Core::ParameterBool paramMergeNonTreeArcs;
    static const Core::ParameterBool paramAddDisambiguators;
    static const Core::ParameterBool paramPushWordLabels;
public:
    StateTreeConverter(const Core::Configuration &c,
                       Bliss::LexiconRef lexicon,
                       Core::Ref<const Am::AcousticModel> am);
    virtual ~StateTreeConverter();

    void createFst(OpenFst::VectorFst *fst);
    bool writeStateSequences(const std::string &filename) const;

private:
    typedef std::map<StateTree::StateId, OpenFst::StateId> StateMap;
    typedef std::map<const Bliss::LemmaPronunciation*, u32> WordEndMap;

    OpenFst::StateId getState(OpenFst::VectorFst *fst, StateTree::StateId s, bool final, bool *isNew);
    OpenFst::Label getLabel(StateTree::StateId s, bool initial, bool final);
    OpenFst::Label getOutputLabel(const StateTree::Exit &exit) const;
    OpenFst::Label encodeHmmState(Am::AcousticModel::EmissionIndex emission,
                                  Am::AcousticModel::StateTransitionIndex transition,
                                  bool isInitial, bool isFinal) const;
    void addWordEndArcs(OpenFst::VectorFst *fst,
                        StateTree::SuccessorIterator nextState, bool isInitial,
                        OpenFst::StateId fstState, std::vector<bool> *visited,
                        std::deque<StateTree::StateId> *queue);
    void addWordEndEpsilonArcs(OpenFst::VectorFst *fst, StateTree::StateId state,
                               OpenFst::StateId fstState, std::vector<bool> *visited,
                               std::deque<StateTree::StateId> *queue);
    void mergeArcs(OpenFst::VectorFst *fst) const;
    void statistics(const OpenFst::VectorFst *fst, const std::string &description) const;
    OpenFst::VectorFst* createStateSequenceToEmissionTransducer() const;
    void factorize(OpenFst::VectorFst *fst);
    void convertToHmmLabels(OpenFst::VectorFst *fst);

    void wordEndLabels(StateTree::StateId s, WordEndMap *labels) const;

    Bliss::LexiconRef lexicon_;
    Core::Ref<const Am::AcousticModel> am_;
    bool factorize_, hmmLabels_, wordEndEpsArcs_, mergeNonTreeArcs_;
    bool addDisambiguators_, pushWordLabels_;
    u32 numDisambiguators_;

    TiedStateSequenceMap *labels_;
    StateSequenceList *stateSequences_;
    StateMap stateMap_;
    std::set<OpenFst::StateId> nonTreeStates_;
    StateTree *stateTree_;
};

} // namespace Wfst
} // namespace Search

#endif // _SEARCH_WFST_STATE_TREE_HH
