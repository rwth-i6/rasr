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
#ifndef _SEARCH_EXPANDING_FSA_SEARCH_HH
#define _SEARCH_EXPANDING_FSA_SEARCH_HH

#include <Fsa/Automaton.hh>
#include <Search/Search.hh>
#include <Search/Wfst/Types.hh>
#include <Speech/ModelCombination.hh>

namespace OpenFst {
class LabelMap;
}

namespace Search {
namespace Wfst {

class SearchSpaceBase;
class StateSequenceList;

/**
 * A WFST-based decoder.
 * Supports dynamic expansion of HMM states (therefore "Expanding").
 * The WFST used as search network can be constructed statically or
 * dynamically.
 * This class serves as interface to the Speech::Recognizer object and handles
 * the parameters. The main work is performed in SearchSpace.
 */
class ExpandingFsaSearch : public SearchAlgorithm {
private:
    static const Core::Choice                choiceNetworkType;
    static const Core::ParameterChoice       paramNetworkType_;
    static const Core::Choice                choiceOutputType;
    static const Core::ParameterChoice       paramOutputType_;
    static const Core::Choice                choiceWordEndType;
    static const Core::ParameterChoice       paramWordEndType_;
    static const Core::ParameterString       paramEmissionSequencesFile_;
    static const Core::ParameterString       paramStateSequencesFile_;
    static const Core::ParameterFloat        paramAcousticPruningThreshold_;
    static const Core::ParameterInt          paramAcousticPruningLimit_;
    static const Core::ParameterInt          paramAcousticPruningBins_;
    static const Core::ParameterBool         paramInitialEpsilonPruning_;
    static const Core::ParameterBool         paramEpsilonArcPruning_;
    static const Core::ParameterBool         paramProspectivePruning_;
    static const Core::ParameterFloat        paramLatticePruning_;
    static const Core::ParameterFloat        paramWordEndPruning_;
    static const Core::ParameterBool         paramMergeSilenceArcs_;
    static const Core::ParameterBool         paramMergeEpsilonPaths_;
    static const Core::ParameterInt          paramPurgeInterval;
    static const Core::ParameterBool         paramCreateLattice;
    static const Core::ParameterFloat        paramWeightScale;
    static const Core::ParameterBool         paramAllowSkips;
    static const Core::ParameterString       paramMapOutput;
    static const Core::ParameterBool         paramNonWordOutput;
    static const Core::ParameterStringVector paramNonWordPhones;
    static const Core::ParameterBool         paramHasNonWords;
    static const Core::ParameterBool         paramIgnoreLastOutput;
    static const Core::ParameterBool         paramDetailedStatistics;
    static const Core::Choice                choiceLatticeType;
    static const Core::ParameterChoice       paramLatticeType;

    Bliss::LexiconRef        lexicon_;
    mutable Core::XmlChannel statisticsChannel_;
    mutable Core::XmlChannel memoryInfoChannel_;
    SearchSpaceBase*         searchSpace_;
    bool                     createLattice_;
    OutputType               outputType_;
    OpenFst::LabelMap*       labelMap_;
    StateSequenceList*       stateSequences_;

public:
    ExpandingFsaSearch(const Core::Configuration&);
    virtual ~ExpandingFsaSearch();

    virtual Speech::ModelCombination::Mode modelCombinationNeeded() const;
    virtual bool                           setModelCombination(const Speech::ModelCombination& modelCombination);
    virtual void                           setGrammar(Fsa::ConstAutomatonRef) {}

    virtual void                                    init();
    virtual void                                    restart();
    virtual void                                    setSegment(const std::string& name);
    virtual void                                    feed(const Mm::FeatureScorer::Scorer&);
    virtual void                                    getPartialSentence(Traceback& result);
    virtual void                                    getCurrentBestSentence(Traceback& result) const;
    virtual Core::Ref<const Search::LatticeAdaptor> getCurrentWordLattice() const;
    virtual void                                    resetStatistics();
    virtual void                                    logStatistics() const;

private:
    SearchSpaceBase* createSearchSpace();
    s32              silenceOutput() const;
};

}  // namespace Wfst
}  // namespace Search

#endif /* _SEARCH_EXPANDING_FSA_SEARCH_HH */
