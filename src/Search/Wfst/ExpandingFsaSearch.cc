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
#include <Core/MemoryInfo.hh>
#include <Lattice/LatticeAdaptor.hh>
#include <OpenFst/LabelMap.hh>
#include <OpenFst/SymbolTable.hh>
#include <Search/Wfst/ExpandingFsaSearch.hh>
#include <Search/Wfst/LatticeAdaptor.hh>
#include <Search/Wfst/LatticeGenerator.hh>
#include <Search/Wfst/Network.hh>
#include <Search/Wfst/NonWordTokens.hh>
#include <Search/Wfst/SearchSpace.hh>
#include <Search/Wfst/Traceback.hh>
#include <Search/Wfst/WordEnd.hh>

using namespace Search::Wfst;

/******************************************************************************/

const Core::Choice ExpandingFsaSearch::choiceNetworkType(
        "static", NetworkTypeStatic,
        "compressed", NetworkTypeCompressed,
        "dynamic", NetworkTypeStatic, /* @deprecated */
        "composed", NetworkTypeComposed,
        "lattice", NetworkTypeLattice,
        Core::Choice::endMark());
const Core::ParameterChoice ExpandingFsaSearch::paramNetworkType_(
        "network-type", &choiceNetworkType, "type of network", NetworkTypeStatic);
const Core::Choice ExpandingFsaSearch::choiceOutputType(
        "lemma-pronunciation", OutputLemmaPronunciation,
        "lemma", OutputLemma,
        "syntactic-token", OutputSyntacticToken,
        Core::Choice::endMark());
const Core::ParameterChoice ExpandingFsaSearch::paramOutputType_(
        "output-type", &choiceOutputType, "type of output labels in the search network", OutputLemmaPronunciation);
const Core::Choice ExpandingFsaSearch::choiceWordEndType(
        "hmm-flag", WordEndDetector::WordEndHmm,
        "output-label", WordEndDetector::WordEndOutput,
        Core::Choice::endMark());
const Core::ParameterChoice ExpandingFsaSearch::paramWordEndType_(
        "word-end-type", &choiceWordEndType, "method for word end detection",
        WordEndDetector::WordEndHmm);
const Core::ParameterString ExpandingFsaSearch::paramEmissionSequencesFile_(
        "emission-sequences-file", "file name of the emission sequences (deprecated)", "");
const Core::ParameterString ExpandingFsaSearch::paramStateSequencesFile_(
        "state-sequences", "file name of the state sequences", "");
const Core::ParameterFloat ExpandingFsaSearch::paramAcousticPruningThreshold_(
        "acoustic-pruning", "threshold for pruning of state hypotheses",
        1000.0, 0.0);
const Core::ParameterInt ExpandingFsaSearch::paramAcousticPruningLimit_(
        "acoustic-pruning-limit", "maximum number of state hypotheses", Core::Type<s32>::max, 1);
const Core::ParameterInt ExpandingFsaSearch::paramAcousticPruningBins_(
        "acoustic-pruning-bins", "number of bins for histogram pruning of states", 100, 2);
const Core::ParameterBool ExpandingFsaSearch::paramInitialEpsilonPruning_(
        "initial-epsilon-pruning",
        "prune epsilon arcs at segment begin using anticipated pruning"
        "useful for networks with large amounts of epsilon arcs and long epsilon paths",
        false);
const Core::ParameterBool ExpandingFsaSearch::paramEpsilonArcPruning_(
        "epsilon-arc-pruning",
        "prune epsilon arc hypotheses relative to the current best hypothesis", true);
const Core::ParameterBool ExpandingFsaSearch::paramProspectivePruning_(
        "prospective-pruning",
        "prune hypotheses already before acoustic score computations", true);
const Core::ParameterFloat ExpandingFsaSearch::paramLatticePruning_(
        "lattice-pruning",
        "pruning of lattice arcs relative to the shortest path",
        Core::Type<Score>::max, 0.0);
const Core::ParameterFloat ExpandingFsaSearch::paramWordEndPruning_(
        "word-end-pruning",
        "word of word end hypotheses",
        Core::Type<Score>::max, 0.0);
const Core::ParameterBool ExpandingFsaSearch::paramMergeSilenceArcs_(
        "merge-silence-arcs",
        "merge consecutive silence arcs in lattice", true);
const Core::ParameterBool ExpandingFsaSearch::paramMergeEpsilonPaths_(
        "merge-epsilon-paths",
        "re-combine epsilon paths as early as possible"
        "useful for networks with large amounts of epsilon arcs and long epsilon paths",
        false);
const Core::ParameterInt ExpandingFsaSearch::paramPurgeInterval(
        "purge-interval", "number of time frames between purging the book keeping array", 50, 0);
const Core::ParameterBool ExpandingFsaSearch::paramCreateLattice(
        "create-lattice", "enable generation of word lattice", false);
const Core::ParameterFloat ExpandingFsaSearch::paramWeightScale(
        "weight-scale", "scaling applied to network arc weights", 1.0);
const Core::ParameterBool ExpandingFsaSearch::paramAllowSkips(
        "allow-skips", "allow skip transitions between HMM states", true);
const Core::ParameterString ExpandingFsaSearch::paramMapOutput(
        "map-output", "output label mapping");
const Core::ParameterBool ExpandingFsaSearch::paramNonWordOutput(
        "nonword-output", "non-word tokens have output labels in search graph", true);
const Core::ParameterStringVector ExpandingFsaSearch::paramNonWordPhones(
        "nonword-phones", "list of non-word phones (used with non-word-output=true)", ",");
const Core::ParameterBool ExpandingFsaSearch::paramHasNonWords(
        "has-non-words", "network has non-word input labels without corresponding output", false);
const Core::ParameterBool ExpandingFsaSearch::paramIgnoreLastOutput(
        "ignore-last-output", "ignore last output token in the traceback"
                              "required if C's sequence end symbol != epsilon and disambiguators are not exploited",
        false);
const Core::ParameterBool ExpandingFsaSearch::paramDetailedStatistics(
        "detailed-statistics", "compute (computationally expensive) search space statistics", false);
const Core::Choice ExpandingFsaSearch::choiceLatticeType(
        "hmm", LatticeTraceRecorder::HmmLattice,
        "det-hmm", LatticeTraceRecorder::DetermisticHmmLattice,
        "simple-word", LatticeTraceRecorder::SimpleWordLattice,
        "simple-word-nondet", LatticeTraceRecorder::SimpleNonDetWordLattice,
        "word", LatticeTraceRecorder::WordLattice,
        Core::Choice::endMark());
const Core::ParameterChoice ExpandingFsaSearch::paramLatticeType(
        "lattice-type", &choiceLatticeType, "type of generated lattices", LatticeTraceRecorder::HmmLattice);

ExpandingFsaSearch::ExpandingFsaSearch(const Core::Configuration& c)
        : Core::Component(c), SearchAlgorithm(c), statisticsChannel_(c, "statistics"), memoryInfoChannel_(c, "memory-info"), searchSpace_(0), createLattice_(paramCreateLattice(config)), labelMap_(0), stateSequences_(0) {
    searchSpace_ = createSearchSpace();
    outputType_  = static_cast<OutputType>(paramOutputType_(config));
    log("output type: %s", choiceOutputType[outputType_].c_str());
    const std::string outputMap = paramMapOutput(config);
    if (!outputMap.empty()) {
        log("using output map: %s", outputMap.c_str());
        labelMap_ = new OpenFst::LabelMap();
        if (!labelMap_->load(outputMap)) {
            error("cannot load output map");
        }
    }
}

SearchSpaceBase* ExpandingFsaSearch::createSearchSpace() {
    NetworkType networkType = static_cast<NetworkType>(paramNetworkType_(config));
    const bool  allowSkips  = paramAllowSkips(config);
    log("HMM skips: %s", (allowSkips ? "true" : "false"));
    Core::Timer timer;
    timer.start();
    SearchSpaceBase* result = SearchSpaceBase::create(networkType, allowSkips, config);
    result->setPruningThreshold(paramAcousticPruningThreshold_(config));
    log("using acoustic pruning threshold %0.2f", paramAcousticPruningThreshold_(config));
    result->setPruningLimit(paramAcousticPruningLimit_(config));
    result->setPruningBins(paramAcousticPruningBins_(config));
    log("using acoustic pruning limit %d using %d bins",
        paramAcousticPruningLimit_(config), paramAcousticPruningBins_(config));
    result->setInitialEpsilonPruning(paramInitialEpsilonPruning_(config));
    if (paramInitialEpsilonPruning_(config))
        log("using initial epsilon pruning");
    result->setEpsilonPruning(paramEpsilonArcPruning_(config));
    if (paramEpsilonArcPruning_(config))
        log("using epsilon arc pruning");
    result->setProspectivePruning(paramProspectivePruning_(config));
    if (paramProspectivePruning_(config))
        log("using prospective pruning");
    result->setLatticePruning(paramLatticePruning_(config));
    log("using lattice pruning threshold %0.2f", paramLatticePruning_(config));
    if (!Core::isAlmostEqual(paramWordEndPruning_(config), paramWordEndPruning_.defaultValue(), 0.1)) {
        result->setWordEndPruning(true, paramWordEndPruning_(config));
        log("using word end pruning. threshold %0.2f", paramWordEndPruning_(config));
    }
    result->setMergeSilenceLatticeArcs(paramMergeSilenceArcs_(config));
    result->setMergeEpsilonPaths(paramMergeEpsilonPaths_(config));
    result->setCreateLattice(paramCreateLattice(config),
                             static_cast<LatticeTraceRecorder::LatticeType>(paramLatticeType(config)));
    result->setPurgeInterval(paramPurgeInterval(config));
    result->setWeightScale(paramWeightScale(config));
    log("arc weight scale: %0.2f", paramWeightScale(config));
    result->setWordEndType(static_cast<WordEndDetector::WordEndType>(paramWordEndType_(config)));
    result->setIgnoreLastOutput(paramIgnoreLastOutput(config));

    std::string statesFile = paramStateSequencesFile_(config);
    if (statesFile.empty()) {
        statesFile = paramEmissionSequencesFile_(config);
        warning("using deprecated parameter %s", paramEmissionSequencesFile_.name().c_str());
    }
    log("reading state sequences from '%s'", statesFile.c_str());
    stateSequences_ = new StateSequenceList();
    if (!stateSequences_->read(statesFile))
        error("cannot read state sequence file from '%s'", statesFile.c_str());
    else
        log("# state sequences: %d", u32(stateSequences_->size()));
    result->setStateSequences(stateSequences_);

    result->setStatistics(paramDetailedStatistics(config));
    timer.stop();
    if (statisticsChannel_.isOpen()) {
        statisticsChannel_ << Core::XmlOpen("search-space-preparation");
        statisticsChannel_ << timer;
        statisticsChannel_ << Core::XmlClose("search-space-preparation");
    }
    return result;
}

ExpandingFsaSearch::~ExpandingFsaSearch() {
    if (statisticsChannel_.isOpen()) {
        SearchSpaceBase::MemoryUsage m = searchSpace_->memoryUsage();
        statisticsChannel_ << Core::XmlOpen("memory-usage")
                           << Core::XmlFull("bookkeeping", m.bookkeeping)
                           << Core::XmlFull("state-sequences", m.stateSequences)
                           << Core::XmlFull("states", m.states)
                           << Core::XmlFull("arcs", m.arcs)
                           << Core::XmlFull("epsilon-arcs", m.epsilonArcs)
                           << Core::XmlFull("state-hypotheses", m.stateHyps)
                           << Core::XmlFull("arc-hypotheses", m.arcHyps)
                           << Core::XmlFull("hmm-state-hypotheses", m.hmmStateHyps)
                           << Core::XmlFull("total", m.sum())
                           << Core::XmlClose("memory-usage");
    }
    delete searchSpace_;
    delete labelMap_;
}

void ExpandingFsaSearch::restart() {
    if (memoryInfoChannel_.isOpen()) {
        Core::MemoryInfo meminfo;
        memoryInfoChannel_ << meminfo;
    }
    searchSpace_->reset();
}

void ExpandingFsaSearch::setSegment(const std::string& name) {
    searchSpace_->setSegment(name);
}

void ExpandingFsaSearch::feed(const Mm::FeatureScorer::Scorer& scorer) {
    searchSpace_->feed(scorer);
}

void ExpandingFsaSearch::getPartialSentence(Traceback& result) {
}

void ExpandingFsaSearch::getCurrentBestSentence(Traceback& result) const {
    BestPath path;
    searchSpace_->getTraceback(&path);
    if (path.empty()) {
        error("no word end found. empty traceback");
    }
    else {
        path.getTraceback(lexicon_, outputType_, labelMap_, &result);
    }
}

Core::Ref<const Search::LatticeAdaptor> ExpandingFsaSearch::getCurrentWordLattice() const {
    if (!createLattice_)
        return Core::ref(new WfstLatticeAdaptor);
    return Core::ref(new WfstLatticeAdaptor(searchSpace_->createLattice(outputType_)));
}

void ExpandingFsaSearch::resetStatistics() {
    searchSpace_->resetStatistics();
}

void ExpandingFsaSearch::logStatistics() const {
    if (statisticsChannel_.isOpen()) {
        statisticsChannel_ << Core::XmlOpen("search-space-statistics");
        searchSpace_->logStatistics(statisticsChannel_);
        statisticsChannel_ << Core::XmlClose("search-space-statistics");
    }
}

Speech::ModelCombination::Mode ExpandingFsaSearch::modelCombinationNeeded() const {
    return Speech::ModelCombination::useLexicon | Speech::ModelCombination::useAcousticModel;
}

bool ExpandingFsaSearch::setModelCombination(const Speech::ModelCombination& modelCombination) {
    lexicon_ = modelCombination.lexicon();
    searchSpace_->setLexicon(lexicon_);
    searchSpace_->setTransitionModel(modelCombination.acousticModel());
    if (paramHasNonWords(config)) {
        NonWordTokens nonWordTokens(select("non-word-tokens"), *lexicon_);
        nonWordTokens.init();
        u32 nNonWordModels = nonWordTokens.phones().size();
        log("assuming last %d state sequences are non-word models", nNonWordModels);
        searchSpace_->setUseNonWordModels(nNonWordModels);
    }

    if (!paramNonWordOutput(config)) {
        if (!searchSpace_->setNonWordPhones(modelCombination.acousticModel(), paramNonWordPhones(config))) {
            error("cannot set non-word phones");
            return false;
        }
        else {
            log("%d non-word tokens without output", int(paramNonWordPhones(config).size()));
        }
    }
    verify(stateSequences_);
    StateSequenceResolver resolver(modelCombination.acousticModel(), *stateSequences_);
    searchSpace_->setSilence(resolver.findSilence(lexicon_), silenceOutput());
    return true;
}

void ExpandingFsaSearch::init() {
    std::string errorMsg;
    require(searchSpace_);
    require(lexicon_);  // make sure setModelCombination was called before
    if (!searchSpace_->init(errorMsg)) {
        error() << errorMsg;
    }
}

s32 ExpandingFsaSearch::silenceOutput() const {
    OpenFst::Label      label = OpenFst::InvalidLabelId;
    const Bliss::Lemma* lemma = lexicon_->specialLemma("silence");
    if (outputType_ == OutputLemma) {
        label = OpenFst::convertLabelFromFsa(lemma->id());
    }
    else if (outputType_ == OutputLemmaPronunciation) {
        verify_eq(lemma->nPronunciations(), 1);
        label = OpenFst::convertLabelFromFsa(lemma->pronunciations().first->id());
    }
    return label;
}
