/** Copyright 2026 RWTH Aachen University. All rights reserved.
 *
 * Licensed under the RWTH ASR License.
 */

#include "NgramLinearSearch.hh"

#include <algorithm>
#include <limits>
#include <numeric>
#include <sstream>
#include <utility>

#include <Core/Assertions.hh>
#include <Core/XmlStream.hh>
#include <Search/TracebackHelper.hh>

namespace Search {

/*
 * =======================
 * === LabelHypothesis ===
 * =======================
 */

NgramLinearSearch::LabelHypothesis::LabelHypothesis()
        : scoringContext(),
          currentToken(Nn::invalidLabelIndex),
          lmHistory(),
          timeframe(0),
          score(0.0),
          acousticScore(0.0),
          trace(Core::ref(new LatticeTrace(0, {0.0, 0.0}, {}))) {
}

std::string NgramLinearSearch::LabelHypothesis::toString() const {
    std::stringstream ss;

    ss << "Score: " << score
       << ", AM score: " << acousticScore
       << ", total score: " << score
       << ", timeframe: " << timeframe
       << ", traceback: ";

    auto traceback = trace->performTraceback();
    for (auto& item : *traceback) {
        if (item.pronunciation && item.pronunciation->lemma()) {
            ss << item.pronunciation->lemma()->symbol() << " ";
        }
    }

    return ss.str();
}

/*
 * ==========================
 * === NgramLinearSearch ===
 * ==========================
 */

const Core::ParameterInt NgramLinearSearch::paramMaxBeamSize(
        "max-beam-size",
        "Maximum number of hypotheses in the search beam.",
        1);

const Core::ParameterFloat NgramLinearSearch::paramScoreThreshold(
        "score-threshold",
        "Prune hypotheses whose score is at least this much worse than the best hypothesis.",
        Core::Type<Score>::max);

const Core::ParameterInt NgramLinearSearch::paramNumHistogramBins(
        "num-histogram-bins",
        "Number of bins for histogram pruning of hypotheses.",
        2);

const Core::ParameterInt NgramLinearSearch::paramBlankLabelIndex(
        "blank-label-index",
        "Index of the blank label in the lexicon. Can also be inferred from lexicon if it has a lemma with `special='blank'`. If not set, the search will not use blank.",
        Nn::invalidLabelIndex);

const Core::ParameterBool NgramLinearSearch::paramLogStatistics(
        "log-statistics",
        "Log statistics about the beam and the timing at the end of every search step.",
        false);

const Core::ParameterBool NgramLinearSearch::paramLogStepwiseStatistics(
        "log-stepwise-statistics",
        "Log statistics about the beam at every search step.",
        false);

NgramLinearSearch::NgramLinearSearch(Core::Configuration const& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          maxBeamSize_(paramMaxBeamSize(config)),
          scoreThreshold_(paramScoreThreshold(config)),
          scoreHistogram_(paramNumHistogramBins(config)),
          blankLabelIndex_(paramBlankLabelIndex(config)),
          logStatistics_(paramLogStatistics(config)),
          logStepwiseStatistics_(paramLogStepwiseStatistics(config)),
          lexicon_(),
          acousticModel_(),
          languageModel_(),
          labelScorer_(),
          debugChannel_(config, "debug"),
          pronunciations_(),
          beam_(),
          newBeam_(),
          seenHistories_(),
          scoringContexts_(),
          currentSearchStep_(0ul),
          finishedSegment_(false),
          initializationTime_(),
          featureProcessingTime_(),
          scoringTime_(),
          numHypsBeforeRecombination_("num-hyps-before-recombination"),
          numHypsAfterRecombination_("num-hyps-after-recombination"),
          numHypsAfterPruning_("num-hyps-after-pruning") {}

Speech::ModelCombination::Mode NgramLinearSearch::requiredModelCombination() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon | Speech::ModelCombination::useAcousticModel | Speech::ModelCombination::useLanguageModel;
}

Am::AcousticModel::Mode NgramLinearSearch::requiredAcousticModel() const {
    return Am::AcousticModel::noEmissions;
}

bool NgramLinearSearch::setModelCombination(Speech::ModelCombination const& modelCombination) {
    lexicon_            = modelCombination.lexicon();
    labelScorer_       = modelCombination.labelScorers()[0];
    acousticModel_      = modelCombination.acousticModel();
    languageModel_      = modelCombination.languageModel();

    auto blankLemma = lexicon_->specialLemma("blank");
    if (blankLemma) {
        if (blankLabelIndex_ == Nn::invalidLabelIndex) {
            blankLabelIndex_ = blankLemma->id();
            log() << "Use blank index " << blankLabelIndex_ << " inferred from lexicon";
        }
        else if (blankLabelIndex_ != static_cast<Nn::LabelIndex>(blankLemma->id())) {
            warning() << "Blank lemma exists in lexicon with id " << blankLemma->id() << " but is overwritten by config parameter with value " << blankLabelIndex_;
        }
    }

    initializePronunciations();

    return true;
}

void NgramLinearSearch::enterSegment(Bliss::SpeechSegment const* segment) {
    if (logStatistics_) {
    	initializationTime_.reset();
    	featureProcessingTime_.reset();
    	scoringTime_.reset();
    	numHypsBeforeRecombination_.clear();
    	numHypsAfterRecombination_.clear();
    	numHypsAfterPruning_.clear();
    }

    initializationTime_.start();

    labelScorer_->reset();

    // Reset beam to a single empty hypothesis
    beam_.clear();
    beam_.push_back(LabelHypothesis());
    beam_.front().scoringContext = labelScorer_->getInitialScoringContext();
    beam_.front().lmHistory = languageModel_->startHistory();

    currentSearchStep_ = 0ul;
    finishedSegment_   = false;

    initializationTime_.stop();
    if (segment != nullptr) {
        languageModel_->setSegment(segment);
        for (auto& hyp : beam_) {
            hyp.lmHistory = languageModel_->startHistory();
        }
    }
}

void NgramLinearSearch::finishSegment() {
    featureProcessingTime_.start();
    labelScorer_->signalNoMoreFeatures();
    featureProcessingTime_.stop();
    decodeManySteps();
    finishedSegment_ = true;
    if (logStatistics_) {
    	logStatistics();
    }
}

void NgramLinearSearch::putFeature(Nn::DataView const& feature) {
    featureProcessingTime_.start();
    labelScorer_->addInput(feature);
    featureProcessingTime_.stop();
}

void NgramLinearSearch::putFeatures(Nn::DataView const& features, size_t nTimesteps) {
    featureProcessingTime_.start();
    labelScorer_->addInputs(features, nTimesteps);
    featureProcessingTime_.stop();
}

Core::Ref<const Traceback> NgramLinearSearch::getCurrentBestTraceback() const {
    return getBestHypothesis().trace->performTraceback();
}

Core::Ref<const LatticeAdaptor> NgramLinearSearch::getCurrentBestWordLattice() const {
    auto& bestHypothesis = getBestHypothesis();
    LatticeTrace endTrace(bestHypothesis.trace, 0, bestHypothesis.trace->time + 1, bestHypothesis.trace->score, {});

    for (size_t hypIdx = 1ul; hypIdx < beam_.size(); ++hypIdx) {
        auto& hyp = beam_[hypIdx];
        auto siblingTrace = Core::ref(new LatticeTrace( hyp.trace, 0, hyp.trace->time, hyp.trace->score, {}));
        endTrace.appendSiblingToChain(siblingTrace);
    }

    return endTrace.buildWordLattice(lexicon_);
}

Core::Ref<const LatticeTrace> NgramLinearSearch::getCurrentBestLatticeTrace() const {
    return getBestHypothesis().trace;
}

// no-op
Core::Ref<const LatticeTrace> NgramLinearSearch::getCommonPrefix() const {
    return Core::ref(new LatticeTrace(0, {0.0, 0.0}, {}));
}

bool NgramLinearSearch::decodeStep() {
    if (finishedSegment_) {
        return false;
    }

    if (beam_.empty()) {
        return false;
    }

    if (logStepwiseStatistics_) {
        clog() << Core::XmlOpen("search-step-stats");
    }

    /*
     * Collect one scoring context per active hypothesis.
     */
    scoringContexts_.clear();
    scoringContexts_.reserve(beam_.size());

    for (auto const& hyp : beam_) {
        scoringContexts_.push_back(hyp.scoringContext);
    }

    scoringTime_.start();
    auto scoreAccessors = labelScorer_->getScoreAccessors(scoringContexts_);
    scoringTime_.stop();

    newBeam_.clear();
    seenHistories_.clear();

    size_t numHypsBeforeRecombination = 0ul;

    //log() << "timestep " << currentSearchStep_;
    for (size_t hypIndex = 0ul; hypIndex < beam_.size(); ++hypIndex) {
        LabelHypothesis const& hyp = beam_[hypIndex];

        auto const& scoreAccessor = scoreAccessors[hypIndex];

        if (!scoreAccessor) {
            continue;
        }

        for (size_t pronIndex = 0ul; pronIndex < pronunciations_.size(); ++pronIndex) {
            Pronunciation const& pron = pronunciations_[pronIndex];

            if (pron.empty()) {
                continue;
            }

            Nn::LabelIndex nextToken = pron.label;

            Nn::TransitionType transitionType = inferTransitionType(hyp.currentToken, nextToken);

            Lm::History newLmHistory = hyp.lmHistory;
            Score lmScore = 0.0;

            // Add LM-score and update LM-history if we predict a new label (and don't have a loop)
            if (not (transitionType == Nn::TransitionType::LABEL_LOOP or transitionType == Nn::TransitionType::BLANK_LOOP)) {
                Bliss::Lemma const* lemma = pron.lemmaPronunciation->lemma();

                // get LM score
                lmScore += languageModel_->score(newLmHistory, pron.st);

                // update the LM history: new history is the new token
                newLmHistory = languageModel_->extendedHistory(newLmHistory, pron.st);
            }

            Score amScore = 0.0;
            Speech::TimeframeIndex timeframe = hyp.timeframe;

            if (labelScorer_->scoresTransition(transitionType)) {
                amScore += (*scoreAccessor)->getScore(transitionType, nextToken);
                timeframe = std::max(timeframe, (*scoreAccessor)->getTime());
            }

            Score newScore = hyp.score + amScore + lmScore;

            //log() << "next token " << nextToken << " amScore " << amScore << " lmScore " << lmScore << "prev token " << hyp.currentToken << " extScore " << newScore;

            ++numHypsBeforeRecombination;

            auto [it, inserted] = seenHistories_.try_emplace(newLmHistory, newBeam_.size());

            if (not inserted) {
                LabelHypothesis const& existingHyp = newBeam_[it->second];

                if (newScore >= existingHyp.score) {
                    continue;
                }
            }

            LabelHypothesis newHyp;
            newHyp.scoringContext = labelScorer_->extendedScoringContext(
                                                hyp.scoringContext,
                                                nextToken,
                                                transitionType);

            newHyp.currentToken    = nextToken;
            newHyp.timeframe       = timeframe;
            newHyp.score           = newScore;
            newHyp.acousticScore   = hyp.acousticScore + amScore;
            newHyp.lmHistory       = newLmHistory;

            Core::Ref<LatticeTrace> predecessor;
            if (transitionType == Nn::TransitionType::LABEL_LOOP or transitionType == Nn::TransitionType::BLANK_LOOP) {
                predecessor = hyp.trace->predecessor;
            }
            else {
                predecessor = hyp.trace;
            }

            newHyp.trace = Core::ref(new LatticeTrace(
                predecessor,
                pron.lemmaPronunciation,
                newHyp.timeframe + 1,
                {newHyp.acousticScore, newHyp.score - newHyp.acousticScore},
                {}));

            if (inserted) {
                newBeam_.push_back(std::move(newHyp));
            }
            else {
                newBeam_[it->second] = std::move(newHyp);
            }
        }
    }

    if (newBeam_.empty()) {
        if (logStepwiseStatistics_) {
            clog() << Core::XmlClose("search-step-stats");
        }
        return false;
    }

    if (logStatistics_) {
    	numHypsBeforeRecombination_ += numHypsBeforeRecombination;

    	if (logStepwiseStatistics_) {
        	clog() << Core::XmlFull("num-hyps-before-recombination", numHypsBeforeRecombination);
    	}

    	// Recombination was already done online while creating newBeam_
    	numHypsAfterRecombination_ += newBeam_.size();

    	if (logStepwiseStatistics_) {
        	clog() << Core::XmlFull("num-hyps-after-recombination", newBeam_.size());
    	}

   		// scorePruning(newBeam_, Core::Type<Score>::max, maxBeamSize_);

   		 numHypsAfterPruning_ += newBeam_.size();

    	if (logStepwiseStatistics_) {
        	clog() << Core::XmlFull("num-hyps-after-pruning", newBeam_.size());
    	}

    }

    beam_.swap(newBeam_);


    if (debugChannel_.isOpen()) {
        std::stringstream ss;

        for (size_t hypIdx = 0ul; hypIdx < beam_.size(); ++hypIdx) {
            ss << "Hypothesis " << hypIdx + 1ul << ": " << beam_[hypIdx].toString() << "\n";
        }

        ss << "\n";
        debugChannel_ << ss.str();
    }

    if (logStepwiseStatistics_) {
        clog() << Core::XmlFull("active-hyps", beam_.size());
        clog() << Core::XmlFull("best-hyp-score", getBestHypothesis().score);
        clog() << Core::XmlFull("worst-hyp-score", getWorstHypothesis().score);
        clog() << Core::XmlClose("search-step-stats");
    }

    ++currentSearchStep_;

    return true;
}


/*
 * ============================================================================
 * Pronunciation initialization
 * ============================================================================
 */

void NgramLinearSearch::initializePronunciations() {
    pronunciations_.clear();

    auto lemmas = lexicon_->lemmaPronunciations();

    for (auto it = lemmas.first; it != lemmas.second; ++it) {
        Bliss::LemmaPronunciation const* lemmaPronunciation = *it;

        Pronunciation pron = createPronunciation(lemmaPronunciation);

        if (!pron.empty()) {
            pronunciations_.push_back(pron);
        }
    }

    log() << "Initialized " << pronunciations_.size() << " linear pronunciations.";
}

NgramLinearSearch::Pronunciation NgramLinearSearch::createPronunciation(Bliss::LemmaPronunciation const* lemmaPronunciation) const {
    Pronunciation result;
    result.lemmaPronunciation = lemmaPronunciation;

    Bliss::Pronunciation const* blissPron = lemmaPronunciation->pronunciation();

    if (!blissPron or blissPron->length() == 0) {
        return result;
    }

    Bliss::Lemma const* lemma = lemmaPronunciation->lemma();
    Bliss::SyntacticTokenSequence const sts = lemma->syntacticTokenSequence();
    result.st = sts.front();


    Am::Phonology const& phonology = *(acousticModel_->phonology());

    require(blissPron->length() == 1);

    for (u32 phone = 0; phone < blissPron->length(); ++phone) {
        u16 boundary = 0;
        if (phone == 0) {
            boundary |= Am::Allophone::isInitialPhone;
        }
        if (phone + 1 == blissPron->length()) {
            boundary |= Am::Allophone::isFinalPhone;
        }
        Am::Allophone allophone(phonology(*blissPron, phone), boundary);
        Am::Allophone const* allo = acousticModel_->allophoneAlphabet()->allophone(allophone);
        verify(allo);

        Am::ClassicHmmTopology const* topology = acousticModel_->hmmTopology((*blissPron)[phone]);
        verify(topology);
        int nPhoneStates = topology->nPhoneStates();
        int nSubStates   = topology->nSubStates();
        //verify(nPhoneStates > 0);
        //verify(nSubStates > 0);
        require(nPhoneStates == 1);
        require(nSubStates == 1);

       // for (int state = 0; state < nPhoneStates; ++state) {
        //    for (int subState = 0; subState < nSubStates; ++subState) {
        //	Am::AllophoneState alloState = acousticModel_->allophoneStateAlphabet()->allophoneState(allo, state);
        	Am::AllophoneState alloState = acousticModel_->allophoneStateAlphabet()->allophoneState(allo, 0);
        	Nn::LabelIndex label = acousticModel_->emissionIndex(alloState);
        	result.label = label;
        //    }
        //}
    }

    return result;
}



NgramLinearSearch::LabelHypothesis const& NgramLinearSearch::getBestHypothesis() const {
    verify(!beam_.empty());

    return *std::min_element(beam_.begin(), beam_.end());
}

NgramLinearSearch::LabelHypothesis const& NgramLinearSearch::getWorstHypothesis() const {
    verify(!beam_.empty());

    return *std::max_element(beam_.begin(), beam_.end());
}

void NgramLinearSearch::logStatistics() const {
    clog() << Core::XmlOpen("timing-statistics") + Core::XmlAttribute("unit", "milliseconds");
    clog() << Core::XmlOpen("initialization-time") << initializationTime_.elapsedMilliseconds() << Core::XmlClose("initialization-time");
    clog() << Core::XmlOpen("feature-processing-time") << featureProcessingTime_.elapsedMilliseconds() << Core::XmlClose("feature-processing-time");
    clog() << Core::XmlOpen("scoring-time") << scoringTime_.elapsedMilliseconds() << Core::XmlClose("scoring-time");
    clog() << Core::XmlClose("timing-statistics");
    numHypsBeforeRecombination_.write(clog());
    numHypsAfterRecombination_.write(clog());
    numHypsAfterPruning_.write(clog());
}

Nn::TransitionType NgramLinearSearch::inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const {
    bool prevIsBlank = prevLabel == blankLabelIndex_;
    bool nextIsBlank = nextLabel == blankLabelIndex_;

    if (prevLabel == Nn::invalidLabelIndex) {
        if (nextIsBlank) {
            return Nn::TransitionType::INITIAL_BLANK;
        }
        else {
            return Nn::TransitionType::INITIAL_LABEL;
        }
    }

    if (prevIsBlank) {
        if (nextIsBlank) {
            return Nn::TransitionType::BLANK_LOOP;
        }
        else {
            return Nn::TransitionType::BLANK_TO_LABEL;
        }
    }
    else {
        if (nextIsBlank) {
            return Nn::TransitionType::LABEL_TO_BLANK;
        }
        else if (prevLabel == nextLabel) {
            return Nn::TransitionType::LABEL_LOOP;
        }
        else {
            return Nn::TransitionType::LABEL_TO_LABEL;
        }
    }
}

template<class Element>
void NgramLinearSearch::scorePruning(std::vector<Element>& hypotheses, Score relativeThreshold, size_t maxBeamSize) {
    if (hypotheses.size() <= maxBeamSize && relativeThreshold == Core::Type<Score>::max) {
        // Neither relative score pruning nor max beam size pruning triggers
        return;
    }

    // Find ranges for score histogram and setting absolute threshold
    Score lowerScore = Core::Type<Score>::max;
    Score upperScore = Core::Type<Score>::min;

    for (auto const& hyp : hypotheses) {
        lowerScore = std::min(lowerScore, hyp.score);
        upperScore = std::max(upperScore, hyp.score);
    }

    if (lowerScore == upperScore) {
        // All scores are the same (usually only happens when exactly 1 hyp is active)
        if (hypotheses.size() > maxBeamSize) {
            hypotheses.resize(maxBeamSize);
        }
        return;
    }

    Score absoluteThreshold = upperScore;

    // Pruning by relative score threshold
    if (relativeThreshold != Core::Type<Score>::max) {
        absoluteThreshold = lowerScore + relativeThreshold;
    }

    // Pruning by max beam size
    if (hypotheses.size() > maxBeamSize) {
        scoreHistogram_.clear();
        scoreHistogram_.setLimits(lowerScore, upperScore);

        for (auto const& hyp : hypotheses) {
            scoreHistogram_ += hyp.score;
        }

        absoluteThreshold = std::min(absoluteThreshold, scoreHistogram_.quantile(maxBeamSize));
    }

    if (absoluteThreshold >= upperScore) {
        // Nothing will be pruned
        return;
    }

    // Remove elements with score > absoluteThreshold
    hypotheses.erase(
            std::remove_if(
                    hypotheses.begin(),
                    hypotheses.end(),
                    [absoluteThreshold](auto const& hyp) { return hyp.score > absoluteThreshold; }),
            hypotheses.end());
}

template void NgramLinearSearch::scorePruning(std::vector<NgramLinearSearch::LabelHypothesis>&, Score, size_t);


}  // namespace Search