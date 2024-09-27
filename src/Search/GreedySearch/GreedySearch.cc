#include "GreedySearch.hh"
#include <Lattice/Lattice.hh>
#include <Lattice/LatticeAdaptor.hh>
#include <Nn/LabelScorer.hh>
#include <Speech/ModelCombination.hh>
#include <Speech/Types.hh>
#include "Core/Types.hh"
#include "Nn/LabelHistory.hh"
#include "Nn/Types.hh"

namespace Search {

const Core::ParameterBool GreedyTimeSyncSearch::paramUseBlank("use-blank", "Allow any amount of blank transitions between every label output", false);
const Core::ParameterInt  GreedyTimeSyncSearch::paramBlankLabelIndex("blank-label-index", "Index of the blank label in the lexicon. Only necessary if `use-blank` is true.", 0);
const Core::ParameterBool GreedyTimeSyncSearch::paramAllowLabelLoop("allow-label-loop", "Allow repetition of a label", false);

GreedyTimeSyncSearch::GreedyTimeSyncSearch(const Core::Configuration& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          useBlank_(paramUseBlank(config)),
          allowLabelLoop_(paramAllowLabelLoop(config)),
          blankLabelIndex_(paramBlankLabelIndex(config)),
          labelScorer_(),
          numClasses_(0ul),
          hyp_() {
}

void GreedyTimeSyncSearch::reset() {
    verify(labelScorer_);
    labelScorer_->reset();
    hyp_.reset();
    hyp_.history = labelScorer_->getStartHistory();
}

Speech::ModelCombination::Mode GreedyTimeSyncSearch::modelCombinationNeeded() const {
    return Speech::ModelCombination::useLabelScorer | Speech::ModelCombination::useLexicon;
}

bool GreedyTimeSyncSearch::setModelCombination(const Speech::ModelCombination& modelCombination) {
    lexicon_     = modelCombination.lexicon();
    labelScorer_ = modelCombination.labelScorer();

    reset();
    return true;
}

void GreedyTimeSyncSearch::enterSegment() {
    verify(labelScorer_);
    labelScorer_->reset();
}

void GreedyTimeSyncSearch::enterSegment(Bliss::SpeechSegment const*) {
    verify(labelScorer_);
    labelScorer_->reset();
}

void GreedyTimeSyncSearch::finishSegment() {
    verify(labelScorer_);
    labelScorer_->signalNoMoreFeatures();
    decodeMore();
}

void GreedyTimeSyncSearch::addFeature(Nn::FeatureVectorRef feature) {
    verify(labelScorer_);
    labelScorer_->addInput(feature);
}

void GreedyTimeSyncSearch::addFeature(Core::Ref<const Speech::Feature> feature) {
    verify(labelScorer_);
    labelScorer_->addInput(feature);
}

Core::Ref<const SearchAlgorithmV2::Traceback> GreedyTimeSyncSearch::getCurrentBestTraceback() const {
    return Core::ref(new Traceback(hyp_.traceback));
}

Core::Ref<const LatticeAdaptor> GreedyTimeSyncSearch::getCurrentBestWordLattice() const {
    if (hyp_.traceback.empty()) {
        return Core::ref(new Lattice::WordLatticeAdaptor());
    }

    // use default LemmaAlphabet mode of StandardWordLattice
    Core::Ref<Lattice::StandardWordLattice> result(new Lattice::StandardWordLattice(lexicon_));
    Core::Ref<Lattice::WordBoundaries>      wordBoundaries(new Lattice::WordBoundaries);

    // create a linear lattice from the traceback
    Fsa::State* currentState = result->initialState();
    for (auto it = hyp_.traceback.begin(); it != hyp_.traceback.end(); ++it) {
        // wordBoundaries->set(currentState->id(), Lattice::WordBoundary(static_cast<Speech::TimeframeIndex>(it->time.endTime())));
        wordBoundaries->set(currentState->id(), Lattice::WordBoundary(it->time));
        Fsa::State* nextState;
        if (std::next(it) == hyp_.traceback.end()) {
            nextState = result->finalState();
        }
        else {
            nextState = result->newState();
        }
        ScoreVector scores = it->scores;
        if (it != hyp_.traceback.begin()) {
            scores -= std::prev(it)->scores;
        }
        result->newArc(currentState, nextState, it->lemma, scores.acoustic, scores.lm);
        currentState = nextState;
    }

    result->setWordBoundaries(wordBoundaries);
    result->addAcyclicProperty();

    return Core::ref(new Lattice::WordLatticeAdaptor(result));
}

void GreedyTimeSyncSearch::resetStatistics() {}

void GreedyTimeSyncSearch::logStatistics() const {}

Nn::LabelScorer::TransitionType GreedyTimeSyncSearch::inferTransitionType(Nn::LabelIndex prevLabel, Nn::LabelIndex nextLabel) const {
    bool prevIsBlank = (prevLabel == blankLabelIndex_);
    bool nextIsBlank = (nextLabel == blankLabelIndex_);

    if (prevIsBlank) {
        if (nextIsBlank) {
            return Nn::LabelScorer::TransitionType::BLANK_LOOP;
        }
        else {
            return Nn::LabelScorer::TransitionType::BLANK_TO_LABEL;
        }
    }
    else {
        if (nextIsBlank) {
            return Nn::LabelScorer::TransitionType::LABEL_TO_BLANK;
        }
        else if (allowLabelLoop_ and prevLabel == nextLabel) {
            return Nn::LabelScorer::TransitionType::LABEL_LOOP;
        }
        else {
            return Nn::LabelScorer::TransitionType::LABEL_TO_LABEL;
        }
    }
}

void GreedyTimeSyncSearch::LabelHypothesis::reset() {
    history      = Core::Ref<Nn::LabelHistory>();
    currentLabel = Core::Type<Nn::LabelIndex>::max;
    score        = 0.0f;
    traceback.clear();
}

void GreedyTimeSyncSearch::LabelHypothesis::extend(const HypothesisExtension& extension, Core::Ref<Nn::LabelScorer> labelScorer) {
    labelScorer->extendHistory({history, extension.label, extension.transitionType});
    score += extension.score;
    currentLabel = extension.label;
    switch (extension.transitionType) {
        case Nn::LabelScorer::LABEL_TO_LABEL:
        case Nn::LabelScorer::LABEL_TO_BLANK:
        case Nn::LabelScorer::BLANK_TO_LABEL:
            traceback.push_back(TracebackItem(nullptr, extension.lemma, extension.timestep, ScoreVector(score, {})));
            break;
        case Nn::LabelScorer::LABEL_LOOP:
        case Nn::LabelScorer::BLANK_LOOP:
            if (not traceback.empty()) {
                traceback.back().scores.acoustic = score;
            }
            break;
    }
}

bool GreedyTimeSyncSearch::decodeStep() {
    verify(labelScorer_);
    verify(hyp_.history);

    HypothesisExtension bestExtension;

    // Fetch prev label from hypothesis because this may be expanded with a loop transition
    Nn::LabelIndex prevLabel = hyp_.currentLabel;

    // assume the output labels are stored as lexicon lemma orth and ordered consistently with NN output index
    auto lemmas = lexicon_->lemmas();
    for (auto lemmaIt = lemmas.first; lemmaIt != lemmas.second; ++lemmaIt) {
        const Bliss::Lemma* lemma(*lemmaIt);
        Nn::LabelIndex      idx = lemma->id();

        auto transitionType = inferTransitionType(prevLabel, idx);

        auto scoreWithTime = labelScorer_->getScoreWithTime({hyp_.history, idx, transitionType});
        if (not scoreWithTime.has_value()) {
            return false;
        }

        if (scoreWithTime->first < bestExtension.score) {
            bestExtension = {lemma, idx, scoreWithTime->first, scoreWithTime->second, transitionType};
        }
    }

    hyp_.extend(bestExtension, labelScorer_);

    return true;
}

}  // namespace Search
