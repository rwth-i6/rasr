#include "GreedySearch.hh"
#include <Lattice/Lattice.hh>
#include <Lattice/LatticeAdaptor.hh>
#include <Nn/LabelScorer.hh>
#include <Speech/ModelCombination.hh>
#include <Speech/Types.hh>
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
    hyp_.history   = labelScorer_->getStartHistory();
    hyp_.traceback = Traceback();
    hyp_.score     = Nn::NegLogScore();
    hyp_.labelSeq.clear();
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

void GreedyTimeSyncSearch::finalize() {
    decodeMore();
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
        wordBoundaries->set(currentState->id(), Lattice::WordBoundary(static_cast<Speech::TimeframeIndex>(it->time.endTime())));
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
        result->newArc(currentState, nextState, it->lemma, scores.acoustic(), scores.lm());
        currentState = nextState;
    }

    result->setWordBoundaries(wordBoundaries);
    result->addAcyclicProperty();

    return Core::ref(new Lattice::WordLatticeAdaptor(result));
}

void GreedyTimeSyncSearch::resetStatistics() {}

void GreedyTimeSyncSearch::logStatistics() const {}

bool GreedyTimeSyncSearch::decodeStep() {
    verify(labelScorer_);
    verify(hyp_.history);
    Nn::LabelScorer::ScoreWithTime bestScoreWithTime{Nn::NegLogScore::max(), {}};
    Nn::LabelIndex                 bestIdx            = Core::Type<Nn::LabelIndex>::max;
    const Bliss::Lemma*            bestLemma          = nullptr;
    auto                           bestTransitionType = Nn::LabelScorer::TransitionType::FORWARD;

    // Fetch prev label from hypothesis because this may be expanded with a loop transition
    Nn::LabelIndex prevLabel = Core::Type<Nn::LabelIndex>::max;
    if (not hyp_.labelSeq.empty()) {
        prevLabel = hyp_.labelSeq.back();
    }

    std::optional<Nn::LabelScorer::ScoreWithTime> scoreWithTime;
    // assume the output labels are stored as lexicon lemma orth and ordered consistently with NN output index
    auto lemmas = lexicon_->lemmas();
    for (auto lemmaIt = lemmas.first; lemmaIt != lemmas.second; ++lemmaIt) {
        const Bliss::Lemma* lemma(*lemmaIt);
        verify(lemma->nOrthographicForms() == 1);  // one lemma with exactly one output label as orth
        Nn::LabelIndex idx             = lemma->id();
        auto           transition_type = Nn::LabelScorer::TransitionType::FORWARD;
        if (allowLabelLoop_ and idx == prevLabel) {
            transition_type = Nn::LabelScorer::TransitionType::LOOP;
        }
        scoreWithTime = labelScorer_->getScoreWithTime({hyp_.history, idx, transition_type});
        if (not scoreWithTime.has_value()) {
            return false;
        }

        if (scoreWithTime->score < bestScoreWithTime.score) {
            bestLemma          = lemma;
            bestIdx            = idx;
            bestScoreWithTime  = scoreWithTime.value();
            bestTransitionType = transition_type;
        }
    }

    verify(bestLemma != nullptr);

    labelScorer_->extendHistory({hyp_.history, bestIdx, bestTransitionType});
    hyp_.labelSeq.push_back(bestIdx);
    hyp_.score += bestScoreWithTime.score;

    hyp_.traceback.push_back(TracebackItem(nullptr, bestLemma, bestScoreWithTime.timestamp, ScoreVector(hyp_.score, Nn::NegLogScore())));

    return true;
}

void GreedyTimeSyncSearch::decodeMore() {
    while (decodeStep())
        ;
}

}  // namespace Search
