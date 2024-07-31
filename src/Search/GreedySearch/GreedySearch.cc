#include "GreedySearch.hh"
#include <algorithm>
#include <fstream>
#include <ios>

namespace Search {

const Core::ParameterBool   GreedyTimeSyncSearch::paramUseBlank("use-blank", "Allow any amount of blank transitions between every label output", false);
const Core::ParameterInt    GreedyTimeSyncSearch::paramBlankLabelIndex("blank-label-index", "Index of the blank label in the vocab file. Only necessary if `use-blank` is true.", 0);
const Core::ParameterBool   GreedyTimeSyncSearch::paramAllowLabelLoop("allow-label-loop", "Allow repetition of a label", false);
const Core::ParameterString GreedyTimeSyncSearch::paramVocabFile("", "Text file where each line contains a string token and a label index integer representing the corresponding output index of the label scorer");

GreedyTimeSyncSearch::GreedyTimeSyncSearch(const Core::Configuration& config)
        : Core::Component(config),
          SearchAlgorithmV2(config),
          useBlank_(paramUseBlank(config)),
          allowLabelLoop_(paramAllowLabelLoop(config)),
          blankLabelIndex_(paramBlankLabelIndex(config)),
          labelScorer_(Nn::Module::instance().createLabelScorer(config)),
          numClasses_(0ul),
          vocabMap_(),
          hyp_(),
          currentStep_(0),
          previousPassedTraceback_(new Traceback()) {
    parseVocabFile(paramVocabFile(config));
}

void GreedyTimeSyncSearch::reset() {
    labelScorer_->reset();
    hyp_.history   = labelScorer_->getStartHistory();
    hyp_.traceback = Traceback();
    hyp_.score     = 0.0;
    hyp_.labelSeq.clear();
}
void GreedyTimeSyncSearch::enterSegment() {
    labelScorer_->reset();
}

void GreedyTimeSyncSearch::enterSegment(Bliss::SpeechSegment const*) {
    labelScorer_->reset();
}

void GreedyTimeSyncSearch::finishSegment() {
    labelScorer_->signalSegmentEnd();
    decodeMore();
}

void GreedyTimeSyncSearch::finalize() {
    decodeMore();
}

void GreedyTimeSyncSearch::addFeature(Core::Ref<const Speech::Feature> feature) {
    labelScorer_->addInput(feature);
}

Core::Ref<const SearchAlgorithmV2::Traceback> GreedyTimeSyncSearch::stablePartialTraceback() {
    previousPassedTraceback_ = Core::ref(new Traceback(hyp_.traceback));
    return previousPassedTraceback_;
}

Core::Ref<const SearchAlgorithmV2::Traceback> GreedyTimeSyncSearch::recentStablePartialTraceback() {
    auto newTraceback = Core::ref(new Traceback(hyp_.traceback));
    newTraceback->erase(newTraceback->begin(), newTraceback->begin() + previousPassedTraceback_->size());
    previousPassedTraceback_ = Core::ref(new Traceback(hyp_.traceback));
    return newTraceback;
}

Core::Ref<const SearchAlgorithmV2::Traceback> GreedyTimeSyncSearch::unstablePartialTraceback() const {
    return Core::ref(new Traceback());
}

Core::Ref<const SearchAlgorithmV2::Traceback> GreedyTimeSyncSearch::getCurrentBestTraceback() const {
    return Core::ref(new Traceback(hyp_.traceback));
}

Core::Ref<const LatticeAdaptor> GreedyTimeSyncSearch::getPartialWordLattice() {
    return {};
}

Core::Ref<const LatticeAdaptor> GreedyTimeSyncSearch::getCurrentBestWordLattice() const {
    return {};
}

void GreedyTimeSyncSearch::resetStatistics() {}

void GreedyTimeSyncSearch::logStatistics() const {}

bool GreedyTimeSyncSearch::decodeStep() {
    Score          bestScore(Core::Type<Score>::max);
    Nn::LabelIndex bestIdx = Core::Type<Nn::LabelIndex>::max;
    std::string    bestLabel;
    bool           bestIsLoop = false;

    // Fetch prev label from hypothesis because this may be expanded with a loop transition
    Nn::LabelIndex prevLabel = Core::Type<Nn::LabelIndex>::max;
    if (not hyp_.labelSeq.empty()) {
        prevLabel = hyp_.labelSeq.back();
    }

    std::optional<Score> score;
    for (auto& [label, idx] : vocabMap_) {
        bool isLoop = allowLabelLoop_ and idx == prevLabel;
        score       = labelScorer_->getDecoderScore(hyp_.history, idx, isLoop);
        if (not score.has_value()) {
            return false;
        }

        if (score < bestScore) {
            bestLabel  = label;
            bestIdx    = idx;
            bestScore  = score.value();
            bestIsLoop = isLoop;
        }
    }

    labelScorer_->extendHistory(hyp_.history, bestIdx, bestIsLoop);
    hyp_.labelSeq.push_back(bestIdx);
    hyp_.score += bestScore;

    // TODO: Create/extend traceback

    ++currentStep_;

    return true;
}

void GreedyTimeSyncSearch::decodeMore() {
    while (decodeStep())
        ;
}

void GreedyTimeSyncSearch::parseVocabFile(const std::string& filename) {
    // Should contain any number of lines in format "<token> <idx". E.g.:
    // <blank> 0
    // AA 1
    // AE 2
    // ...

    if (filename.empty()) {
        error("No vocab file provided");
        return;
    }

    vocabMap_.clear();
    numClasses_ = 0ul;

    std::ifstream input(filename, std::ios::in);
    if (!input.is_open()) {
        error("Could not open vocab file '%s'.", filename.c_str());
    }

    log("Load vocab from file '%s'.", filename.c_str());

    std::string line;
    while (std::getline(input, line)) {
        std::getline(input, line);
        if (line.empty()) {
            continue;
        }
        std::stringstream ss(line);
        std::string       label;
        Nn::LabelIndex    idx;

        if (not(ss >> label >> idx)) {
            warning("Invalid format in vocab file at line: '%s'.", line.c_str());
            continue;
        }

        vocabMap_.emplace(label, idx);
        numClasses_ = std::max(numClasses_, idx);
    }

    if (vocabMap_.empty()) {
        error("No labels in vocabulary.");
    }
}

}  // namespace Search
