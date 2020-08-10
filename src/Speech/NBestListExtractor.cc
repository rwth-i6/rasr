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
#include "NBestListExtractor.hh"
#include <Fsa/Basic.hh>
#include <Fsa/Best.hh>
#include <Fsa/Output.hh>
#include <Lattice/Posterior.hh>

using namespace Speech;

/**
 * NBestListExtractor
 */
const Core::ParameterInt NBestListExtractor::paramNumberOfHypotheses(
        "number-of-hypotheses",
        "number of hypotheses in n-best list",
        100,
        0);

const Core::ParameterFloat NBestListExtractor::paramMinPruningThreshold(
        "min-pruning",
        "minimum (start) threshold used for posterior pruning of word lattices",
        Core::Type<f32>::max);

const Core::ParameterFloat NBestListExtractor::paramMaxPruningThreshold(
        "max-pruning",
        "maximum threshold used for posterior pruning of word lattices",
        Core::Type<f32>::max);

const Core::ParameterFloat NBestListExtractor::paramPruningIncrement(
        "pruning-increment",
        "increment current threshold by this value",
        5,
        1);

const Core::ParameterBool NBestListExtractor::paramWorkOnOutput(
        "work-on-output",
        "score based on output (default: input==false)",
        false);

const Core::ParameterBool NBestListExtractor::paramLatticeIsDeterministic(
        "lattice-is-deterministic",
        "input lattices are deterministic",
        true);

const Core::ParameterBool NBestListExtractor::paramHasFailArcs(
        "has-fail-arcs",
        "Used Automata have fail arcs",
        false);

const Core::ParameterBool NBestListExtractor::paramNormalize(
        "normalize",
        "get normalization",
        false);

NBestListExtractor::NBestListExtractor(const Core::Configuration& c)
        : Core::Component(c) {
    setNumberOfHypotheses(paramNumberOfHypotheses(config));
    setMinPruningThreshold(paramMinPruningThreshold(config));
    setMaxPruningThreshold(paramMaxPruningThreshold(config));
    setPruningIncrement(paramPruningIncrement(config));
    setWorkOnOutput(paramWorkOnOutput(config));
    setLatticeIsDeterministic(paramLatticeIsDeterministic(config));
    setHasFailArcs(paramHasFailArcs(config));
    setNormalize(paramNormalize(config));
}

NBestListExtractor::~NBestListExtractor() {}

/**
 * NBestListLatticeProcessor
 */
NBestListLatticeProcessor::NBestListLatticeProcessor(const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c),
          statisticsChannel_(config, "statistics"),
          extractor_(config) {}

NBestListLatticeProcessor::~NBestListLatticeProcessor() {}

void NBestListLatticeProcessor::processWordLattice(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* s) {
    Lattice::ConstWordLatticeRef nBestList = extractor_.getNBestList(lattice);
    if (statisticsChannel_.isOpen()) {
        Fsa::ConstAutomatonRef list       = nBestList->mainPart();
        Fsa::ConstStateRef     hypotheses = list->getState(list->initialStateId());
        statisticsChannel_ << Core::XmlOpen("n-best-list-statistics") + Core::XmlAttribute("size", hypotheses->nArcs());
        for (Fsa::State::const_iterator it = hypotheses->begin(); it != hypotheses->end(); ++it) {
            statisticsChannel_ << Core::XmlOpen("hypothesis") + Core::XmlAttribute("rank", it->target()) + Core::XmlAttribute("score", f32(Fsa::bestscore(Fsa::partial(list, it->target()))));
            statisticsChannel_ << Core::XmlClose("hypothesis");
        }
        statisticsChannel_ << Core::XmlClose("n-best-list-statistics");
    }
    Precursor::processWordLattice(nBestList, s);
}

void NBestListLatticeProcessor::initialize(Bliss::LexiconRef lexicon) {
    Precursor::initialize(lexicon);
    extractor_.initialize(lexicon);
}
