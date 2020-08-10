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
#ifndef _SPEECH_MINIMUM_BAYES_RISK_SEARCH_HH
#define _SPEECH_MINIMUM_BAYES_RISK_SEARCH_HH

#include <Speech/LatticeSetProcessor.hh>

namespace Search {
class MinimumBayesRiskSearch;
}

namespace Speech {
class NBestListExtractor;
}

namespace Speech {

/**
 * Search the lattice for string with minimal Bayes risk.
 * The result will always be a linear FSA representing a path in the lattice.
 */
class MinimumBayesRiskSearchNode : public LatticeSetProcessor {
    /** To be documented! */
    typedef LatticeSetProcessor Precursor;

    /** Registered actions to perform for minimum Bayes risk. */
    typedef enum SearchMethod {
        searchDryRun,
        searchMap,
        searchNBestListNaive,
        searchAStar
    };

private:
    /**
     * Choice which search method shall be applied.
     */
    static const Core::ParameterChoice paramSearchMethod;

    /**
     * The possible choices for search methods.
     */
    static const Core::Choice choiceSearchMethod;

    /**
     * The search object performing the actual search.
     *
     * To be deleted in destructor!
     */
    Search::MinimumBayesRiskSearch* search_;

    Bliss::LexiconRef      lexicon_;
    Fsa::ConstAutomatonRef lemmaPronToLemma_;
    Fsa::ConstAutomatonRef lemmaToEval_;

    NBestListExtractor* nBestListExtractor_;

private:
    Lattice::ConstWordLatticeRef mapEvalToLemmaPronunciation(
            Fsa::ConstAutomatonRef, Lattice::ConstWordLatticeRef);

public:
    /**
     * Standard constructor generating a configurable node.
     * The search object will be initialized here.
     *
     * @param config  Configuration for this node.
     */
    MinimumBayesRiskSearchNode(const Core::Configuration& config);

    /**
     * Virtual destructor. Here, the search object will be deleted.
     *
     */
    virtual ~MinimumBayesRiskSearchNode();

    /**
     * Initialization of lexicon and evaluator.
     *
     * @param lexicon  the Bliss lexicon for this corpus.
     */
    void initialize(Bliss::LexiconRef lexicon);

    /**
     * The actual processing of the word lattice where the search will be performed.
     * The lattice's Fsas will be linear combined each with weight 1.0 and the resulting total Fsa
     * will be searched.
     *
     * For the future, the Fsa to take should be made configurable!
     *
     * @param lattice  the word lattice to be searched.
     * @param segment  the speech segment, the lattice belongs to.
     */
    virtual void processWordLattice(Lattice::ConstWordLatticeRef lattice, Bliss::SpeechSegment* segment);
};

}  //namespace Speech

#endif  // _SPEECH_MINIMUM_BAYES_RISK_SEARCH_HH
