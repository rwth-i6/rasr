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
#ifndef _NN_CTCCRITERION_HH
#define _NN_CTCCRITERION_HH

#include "Criterion.hh"
#include "BatchStateScoreIntf.hh"
#include <Math/FastMatrix.hh>
#include <memory>
#include <vector>
#include <Mm/FeatureScorer.hh>
#include <Fsa/Automaton.hh>


namespace Bliss {
class SpeechSegment;
class Lexicon;
}
namespace Am {
class AcousticModel;
}
namespace Speech {
class AllophoneStateGraphBuilder;
class DataSource;
class Alignment;
}
namespace Mm {
class FeatureScorer;
class ScaledFeatureScorer;
}
namespace Search {
class Aligner;
}
namespace Core {
class Archive;
}
namespace Fsa {
class Automaton;
}


namespace Nn {

template<typename FloatT>
class Prior;
template<typename FloatT>
class TimeAlignedAutomaton;

/* Based on a segment with transcription, it will calculate a CTC-like criterion.
 */
template<typename FloatT>
class CtcCriterion : public SegmentCriterion<FloatT>, public BatchStateScoreIntf<FloatT> {
    typedef SegmentCriterion<FloatT> Precursor;
protected:
    typedef typename Types<FloatT>::NnVector NnVector;
    typedef typename Types<FloatT>::NnMatrix NnMatrix;

    Core::Ref<Am::AcousticModel> acousticModel_;
    Core::Ref<const Bliss::Lexicon> lexicon_;
    std::shared_ptr<Speech::AllophoneStateGraphBuilder> allophoneStateGraphBuilder_;
    bool useSearchAligner_;
    std::shared_ptr<Search::Aligner> aligner_;
    bool useDirectAlignmentExtraction_;
    FloatT minAcousticPruningThreshold_, maxAcousticPruningThreshold_;
    FloatT statePosteriorScale_;
    FloatT statePosteriorLogBackoff_;
    std::shared_ptr<Prior<FloatT> > statePriors_;
    Core::Ref<Mm::ScaledFeatureScorer> fixedMixtureSetFeatureScorer_;
    Core::Ref<Speech::DataSource> fixedMixtureSetFeatureExtractionDataSource_;
    std::string fixedMixtureSetExtractAlignmentsPortName_;
    bool posteriorUseSearchAligner_;
    bool posteriorTotalNormalize_;
    FloatT posteriorArcLogThreshold_;
    FloatT posteriorScale_;
    unsigned int posteriorNBestLimit_;
    std::shared_ptr<Core::Archive> dumpViterbiAlignmentsArchive_;
    std::shared_ptr<Core::Archive> dumpReferenceProbsArchive_;
    bool doDebugDumps_;
    bool logTimeStatistics_;
    bool useCrossEntropyAsLoss_;
    bool inputInLogSpace_;

    NnMatrix stateLogPosteriors_; // in -log space
    bool discardCurrentInput_;
    using Precursor::segment_;

    void initLexicon();
    void initAcousticModel();
    void initAllophoneStateGraphBuilder();
    void initSearchAligner();
    void initStatePriors();
    void initFixedMixtureSet();
    void initDebug();

    u32 nEmissions() const;
    u32 getCurSegmentTimeLen() const;
    virtual u32 getBatchLen() { return getCurSegmentTimeLen(); }
    virtual FloatT getStateScore(u32 timeIdx, u32 emissionIdx); // -log space
    void getStateScorers_MixtureSet(
            /*out*/ std::vector<Core::Ref<const Mm::FeatureScorer::ContextScorer> >& scorers);
    void getStateScorers(
            /*out*/ std::vector<Core::Ref<const Mm::FeatureScorer::ContextScorer> >& scorers);
    Core::Ref<const Fsa::Automaton> getHypothesesAllophoneStateFsa();
    Core::Ref<const Fsa::Automaton> getTimeAlignedFsa_SearchAligner();
    Core::Ref<TimeAlignedAutomaton<FloatT> > getTimeAlignedFsa_custom();
    Core::Ref<const Fsa::Automaton> getTimeAlignedFsa();
    Core::Ref<const Fsa::Automaton> getPosteriorFsa();
    bool calcStateProbErrors(
            /*out*/ FloatT& error, // the error (objective function value)
            /*out*/ NnMatrix& referenceProb // the reference prob
            );
    void dumpViterbiAlignments();

public:
    CtcCriterion(const Core::Configuration &c);
    ~CtcCriterion();

    virtual void inputSpeechSegment(Bliss::SpeechSegment& segment, NnMatrix& nnOutput, NnVector* weights = NULL);
    virtual void getObjectiveFunction(FloatT& value);
    virtual void getErrorSignal(NnMatrix& errorSignal);
    virtual void getErrorSignal_naturalPairing(NnMatrix& errorSignal, NeuralNetworkLayer<FloatT>& lastLayer);
    virtual NnMatrix* getPseudoTargets();
    virtual bool discardCurrentInput() {
        return discardCurrentInput_;
    }

    Core::Ref<Am::AcousticModel> getAcousticModel();
    bool getAlignment(Speech::Alignment& out, NnMatrix& logPosteriors, const std::string& orthography, FloatT minProbGT = 0.0, FloatT gamma = 1.0);
};

}

#endif // CTCCRITERION_HH
