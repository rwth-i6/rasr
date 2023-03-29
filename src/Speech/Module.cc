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
#include "Module.hh"
#include <Flow/DataAdaptor.hh>
#include <Flow/Registry.hh>
#include <Mm/Module.hh>
#include "AligningFeatureExtractor.hh"
#include "AlignmentNode.hh"
#include "AlignmentWithLinearSegmentation.hh"
#include "AllophoneStateGraphBuilder.hh"
#include "DataSource.hh"
#include "FeatureScorerNode.hh"
#include "MixtureSetTrainer.hh"
#include "TextDependentSequenceFiltering.hh"
#ifdef MODULE_SPEECH_ALIGNMENT_FLOW_NODES
#include "AlignmentGeneratorNode.hh"
#include "AlignmentTransformNode.hh"
#endif
#ifdef MODULE_SPEECH_LATTICE_FLOW_NODES
#include "AlignmentFromLattice.hh"
#include "LatticeArcAccumulator.hh"
#include "LatticeNodes.hh"
#endif
#ifdef MODULE_SPEECH_LATTICE_RESCORING
#include "LatticeRescorerNodes.hh"
#include "StatePosteriorFeatureScorerNode.hh"
#endif
#ifdef MODULE_SPEECH_DT
#include "DiscriminativeMixtureSetTrainer.hh"
#include "EbwDiscriminativeMixtureSetTrainer.hh"
#include "LatticeExtractor.hh"
#include "RpropDiscriminativeMixtureSetTrainer.hh"
#include "SegmentwiseGmmTrainer.hh"
#endif
#ifdef MODULE_SPEECH_DT_ADVANCED
#include "AdvancedLatticeExtractor.hh"
#endif
#ifdef MODULE_MM_DT
#include <Mm/EbwDiscriminativeMixtureSetEstimator.hh>
#include <Mm/RpropDiscriminativeMixtureSetEstimator.hh>
#endif
#ifdef MODULE_ADAPT_MLLR
#include "FeatureShiftAdaptor.hh"
#endif

using namespace Speech;

Module_::Module_() {
    Flow::Registry::Instance& registry = Flow::Registry::instance();
    registry.registerFilter<AlignmentNode>();
    registry.registerFilter<AlignmentDumpNode>();
    registry.registerFilter<AlignmentWithLinearSegmentationNode>();
    registry.registerFilter<FeatureScorerNode>();
    registry.registerDatatype<Flow::DataAdaptor<Alignment>>();

#ifdef MODULE_GENERIC_SEQ2SEQ_TREE_SEARCH
    registry.registerFilter<Seq2SeqAlignmentNode>();
#endif

#ifdef MODULE_SPEECH_ALIGNMENT_FLOW_NODES
    registry.registerFilter<AlignmentAddWeightNode>();
    registry.registerFilter<AlignmentCombineItemsNode>();
    registry.registerFilter<AlignmentExpmNode>();
    registry.registerFilter<AlignmentFilterWeightsNode>();
    registry.registerFilter<AlignmentGammaCorrectionNode>();
    registry.registerFilter<AlignmentGeneratorNode>();
    registry.registerFilter<AlignmentMultiplyAlignmentsNode>();
    registry.registerFilter<AlignmentMultiplyWeightsNode>();
    registry.registerFilter<AlignmentRemoveEmissionScoreNode>();
    registry.registerFilter<AlignmentResetWeightsNode>();
    registry.registerFilter<AlignmentMapAlphabet>();
    registry.registerFilter<SetAlignmentWeightsByTiedStateAlignmentWeightsNode>();
    registry.registerDatatype<Flow::DataAdaptor<AlignmentGeneratorRef>>();
#endif

#ifdef MODULE_SPEECH_LATTICE_FLOW_NODES
    registry.registerFilter<AlignmentFromLatticeNode>();
    registry.registerFilter<LatticeExpmNode>();
    registry.registerFilter<LatticeNBestNode>();
    registry.registerFilter<LatticeReadNode>();
    registry.registerFilter<LatticeSemiringNode>();
    registry.registerFilter<LatticeSimpleModifyNode>();
    registry.registerFilter<LatticeWordPosteriorNode>();
    registry.registerFilter<ModelCombinationNode>();
    registry.registerFilter<LatticeArcAccumulatorNode>();
    registry.registerDatatype<Flow::DataAdaptor<ModelCombinationRef>>();
#endif

#ifdef MODULE_SPEECH_LATTICE_RESCORING
    registry.registerFilter<AlignmentAcousticLatticeRescorerNode>();
    registry.registerFilter<AcousticLatticeRescorerNode>();
    registry.registerFilter<ApproximatePhoneAccuracyLatticeRescorerNode>();
    registry.registerFilter<FramePhoneAccuracyLatticeRescorerNode>();
    registry.registerFilter<LatticeDumpCtmNode>();
    registry.registerFilter<LatticeExpectationPosteriorNode>();
    registry.registerFilter<LatticeWriteNode>();
    registry.registerFilter<LatticeCacheNode>();
    registry.registerFilter<LatticeCopyNode>();
    registry.registerFilter<NumeratorFromDenominatorNode>();
    registry.registerFilter<SegmentwiseFeaturesNode>();
    registry.registerFilter<SoftFramePhoneAccuracyLatticeRescorerNode>();
    registry.registerFilter<StatePosteriorFeatureScorerNode>();
    registry.registerFilter<WeightedFramePhoneAccuracyLatticeRescorerNode>();
#endif
#ifdef MODULE_ADAPT_MLLR
    registry.registerFilter<FeatureShiftAdaptor>();
#endif
}

namespace {
enum GraphBuilderTopology {
    HMMTopology,
    CTCTopology,
    RNATopology
};

const Core::Choice GraphBuilderTopologyChoice(
    "hmm", HMMTopology,
    "ctc", CTCTopology,
    "rna", RNATopology,
    Core::Choice::endMark());

const Core::ParameterChoice paramTopology(
    "topology", &GraphBuilderTopologyChoice, "topology of graph builder", HMMTopology);
}

AllophoneStateGraphBuilder* Module_::createAllophoneStateGraphBuilder(
        const Core::Configuration& config,
        Core::Ref<const Bliss::Lexicon> lexicon,
        Core::Ref<const Am::AcousticModel> acousticModel,
        bool flatModelAcceptor) const {

    AllophoneStateGraphBuilder* result = nullptr;
    switch (paramTopology(config)) {
        case HMMTopology:
            Core::Application::us()->log("create HMM topology graph builder");
            result = new HMMTopologyGraphBuilder(config, lexicon, acousticModel, flatModelAcceptor);
            break;
        case CTCTopology:
            Core::Application::us()->log("create CTC topology graph builder");
            result = new CTCTopologyGraphBuilder(config, lexicon, acousticModel, flatModelAcceptor);
            break;
        case RNATopology:
            Core::Application::us()->log("create RNA topology graph builder");
            result = new RNATopologyGraphBuilder(config, lexicon, acousticModel, flatModelAcceptor);
            break;
        default:
            Core::Application::us()->criticalError("unknown topology for allophone-state-graph-builder");
    }
    return result;
}


AligningFeatureExtractor* Module_::createAligningFeatureExtractor(
        const Core::Configuration& configuration, AlignedFeatureProcessor& featureProcessor) const {
    return new Speech::AligningFeatureExtractor(configuration, featureProcessor);
}

MixtureSetTrainer* Module_::createMixtureSetTrainer(const Core::Configuration& configuration) const {
    Speech::MixtureSetTrainer* result = 0;
    switch (Mm::Module_::paramEstimatorType(configuration)) {
        case Mm::Module_::maximumLikelihood:
            result = new MlMixtureSetTrainer(configuration);
            break;
#ifdef MODULE_SPEECH_DT
        case Mm::Module_::discriminative:
        case Mm::Module_::discriminativeWithISmoothing:
#endif
#if defined(MODULE_SPEECH_DT)
            result = createDiscriminativeMixtureSetTrainer(configuration);
            break;
#endif
        default:
            defect();
            break;
    }
    return result;
}

Speech::DataSource* Module_::createDataSource(const Core::Configuration& c, bool loadFromFile) const {
    return new Speech::DataSource(c, loadFromFile);
}

#ifdef MODULE_SPEECH_DT
DiscriminativeMixtureSetTrainer* Module_::createDiscriminativeMixtureSetTrainer(
        const Core::Configuration& configuration) const {
    Speech::DiscriminativeMixtureSetTrainer* result = 0;

    switch (Mm::Module_::paramEstimatorType(configuration)) {
#ifdef MODULE_MM_DT
        case Mm::Module_::discriminative:
            result = new EbwDiscriminativeMixtureSetTrainer(configuration);
            break;
        case Mm::Module_::discriminativeWithISmoothing:
            result = new EbwDiscriminativeMixtureSetTrainerWithISmoothing(configuration);
            break;
#endif
        default:
            defect();
            break;
    }
    return result;
}

SegmentwiseGmmTrainer* Module_::createSegmentwiseGmmTrainer(
        const Core::Configuration& config) const {
    switch (AbstractSegmentwiseTrainer::paramCriterion(config)) {
            // standard error-based training criteria without and with
            // I-smoothing, e.g. MPE
        case AbstractSegmentwiseTrainer::minimumError:
        case AbstractSegmentwiseTrainer::minimumErrorWithISmoothing:
            return new MinimumErrorSegmentwiseGmmTrainer(config);
            break;
        default:
            defect();
            break;
    }
    return 0;
}

LatticeRescorer* Module_::createDistanceLatticeRescorer(
        const Core::Configuration& config, Bliss::LexiconRef lexicon) const {
    DistanceLatticeRescorer::DistanceType type =
            static_cast<DistanceLatticeRescorer::DistanceType>(
                    DistanceLatticeRescorer::paramDistanceType(config));
    DistanceLatticeRescorer::SpokenSource source =
            static_cast<DistanceLatticeRescorer::SpokenSource>(
                    DistanceLatticeRescorer::paramSpokenSource(config));
    LatticeRescorer* rescorer = 0;
    switch (type) {
        case DistanceLatticeRescorer::approximateWordAccuracy:
            switch (source) {
                case DistanceLatticeRescorer::orthography:
                    rescorer = new OrthographyApproximateWordAccuracyLatticeRescorer(config, lexicon);
                    break;
                case DistanceLatticeRescorer::archive:
                    rescorer = new ArchiveApproximateWordAccuracyLatticeRescorer(config, lexicon);
                    break;
                default:
                    defect();
                    break;
            }
            break;
        case DistanceLatticeRescorer::approximatePhoneAccuracy:
            switch (source) {
                case DistanceLatticeRescorer::orthography:
                    rescorer = new OrthographyApproximatePhoneAccuracyLatticeRescorer(config, lexicon);
                    break;
                case DistanceLatticeRescorer::archive:
                    rescorer = new ArchiveApproximatePhoneAccuracyLatticeRescorer(config, lexicon);
                    break;
                default:
                    defect();
                    break;
            }
            break;
#ifdef MODULE_SPEECH_DT_ADVANCED
        case DistanceLatticeRescorer::approximatePhoneAccuracyMask:
            switch (source) {
                case DistanceLatticeRescorer::orthography:
                    rescorer = new OrthographyApproximatePhoneAccuracyMaskLatticeRescorer(config, lexicon);
                    break;
                default:
                    defect();
                    break;
            }
            break;
        case DistanceLatticeRescorer::frameStateAccuracy:
            switch (source) {
                case DistanceLatticeRescorer::orthography:
                    rescorer = new OrthographyFrameStateAccuracyLatticeRescorer(config, lexicon);
                    break;
                case DistanceLatticeRescorer::archive:
                    rescorer = new ArchiveFrameStateAccuracyLatticeRescorer(config, lexicon);
                    break;
                default:
                    defect();
                    break;
            }
            break;
        case DistanceLatticeRescorer::smoothedFrameStateAccuracy:
            switch (source) {
                case DistanceLatticeRescorer::orthography:
                    rescorer = new OrthographySmoothedFrameStateAccuracyLatticeRescorer(config, lexicon);
                    break;
                default:
                    defect();
                    break;
            }
            break;
        case DistanceLatticeRescorer::levenshteinOnList:
            require(source == DistanceLatticeRescorer::orthography);
            rescorer = new LevenshteinListRescorer(config, lexicon);
            break;
        case DistanceLatticeRescorer::wordAccuracy:
            require(source == DistanceLatticeRescorer::orthography);
            rescorer = new WordAccuracyLatticeRescorer(config, lexicon);
            break;
        case DistanceLatticeRescorer::phonemeAccuracy:
            require(source == DistanceLatticeRescorer::orthography);
            rescorer = new PhonemeAccuracyLatticeRescorer(config, lexicon);
            break;
        case DistanceLatticeRescorer::frameWordAccuracy:
            switch (source) {
                case DistanceLatticeRescorer::orthography:
                    rescorer = new OrthographyFrameWordAccuracyLatticeRescorer(config, lexicon);
                    break;
                default:
                    defect();
                    break;
            }
            break;
        case DistanceLatticeRescorer::framePhoneAccuracy:
            switch (source) {
                case DistanceLatticeRescorer::orthography:
                    rescorer = new OrthographyFramePhoneAccuracyLatticeRescorer(config, lexicon);
                    break;
                default:
                    defect();
                    break;
            }
            break;
#endif  // MODULE_SPEECH_DT_ADVANCED
        default:
            defect();
            break;
    }
    return rescorer;
}

#endif  // MODULE_SPEECH_DT
