add_library(
    RasrSpeech STATIC
    AcousticModelTrainer.cc
    AlignerModelAcceptor.cc
    AligningFeatureExtractor.cc
    Alignment.cc
    AlignmentNode.cc
    AlignmentWithLinearSegmentation.cc
    AllophoneStateGraphBuilder.cc
    AverageFeatureScorerActivation.cc
    CorpusProcessor.cc
    CorpusVisitor.cc
    CovarianceEstimator.cc
    DataExtractor.cc
    DataSource.cc
    DelayedRecognizer.cc
    Feature.cc
    FeatureScorer.cc
    FeatureScorerNode.cc
    FsaCache.cc
    LabelingFeatureExtractor.cc
    MixtureSetTrainer.cc
    ModelCombination.cc
    Module.cc
    Recognizer.cc
    ScatterMatricesEstimator.cc
    TextDependentSequenceFiltering.cc
    TextIndependentMixtureSetTrainer.cc
)

if(${MODULE_SPEECH_LATTICE_ALIGNMENT})
    target_sources(
        RasrSpeech
        PRIVATE PhonemeSequenceAlignmentGenerator.cc
                SegmentwiseAlignmentGenerator.cc SegmentwiseFeatureExtractor.cc
    )
endif()

if(${MODULE_SPEECH_DT})
    target_sources(
        RasrSpeech
        PRIVATE AbstractSegmentwiseTrainer.cc
                AccuracyFsaBuilder.cc
                AcousticSegmentwiseTrainer.cc
                Confidences.cc
                DiscriminativeMixtureSetTrainer.cc
                EbwDiscriminativeMixtureSetTrainer.cc
                LatticeExtractor.cc
                LatticeSetExtractor.cc
                LatticeSetProcessor.cc
                PruningLatticeSetNode.cc
                SegmentwiseGmmTrainer.cc
                WordLatticeExtractor.cc
    )
endif()

if(${MODULE_SPEECH_DT_ADVANCED})
    target_sources(
        RasrSpeech
        PRIVATE AdvancedAccuracyFsaBuilder.cc
                AdvancedLatticeExtractor.cc
                AdvancedWordLatticeExtractor.cc
                AuxiliarySegmentwiseTrainer.cc
                ClusterMixtureSetTrainer.cc
                AdvancedLatticeSetProcessor.cc
                NBestListExtractor.cc
                RpropDiscriminativeMixtureSetTrainer.cc
    )
endif()

if(${MODULE_FLF_EXT})
    target_sources(RasrSpeech PRIVATE AlignedFeatureCache.cc)
endif()

if(${MODULE_SPEECH_LATTICE_FLOW_NODES})
    target_sources(
        RasrSpeech PRIVATE AlignmentFromLattice.cc LatticeNodes.cc
                           LatticeArcAccumulator.cc
    )
endif()

if(${MODULE_SPEECH_ALIGNMENT_FLOW_NODES})
    target_sources(
        RasrSpeech PRIVATE AlignmentGeneratorNode.cc AlignmentTransformNode.cc
                           SegmentNode.cc
    )
endif()

if(${MODULE_ADAPT_MLLR})
    target_sources(
        RasrSpeech PRIVATE ModelTransformEstimator.cc FeatureShiftAdaptor.cc
    )
endif()

if(${MODULE_ADAPT_CMLLR})
    target_sources(
        RasrSpeech PRIVATE AffineFeatureTransformEstimator.cc KeyedEstimator.cc
    )
endif()

if(${MODULE_CART})
    target_sources(RasrSpeech PRIVATE DecisionTreeTrainer.cc)
endif()

if(${MODULE_SPEECH_LATTICE_RESCORING})
    target_sources(
        RasrSpeech PRIVATE LatticeRescorerAutomaton.cc LatticeRescorerNodes.cc
                           StatePosteriorFeatureScorerNode.cc
    )
endif()

if(${MODULE_SIGNAL_ADVANCED})
    target_sources(RasrSpeech PRIVATE HistogramEstimator.cc MeanEstimator.cc)
endif()

if(${MODULE_SEARCH_MBR})
    target_sources(RasrSpeech PRIVATE MinimumBayesRiskSearch.cc)
endif()

target_link_libraries(
    RasrSpeech
    PUBLIC RasrAm
           RasrBliss
           RasrCore
           RasrFlf
           RasrFlow
           RasrFsa
           RasrLattice
           RasrLm
           RasrMath
           RasrMc
           RasrMm
           RasrNn
           RasrSearch
           RasrSignal
)

if(${MODULE_CART})
    target_link_libraries(RasrSpeech PUBLIC RasrCart)
endif()

add_executable(check-speech check.cc)

target_link_libraries(check-speech PRIVATE RasrSpeech)
