set(Speech_SOURCES
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
    TextIndependentMixtureSetTrainer.cc)

if(${MODULE_SPEECH_LATTICE_ALIGNMENT})
  list(APPEND Speech_SOURCES PhonemeSequenceAlignmentGenerator.cc
       SegmentwiseAlignmentGenerator.cc SegmentwiseFeatureExtractor.cc)
endif()

if(${MODULE_SPEECH_DT})
  list(
    APPEND
    Speech_SOURCES
    AbstractSegmentwiseTrainer.cc
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
    WordLatticeExtractor.cc)
endif()

if(${MODULE_SPEECH_DT_ADVANCED})
  list(
    APPEND
    Speech_SOURCES
    AdvancedAccuracyFsaBuilder.cc
    AdvancedLatticeExtractor.cc
    AuxiliarySegmentwiseTrainer.cc
    ClusterMixtureSetTrainer.cc
    AdvancedLatticeSetProcessor.cc
    NBestListExtractor.cc
    RpropDiscriminativeMixtureSetTrainer.cc)
endif()

if(${MODULE_FLF_EXT})
  list(APPEND Speech_SOURCES AlignedFeatureCache.cc)
endif()

if(${MODULE_SPEECH_LATTICE_FLOW_NODES})
  list(APPEND Speech_SOURCES AlignmentFromLattice.cc LatticeNodes.cc
       LatticeArcAccumulator.cc)
endif()

if(${MODULE_SPEECH_ALIGNMENT_FLOW_NODES})
  list(APPEND Speech_SOURCES AlignmentGeneratorNode.cc
       AlignmentTransformNode.cc SegmentNode.cc)
endif()

if(${MODULE_ADAPT_MLLR})
  list(APPEND Speech_SOURCES ModelTransformEstimator.cc FeatureShiftAdaptor.cc)
endif()

if(${MODULE_ADAPT_CMLLR})
  list(APPEND Speech_SOURCES AffineFeatureTransformEstimator.cc
       KeyedEstimator.cc)
endif()

if(${MODULE_CART})
  list(APPEND Speech_SOURCES DecisionTreeTrainer.cc)
endif()

if(${MODULE_SPEECH_LATTICE_RESCORING})
  list(APPEND Speech_SOURCES LatticeRescorerAutomaton.cc
       LatticeRescorerNodes.cc StatePosteriorFeatureScorerNode.cc)
endif()

if(${MODULE_SIGNAL_ADVANCED})
  list(APPEND Speech_SOURCES HistogramEstimator.cc MeanEstimator.cc)
endif()

if(${MODULE_SEARCH_MBR})
  list(APPEND Speech_SOURCES MinimumBayesRiskSearch.cc)
endif()

add_library(RasrSpeech STATIC ${Speech_SOURCES})

target_link_libraries(
  RasrSpeech
  INTERFACE RasrLm
            RasrAm
            RasrMm
            RasrMc
            RasrNn
            RasrSearch
            RasrBliss
            RasrFlow
            RasrFsa
            RasrCore
            RasrLattice
            RasrMath
            RasrMathLapack)

if(${MODULE_CART})
  target_link_libraries(RasrSpeech INTERFACE RasrCart)
endif()

if(${MODULE_MATH_NR})
  target_link_libraries(RasrSpeech INTERFACE RasrMathNr)
endif()

if(${MODULE_SEARCH_WFST})
  target_link_libraries(RasrSpeech INTERFACE RasrSearchWfst RasrOpenFst)
endif()

if(${MODULE_ADVANCED_TREE_SEARCH})
  target_link_libraries(RasrSpeech INTERFACE RasrAdvancedTreeSearch)
endif()

if(${MODULE_PYTHON})
  target_link_libraries(RasrSpeech INTERFACE RasrPython)
  add_python_dependencies(RasrSpeech)
endif()

if(${MODULE_TENSORFLOW})
  target_link_libraries(RasrSpeech INTERFACE RasrTensorflow)
  add_tf_dependencies(RasrSpeech)
endif()

add_executable(check-speech check.cc)
target_link_libraries(check-speech PRIVATE RasrSpeech)
