set(SPEECH_SOURCES
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
    list(APPEND SPEECH_SOURCES
            PhonemeSequenceAlignmentGenerator.cc
            SegmentwiseAlignmentGenerator.cc
            SegmentwiseFeatureExtractor.cc
    )
endif ()

if (${MODULE_SPEECH_DT})
    list(APPEND Speech_Sources
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
            WordLatticeExtractor.cc
    )
endif ()

if (${MODULE_SPEECH_DT_ADVANCED})
    list(APPEND SPEECH_SOURCES
            AdvancedAccuracyFsaBuilder.cc
            AdvancedLatticeExtractor.cc
            AuxiliarySegmentwiseTrainer.cc
            ClusterMixtureSetTrainer.cc
            AdvancedLatticeSetProcessor.cc
            NBestListExtractor.cc
            RpropDiscriminativeMixtureSetTrainer.cc
    )
endif ()

if (${MODULE_FLF_EXT})
    list(APPEND SPEECH_SOURCES AlignedFeatureCache.cc)
endif ()

if (${MODULE_SPEECH_LATTICE_FLOW_NODES})
    list(APPEND SPEECH_SOURCES
            AlignmentFromLattice.cc
            LatticeNodes.cc
            LatticeArcAccumulator.cc
    )
endif ()

if (${MODULE_SPEECH_ALIGNMENT_FLOW_NODES})
    list(APPEND SPEECH_SOURCES
            AlignmentGeneratorNode.cc
            AlignmentTransformNode.cc
            SegmentNode.cc
    )
endif ()

if (${MODULE_ADAPT_MLLR})
    list(APPEND SPEECH_SOURCES
            ModelTransformEstimator.cc
            FeatureShiftAdaptor.cc
    )
endif ()

if (${MODULE_ADAPT_CMLLR})
    list(APPEND SPEECH_SOURCES
            AffineFeatureTransformEstimator.cc
            KeyedEstimator.cc
    )
endif ()

if (${MODULE_CART})
    list(APPEND SPEECH_SOURCES
            DecisionTreeTrainer.cc
            KeyedEstimator.cc
    )
endif ()

if (${MODULE_SPEECH_LATTICE_RESCORING})
    list(APPEND SPEECH_SOURCES
            LatticeRescorerAutomaton.cc
            LatticeRescorerNodes.cc
            StatePosteriorFeatureScorerNode.cc
    )
endif ()

if (${MODULE_SIGNAL_ADVANCED})
    list(APPEND SPEECH_SOURCES
            HistogramEstimator.cc
            MeanEstimator.cc
    )
endif ()

if (${MODULE_SEARCH_MBR})
    list(APPEND SPEECH_SOURCES
            MinimumBayesRiskSearch.cc
    )
endif ()

add_library(RasrCart STATIC ${Cart_SOURCES})
target_link_libraries(RasrCart PRIVATE RasrCore)

target_include_directories(RasrCart PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})