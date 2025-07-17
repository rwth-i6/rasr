add_subdirectory(LabelScorer)

set(RasrNn_SOURCES
    Module.cc
    ActivationLayer.cc
    Activations.cc
    AllophoneStateFsaExporter.cc
    BatchEstimator.cc
    BatchFeatureScorer.cc
    BufferedAlignedFeatureProcessor.cc
    BufferedFeatureExtractor.cc
    BufferedSegmentFeatureProcessor.cc
    ClassLabelWrapper.cc
    Criterion.cc
    Estimator.cc
    FeatureScorer.cc
    FeedForwardTrainer.cc
    LinearAndActivationLayer.cc
    LinearLayer.cc
    MeanNormalizedSgdEstimator.cc
    NetworkTopology.cc
    NeuralNetwork.cc
    NeuralNetworkForwardNode.cc
    NeuralNetworkLayer.cc
    NeuralNetworkTrainer.cc
    OperationLayer.cc
    PoolingLayer.cc
    PreprocessingLayer.cc
    Prior.cc
    Regularizer.cc
    Statistics.cc
    TrainerFeatureScorer.cc
)

if(${MODULE_NN_SEQUENCE_TRAINING})
    list(
        APPEND
        RasrNn_SOURCES
        EmissionLatticeRescorer.cc
        LatticeAccumulators.cc
        SegmentwiseNnTrainer.cc
        MmiSegmentwiseNnTrainer.cc
        MeSegmentwiseNnTrainer.cc
        SharedNeuralNetwork.cc
    )
endif()

if(${MODULE_PYTHON})
    list(APPEND RasrNn_SOURCES PythonFeatureScorer.cc PythonTrainer.cc PythonControl.cc PythonLayer.cc)
endif()

add_library(RasrNn STATIC ${RasrNn_SOURCES})

target_link_libraries(
    RasrNn
    PUBLIC RasrAm
           RasrBliss
           RasrCore
           RasrFlow
           RasrFsa
           RasrLabelScorer
           RasrLattice
           RasrMath
           RasrMm
           RasrSearch
           RasrSpeech
)

if(${MODULE_PYTHON})
    target_link_libraries(RasrNn PUBLIC RasrPython)
endif()

add_executable(check-nn check.cc)
target_link_libraries(check-nn PRIVATE RasrNn)
