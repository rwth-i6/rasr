add_library(
    RasrNn STATIC
    ActivationLayer.cc
    Activations.cc
    AllophoneStateFsaExporter.cc
    BatchEstimator.cc
    BatchFeatureScorer.cc
    BufferedAlignedFeatureProcessor.cc
    BufferedFeatureExtractor.cc
    BufferedSegmentFeatureProcessor.cc
    ClassLabelWrapper.cc
    CompressedVector.cc
    Criterion.cc
    Estimator.cc
    FeatureScorer.cc
    FeedForwardTrainer.cc
    FixedQuantizationCompressedVectorFactory.cc
    LinearAndActivationLayer.cc
    LinearLayer.cc
    MeanNormalizedSgdEstimator.cc
    Module.cc
    NetworkTopology.cc
    NeuralNetwork.cc
    NeuralNetworkForwardNode.cc
    NeuralNetworkLayer.cc
    NeuralNetworkTrainer.cc
    OperationLayer.cc
    PoolingLayer.cc
    PreprocessingLayer.cc
    Prior.cc
    QuantizedCompressedVectorFactory.cc
    ReducedPrecisionCompressedVectorFactory.cc
    Regularizer.cc
    Statistics.cc
    TrainerFeatureScorer.cc
)

add_subdirectory(LabelScorer)

if(${MODULE_NN_SEQUENCE_TRAINING})
    target_sources(
        RasrNn
        PRIVATE EmissionLatticeRescorer.cc LatticeAccumulators.cc
                SegmentwiseNnTrainer.cc MmiSegmentwiseNnTrainer.cc
                MeSegmentwiseNnTrainer.cc SharedNeuralNetwork.cc
    )
endif()
if(${MODULE_PYTHON})
    target_sources(
        RasrNn PRIVATE PythonFeatureScorer.cc PythonTrainer.cc PythonControl.cc
                       PythonLayer.cc
    )
endif()

target_link_libraries(
    RasrNn
    PUBLIC RasrAm
           RasrBliss
           RasrCore
           RasrFlow
           RasrFsa
           RasrLattice
           RasrMath
           RasrMm
           RasrSearch
           RasrSpeech
)

if(${MODULE_PYTHON})
    target_link_libraries(RasrNn PUBLIC RasrPython)
endif()

add_rasr_check_executable(check-nn check.cc)

target_link_rasr_check_libraries(check-nn PRIVATE RasrNn)
