set(RasrNn_SOURCES Module.cc)

list(
    APPEND
    RasrNn_SOURCES
    LabelScorer/BufferedLabelScorer.cc
    LabelScorer/CombineLabelScorer.cc
    LabelScorer/DataView.cc
    LabelScorer/Encoder.cc
    LabelScorer/EncoderDecoderLabelScorer.cc
    LabelScorer/EncoderFactory.cc
    LabelScorer/FixedContextOnnxLabelScorer.cc
    LabelScorer/LabelScorer.cc
    LabelScorer/LabelScorerFactory.cc
    LabelScorer/NoContextOnnxLabelScorer.cc
    LabelScorer/NoOpLabelScorer.cc
    LabelScorer/PriorLabelScorer.cc
    LabelScorer/ScaledLabelScorer.cc
    LabelScorer/ScoreAccessor.cc
    LabelScorer/ScoringContext.cc
    LabelScorer/StatefulOnnxLabelScorer.cc
    LabelScorer/TransitionLabelScorer.cc
    LabelScorer/TransitionTypes.cc
)

if(${MODULE_NN})
    list(
        APPEND
        RasrNn_SOURCES
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
endif()
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
    list(APPEND RasrNn_SOURCES PythonFeatureScorer.cc PythonTrainer.cc
         PythonControl.cc PythonLayer.cc
    )
endif()

add_library(RasrNn STATIC ${RasrNn_SOURCES})

target_link_libraries(
    RasrNn INTERFACE RasrCore RasrMath RasrMathLapack RasrFlow RasrSignal
                     RasrSpeech
)

target_compile_options(RasrNn PRIVATE "-fexceptions")

if(${MODULE_PYTHON})
    add_python_dependencies(RasrNn)
endif()

if(${MODULE_ONNX})
    target_link_libraries(RasrNn INTERFACE RasrOnnx)
    add_onnx_dependencies(RasrNn)
endif()

if(${MODULE_GENERIC_SEQ2SEQ_TREE_SEARCH} AND ${MODULE_TENSORFLOW})
    add_tf_dependencies(RasrNn)
endif()

add_executable(check-nn check.cc)
target_link_libraries(check-nn PRIVATE RasrNn)
