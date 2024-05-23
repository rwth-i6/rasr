set(RasrNn_SOURCES
        Module.cc
)

if (${MODULE_NN})
    list(APPEND RasrNn_SOURCES
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
            CtcCriterion.cc
            Estimator.cc
            FeatureScorer.cc
            FeedForwardTrainer.cc
            GradientCheck.cc
            LinearAndActivationLayer.cc
            LinearLayer.cc
            LookupLayer.cc
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
            RpropEstimator.cc
            Statistics.cc
            TrainerFeatureScorer.cc
    )
endif ()
if (${MODULE_NN_SEQUENCE_TRAINING})
    list(APPEND RasrNn_SOURCES
            EmissionLatticeRescorer.cc
            LatticeAccumulators.cc
            SegmentwiseNnTrainer.cc
            MmiSegmentwiseNnTrainer.cc
            MeSegmentwiseNnTrainer.cc
            SharedNeuralNetwork.cc
    )
endif ()
if (${MODULE_PYTHON})
    list(APPEND RasrNn_SOURCES
            PythonFeatureScorer.cc
            PythonTrainer.cc
            PythonControl.cc
            PythonLayer.cc
    )
endif ()
if (${MODULE_GENERIC_SEQ2SEQ_TREE_SEARCH})
    list(APPEND RasrNn_SOURCES LabelScorer.cc)
    if (${MODULE_TENSORFLOW})
        list(APPEND RasrNn_SOURCES TFLabelScorer.cc)
    endif ()
    if (${MODULE_ONNX})
        list(APPEND RasrNn_SOURCES OnnxLabelScorer.cc)
    endif ()
endif ()

add_library(RasrNn STATIC ${RasrNn_SOURCES})

target_link_libraries(RasrNn INTERFACE
        RasrCore
        RasrMath
        RasrMathLapack
        RasrFlow
        RasrSignal
)

target_compile_options(RasrNn PRIVATE "-fexceptions")

if (${MODULE_PYTHON})
    add_python_dependencies(RasrNn)
endif ()

if (${MODULE_GENERIC_SEQ2SEQ_TREE_SEARCH} AND ${MODULE_TENSORFLOW})
    add_tf_dependencies(RasrNn)
    target_link_libraries(RasrNn INTERFACE RasrTensorflow)
endif ()
if (${MODULE_GENERIC_SEQ2SEQ_TREE_SEARCH} AND ${MODULE_ONNX})
    add_onnx_dependencies(RasrNn)
    target_link_libraries(RasrNn INTERFACE RasrOnnx)
endif ()
