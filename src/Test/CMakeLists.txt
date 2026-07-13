add_executable(
    unit-test
    Bliss_Orthography.cc
    Bliss_SegmentOrdering.cc
    Core_StringUtilities.cc
    Core_Thread.cc
    Core_ThreadPool.cc
    File.cc
    Fsa_Sssp4SpecialSymbols.cc
    Lexicon.cc
    Lm_ArpaLm.cc
    Math_Blas.cc
    Math_CudaMatrix.cc
    Math_CudaVector.cc
    Math_FastMatrix.cc
    Math_FastVectorOperations.cc
    Math_LinearConjugateGradient.cc
    Math_Utilities.cc
    Registry.cc
    Test_File.cc
    Test_Lexicon.cc
    UnitTester.cc
)
add_install_executable(unit-test)

if(${MODULE_NN})
    target_sources(
        unit-test
        PRIVATE Nn_NetworkTopology.cc
                Nn_BufferedFeatureExtractor.cc
                Nn_BufferedAlignedFeatureProcessor.cc
                Nn_ClassLabelWrapper.cc
                Nn_FeedForwardCrossEntropyTrainer.cc
                Nn_LinearAndActivationLayer.cc
                Nn_LinearLayer.cc
                Nn_NeuralNetwork.cc
                Nn_NeuralNetworkLayer.cc
                Nn_NeuralNetworkTrainer.cc
                Nn_PreprocessingLayer.cc
                Nn_Statistics.cc
    )
endif()

if(${MODULE_OPENMP})
    target_sources(unit-test PRIVATE Math_MultithreadingHelper.cc)
endif()

if(${MODULE_TBB})
    target_sources(unit-test PRIVATE Core_Tbb.cc)
endif()

target_link_libraries(
    unit-test
    PRIVATE RasrBliss
            RasrCore
            RasrFsa
            RasrLm
            RasrMath
            RasrMm
            RasrNn
            cppunit
)

if(${MODULE_NN})
    target_link_libraries(unit-test PRIVATE RasrNn)
endif()
