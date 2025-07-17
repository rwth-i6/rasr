set(RasrTest_SOURCES Registry.cc Lexicon.cc File.cc)

add_library(RasrTest ${RasrTest_SOURCES})

set(test_SOURCES
    UnitTester.cc
    Bliss_SegmentOrdering.cc
    Core_StringUtilities.cc
    Core_Thread.cc
    Core_ThreadPool.cc
    Fsa_Sssp4SpecialSymbols.cc
    Math_Utilities.cc
    Math_Blas.cc
    Math_FastVectorOperations.cc
    Math_CudaVector.cc
    Math_CudaMatrix.cc
    Math_FastMatrix.cc
    Test_File.cc
    Test_Lexicon.cc
)

if(${MODULE_NN})
    list(
        APPEND
        test_SOURCES
        Nn_NetworkTopology.cc
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
    list(APPEND test_SOURCES Math_MultithreadingHelper.cc)
endif()

if(${MODULE_TBB})
    list(APPEND test_SOURCES Core_Tbb.cc)
endif()

add_executable(unit-test ${test_SOURCES})
add_install_executable(unit-test)

target_link_libraries(unit-test PRIVATE RasrBliss RasrCore RasrFsa RasrMath RasrMm RasrNn)
