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
        Test_Lexicon.cc)

if (${MODULE_NN})
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
            Nn_Statistics.cc)
endif ()

if (${MODULE_OPENMP})
    list(APPEND test_SOURCES Math_MultithreadingHelper.cc)
endif ()

if (${MODULE_TBB})
    list(APPEND test_SOURCES Core_Tbb.cc)
endif ()

add_executable(unit-test ${test_SOURCES})
add_install_executable(unit-test)

target_link_libraries(
        unit-test
        PRIVATE RasrTest
        RasrBliss
        RasrFsa
        RasrCore
        RasrSpeech
        RasrSearch
        RasrLattice
        RasrAm
        RasrLm
        RasrMc
        RasrAudio
        RasrSignal
        RasrFlow
        RasrMath
        RasrMathLapack)

if (${MODULE_FLF})
    target_link_libraries(unit-test PRIVATE RasrFlf)
endif ()

if (${MODULE_FLF_CORE})
    target_link_libraries(unit-test PRIVATE RasrFlfCore)
endif ()

if (${MODULE_FLF_EXT})
    target_link_libraries(unit-test PRIVATE RasrFlfExt)
endif ()

if (${MODULE_ADVANCED_TREE_SEARCH})
    target_link_libraries(unit-test PRIVATE RasrAdvancedTreeSearch)
endif ()

if (${MODULE_GENERIC_SEQ2SEQ_TREE_SEARCH})
    target_link_libraries(unit-test PRIVATE RasrGenericSeq2SeqTreeSearch)
endif ()

if (${MODULE_PYTHON})
    target_link_libraries(unit-test PRIVATE RasrPython)
    add_python_dependencies(unit-test)
endif ()

if (${MODULE_NN})
    target_link_libraries(unit-test PRIVATE RasrNn)
endif ()

if (${MODULE_CART})
    target_link_libraries(unit-test PRIVATE RasrCart)
endif ()

if (${MODULE_MATH_NR})
    target_link_libraries(unit-test PRIVATE RasrMathNr)
endif ()

if (${MODULE_SEARCH_WFST})
    target_link_libraries(unit-test PRIVATE RasrSearchWfst)
endif ()

if (${MODULE_OPENFST})
    target_link_libraries(unit-test PRIVATE RasrOpenFst)
endif ()

if (${MODULE_TENSORFLOW})
    target_link_libraries(unit-test PRIVATE RasrTensorflow)
    add_tf_dependencies(unit-test)
endif ()
