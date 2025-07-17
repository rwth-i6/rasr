function(add_module_option MODULE_NAME DEFAULT_VALUE)
    option(${MODULE_NAME} "Enable module ${MODULE_NAME}" ${DEFAULT_VALUE})
    if(${MODULE_NAME})
        add_compile_definitions(${MODULE_NAME})
        message(STATUS "Module ${MODULE_NAME} is enabled")
    else()
        message(STATUS "Module ${MODULE_NAME} disabled")
    endif()
endfunction()

set(TOOLS "")

function(add_tool_option TOOL_NAME DEFAULT_VALUE)
    option(${TOOL_NAME} "Enable tool ${TOOL_NAME}" ${DEFAULT_VALUE})
    if(${TOOL_NAME})
        set(TOOLS
            ${TOOLS} ${TOOL_NAME}
            PARENT_SCOPE
        )
        message(STATUS "Tool ${TOOL_NAME} is enabled")
    else()
        message(STATUS "Tool ${TOOL_NAME} is disabled")
    endif()
endfunction()

# ****** Adaptation ******
add_module_option(MODULE_ADAPT_CMLLR ON)
add_module_option(MODULE_ADAPT_MLLR ON)
add_module_option(MODULE_ADAPT_ADVANCED ON)

# ****** Audio ******
add_module_option(MODULE_AUDIO_FFMPEG OFF)
add_module_option(MODULE_AUDIO_FLAC ON)
add_module_option(MODULE_AUDIO_OTK ON)
add_module_option(MODULE_AUDIO_OSS ON)
add_module_option(MODULE_AUDIO_RAW ON)
add_module_option(MODULE_AUDIO_WAV_SYSTEM ON)

# ****** Cache Manager integration ******
add_module_option(MODULE_CORE_CACHE_MANAGER ON)

# ****** Cart ******
add_module_option(MODULE_CART ON)

# ****** Flf ******
add_module_option(MODULE_FLF_CORE ON)
add_module_option(MODULE_FLF_EXT ON)
add_module_option(MODULE_FLF ON)

# ****** Lattice ******
add_module_option(MODULE_LATTICE_BASIC ON)
add_module_option(MODULE_LATTICE_HTK ON)
add_module_option(MODULE_LATTICE_DT ON)

# ****** Lm ******
add_module_option(MODULE_LM_ARPA ON)
add_module_option(MODULE_LM_FSA ON)
add_module_option(MODULE_LM_ZEROGRAM ON)
add_module_option(MODULE_LM_FFNN ON)
add_module_option(MODULE_LM_TFRNN ON)

# ****** Math ******
add_module_option(MODULE_MATH_NR ON)

# ****** Mm ******
add_module_option(MODULE_MM_BATCH ON)
add_module_option(MODULE_MM_DT ON)

# ****** Neural Network ******
add_module_option(MODULE_NN ON)
add_module_option(MODULE_NN_SEQUENCE_TRAINING ON)
add_module_option(MODULE_PYTHON ON)

# ****** OpenFst ******
add_module_option(MODULE_OPENFST OFF)

# ****** Search ******
add_module_option(MODULE_SEARCH_MBR ON)
add_module_option(MODULE_SEARCH_WFST OFF)
add_module_option(MODULE_SEARCH_LINEAR ON)
add_module_option(MODULE_ADVANCED_TREE_SEARCH ON)

# ****** Signal ******
add_module_option(MODULE_SIGNAL_GAMMATONE ON)
add_module_option(MODULE_SIGNAL_PLP ON)
add_module_option(MODULE_SIGNAL_VTLN ON)
add_module_option(MODULE_SIGNAL_VOICEDNESS ON)
add_module_option(MODULE_SIGNAL_ADVANCED ON)
add_module_option(MODULE_SIGNAL_ADVANCED_NR ON)

# ****** Speech ******
add_module_option(MODULE_SPEECH_DT ON)
add_module_option(MODULE_SPEECH_DT_ADVANCED ON)
add_module_option(MODULE_SPEECH_ALIGNMENT_FLOW_NODES ON)
add_module_option(MODULE_SPEECH_LATTICE_FLOW_NODES ON)
add_module_option(MODULE_SPEECH_LATTICE_ALIGNMENT ON)
add_module_option(MODULE_SPEECH_LATTICE_RESCORING ON)

# ****** Unit Tests ******
add_module_option(MODULE_TEST OFF)

# ****** Intel Threading Building Blocks ******
add_module_option(MODULE_TBB OFF)

# ****** OpenMP library ******
add_module_option(MODULE_OPENMP OFF)
# **** choose optimized blas library if available ******
add_module_option(MODULE_INTEL_MKL OFF)
add_module_option(MODULE_ACML OFF)
add_module_option(MODULE_CUDA ON)

# ****** Tensorflow integration ******
add_module_option(MODULE_TENSORFLOW ON)

# ****** ONNX integration ******
add_module_option(MODULE_ONNX ON)

# ****** Tools ******
add_tool_option(AcousticModelTrainer ON)
add_tool_option(Archiver ON)
add_tool_option(CorpusStatistics ON)
add_tool_option(FeatureExtraction ON)
add_tool_option(FeatureStatistics ON)
add_tool_option(Fsa ON)
add_tool_option(Lm ON)
add_tool_option(SpeechRecognizer ON)
add_tool_option(Xml ON)

if(${MODULE_PYTHON})
    add_tool_option(LibRASR ON)
endif()

if(${MODULE_CART})
    add_tool_option(Cart ON)
endif()

if(${MODULE_MM_DT}
   AND ${MODULE_LATTICE_DT}
   AND ${MODULE_SPEECH_DT}
)
    add_tool_option(LatticeProcessor ON)
endif()

if(${MODULE_FLF})
    add_tool_option(Flf ON)
endif()

if(${MODULE_NN})
    add_tool_option(NnTrainer ON)
endif()

set(SEARCH_LIBS RasrLexiconfreeTimesyncBeamSearch RasrLexiconfreeLabelsyncBeamSearch RasrTreeTimesyncBeamSearch)
if(${MODULE_SEARCH_WFST})
    list(APPEND SEARCH_LIBS RasrSearchWfst)
endif()
if(${MODULE_ADVANCED_TREE_SEARCH})
    list(APPEND SEARCH_LIBS RasrAdvancedTreeSearch)
endif()
