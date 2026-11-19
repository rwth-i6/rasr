add_library(
    RasrTokenizer STATIC AbstractTokenizer.cc BpeTokenizer.cc Module.cc
                         TokenizerFactory.cc
)

target_link_libraries(RasrTokenizer PUBLIC RasrCore)

if(${MODULE_SENTENCEPIECE})
    target_sources(RasrTokenizer PRIVATE SentencePieceTokenizer.cc)
    target_link_libraries(RasrTokenizer PUBLIC SentencePiece::SentencePiece)
endif()

if(${MODULE_PYTHON})
    find_package(pybind11 REQUIRED)
    target_sources(RasrTokenizer PRIVATE HuggingFaceTokenizer.cc)
    target_link_libraries(RasrTokenizer PUBLIC RasrPython pybind11::headers)
endif()
