set(SENTENCEPIECE_ROOT
    ""
    CACHE PATH "Optional root directory of a SentencePiece installation"
)

find_path(
    sentencepiece_INCLUDE_DIR
    NAMES sentencepiece_processor.h
    HINTS ${SENTENCEPIECE_ROOT}
    PATH_SUFFIXES include
)

find_library(
    sentencepiece_LIBRARY
    NAMES sentencepiece
    HINTS ${SENTENCEPIECE_ROOT}
    PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    SentencePiece REQUIRED_VARS sentencepiece_LIBRARY sentencepiece_INCLUDE_DIR
)

add_library(SentencePiece::SentencePiece UNKNOWN IMPORTED)
set_target_properties(
    SentencePiece::SentencePiece
    PROPERTIES IMPORTED_LOCATION "${sentencepiece_LIBRARY}"
               INTERFACE_INCLUDE_DIRECTORIES "${sentencepiece_INCLUDE_DIR}"
)
