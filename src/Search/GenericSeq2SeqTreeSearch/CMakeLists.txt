set(RasrGenericSeq2SeqTreeSearch_SOURCES
        Seq2SeqAligner.cc
        Seq2SeqTreeSearch.cc
)

add_library(RasrGenericSeq2SeqTreeSearch STATIC ${RasrgenericSeq2SeqTreeSearch_SOURCES})
target_link_libraries(RasrGenericSeq2SeqTreeSearch INTERFACE
        RasrSearch
        RasrBliss
        RasrFsa
        RasrCore
)
