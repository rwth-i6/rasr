set(RasrGenericSeq2SeqTreeSearch_SOURCES
        Seq2SeqTreeSearch.cc
        Seq2SeqAligner.cc
)

add_library(RasrGenericSeq2SeqTreeSearch STATIC ${RasrGenericSeq2SeqTreeSearch_SOURCES})
target_link_libraries(RasrGenericSeq2SeqTreeSearch INTERFACE
        RasrSearch
        RasrBliss
        RasrFsa
        RasrCore
)
target_compile_options(RasrGenericSeq2SeqTreeSearch PRIVATE
        "-Wno-sign-compare -Winline --param max-inline-insns-auto=10000 --param max-inline-insns-single=10000 --param large-function-growth=25000 --param inline-unit-growth=400"
)
