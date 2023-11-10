set(RasrMathLapack_SOURCES
        Lapack.cc
        EigenvalueProblem.cc
        Svd.cc
)

add_library(RasrMathLapack STATIC ${RasrMathLapack_SOURCES})

target_link_libraries(RasrMathLapack INTERFACE RasrCore)