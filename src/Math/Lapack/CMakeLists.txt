set(RasrMathLapack_SOURCES
        Lapack.cc
        EigenvalueProblem.cc
        Svd.cc
)

add_library(RasrMathLapack STATIC ${RasrMathLapack_SOURCES})

target_link_libraries(RasrMathLapack INTERFACE RasrCore)

add_executable(check-math-lapack check.cc)
target_link_libraries(check-math-lapack PRIVATE RasrMathLapack)
