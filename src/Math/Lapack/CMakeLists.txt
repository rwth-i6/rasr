add_library(RasrMathLapack STATIC EigenvalueProblem.cc Lapack.cc Svd.cc)

target_link_libraries(
    RasrMathLapack PUBLIC RasrCore RasrLapackDependencies RasrMath
)

add_rasr_check_executable(check-math-lapack check.cc)

target_link_rasr_check_libraries(check-math-lapack PRIVATE RasrMathLapack)
