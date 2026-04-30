add_library(RasrMathLapack STATIC EigenvalueProblem.cc Lapack.cc Svd.cc)

target_link_libraries(RasrMathLapack PUBLIC RasrCore RasrMath)

add_executable(check-math-lapack check.cc)

target_link_libraries(check-math-lapack PRIVATE RasrMathLapack)
