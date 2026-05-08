add_library(
    RasrMathNr STATIC
    BesselFunctions.cc
    ConjugateGradient.cc
    FastFourierTransform.cc
    integration.cc
    ludcmp.cc
    PowerSpectrum.cc
    pythag.cc
    Random.cc
    svdcmp.cc
)

target_link_libraries(RasrMathNr PUBLIC RasrCore RasrMath)

add_rasr_check_executable(check-math-nr check.cc)

target_link_rasr_check_libraries(check-math-nr PRIVATE RasrMathNr)
