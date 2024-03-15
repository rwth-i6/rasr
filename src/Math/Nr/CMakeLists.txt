set(RasrMathNr_SOURCES
        BesselFunctions.cc
        ConjugateGradient.cc
        FastFourierTransform.cc
        integration.cc
        ludcmp.cc
        pythag.cc
        PowerSpectrum.cc
        Random.cc
        svdcmp.cc
)

add_library(RasrMathNr STATIC ${RasrMathNr_SOURCES})

target_link_libraries(RasrMathNr INTERFACE RasrCore)

add_executable(check-math-nr check.cc)
target_link_libraries(check-math-nr PRIVATE RasrMathNr)