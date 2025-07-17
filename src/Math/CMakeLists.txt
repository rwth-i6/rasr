add_subdirectory(Lapack)

if(${MODULE_MATH_NR})
    add_subdirectory(Nr)
endif()

set(RasrMath_SOURCES
    AcousticalAnalyticFunctions.cc
    AnalyticFunctionFactory.cc
    EigenvalueProblem.cc
    FastFourierTransform.cc
    Module.cc
    PiecewiseLinearFunction.cc
    Random.cc
    CudaDataStructure.cc
)

if(${MODULE_CUDA})
    list(APPEND RasrMath_SOURCES CudaMatrixKernels.cu)
endif()

if(${MODULE_SIGNAL_PLP})
    list(APPEND RasrMath_SOURCES LevinsonLse.cc)
endif()

add_library(RasrMath STATIC ${RasrMath_SOURCES})

if(${MODULE_CUDA})
    add_cuda_dependencies(RasrMath)
endif()

target_link_libraries(RasrMath PUBLIC RasrCore RasrMathLapack RasrMm)

if(${MODULE_MATH_NR})
    target_link_libraries(RasrMath PUBLIC RasrMathNr)
endif()

# check.cc code is broken

# add_executable(check-math check.cc) target_link_libraries(check-math PRIVATE RasrMath)
