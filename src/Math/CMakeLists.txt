add_subdirectory(Lapack)
if(${MODULE_MATH_NR})
    add_subdirectory(Nr)
endif()

add_library(
    RasrMath STATIC
    AcousticalAnalyticFunctions.cc
    AnalyticFunctionFactory.cc
    CudaDataStructure.cc
    EigenvalueProblem.cc
    FastFourierTransform.cc
    Module.cc
    PiecewiseLinearFunction.cc
    Random.cc
)

if(${MODULE_CUDA})
    target_sources(RasrMath PRIVATE CudaMatrixKernels.cu)

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(
        RasrMath PUBLIC CUDA::cublas CUDA::cudart CUDA::curand
    )
endif()

if(${MODULE_SIGNAL_PLP})
    target_sources(RasrMath PRIVATE LevinsonLse.cc)
endif()

target_link_libraries(RasrMath PUBLIC RasrCore RasrMathLapack RasrMm)
if(${MODULE_MATH_NR})
    target_link_libraries(RasrMath PUBLIC RasrMathNr)
endif()

# check.cc code is broken add_executable(check-math check.cc)
# target_link_libraries(check-math PRIVATE RasrMath)
