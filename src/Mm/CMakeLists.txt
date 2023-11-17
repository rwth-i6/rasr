set(RasrMm_SOURCES
        AbstractMixtureSetEstimator.cc
        CombinedFeatureScorer.cc
        CovarianceFeatureScorerElement.cc
        CovarianceWeightedFeatureScorerElement.cc
        Feature.cc
        GaussDensity.cc
        GaussDensityEstimator.cc
        GaussDiagonalMaximumFeatureScorer.cc
        IntelCodeGenerator.cc
        IntelOptimization.cc
        Mixture.cc
        MixtureEstimator.cc
        MixtureFeatureScorerElement.cc
        MixtureSet.cc
        MixtureSetBuilder.cc
        MixtureSetEstimator.cc
        MixtureSetLoader.cc
        MixtureSetReader.cc
        MixtureSetSplitter.cc
        MixtureSetTopology.cc
        Module.cc
        StatePosteriorFeatureScorer.cc
        SimdFeatureScorer.cc
        SSE2CodeGenerator.cc
        Utilities.cc
)

if(${MODULE_MM_DT})
    list(APPEND RasrMm_SOURCES
            ConvertGaussDensityEstimator.cc
            ConvertMixtureEstimator.cc
            ConvertMixtureSetEstimator.cc
            DiscriminativeGaussDensityEstimator.cc
            DiscriminativeMixtureEstimator.cc
            DiscriminativeMixtureSetEstimator.cc
            EbwDiscriminativeGaussDensityEstimator.cc
            EbwDiscriminativeMixtureEstimator.cc
            EbwDiscriminativeMixtureSetEstimator.cc
            ISmoothingGaussDensityEstimator.cc
            ISmoothingMixtureEstimator.cc
            ISmoothingMixtureSetEstimator.cc
            IterationConstants.cc
            RpropOptimization.cc
            RpropDiscriminativeGaussDensityEstimator.cc
            RpropDiscriminativeMixtureEstimator.cc
            RpropDiscriminativeMixtureSetEstimator.cc
    )
endif()
if(${MODULE_MM_BATCH})
    list(APPEND RasrMm_SOURCES
            BatchFeatureScorer.cc
            DensityClustering.cc
    )
endif()
if(${MODULE_ADAPT_ADVANCED})
    list(APPEND RasrMm_SOURCES
            BandMllrAdaptation.cc
            SemiTiedAdaptation.cc
    )
endif()
if(${MODULE_ADAPT_MLLR})
    list(APPEND RasrMm_SOURCES
            MllrAdaptation.cc
    )
endif()
if(${MODULE_ADAPT_CMLLR})
    list(APPEND RasrMm_SOURCES
            AffineFeatureTransformAccumulator.cc
    )
endif()

add_library(RasrMm STATIC ${RasrMm_SOURCES})

target_link_libraries(RasrMm INTERFACE
        RasrMath
        RasrMathLapack
        RasrFsa
        RasrCore
        RasrMc
)

if(${MODULE_MM_BATCH})
    target_compile_options(RasrMm PRIVATE "-msse2")
endif()
