add_library(
    RasrMm STATIC
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
    SimdFeatureScorer.cc
    SSE2CodeGenerator.cc
    StatePosteriorFeatureScorer.cc
    Utilities.cc
)

if(${MODULE_MM_DT})
    target_sources(
        RasrMm
        PRIVATE ConvertGaussDensityEstimator.cc
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
    target_sources(RasrMm PRIVATE BatchFeatureScorer.cc DensityClustering.cc)
endif()
if(${MODULE_ADAPT_ADVANCED})
    target_sources(RasrMm PRIVATE BandMllrAdaptation.cc SemiTiedAdaptation.cc)
endif()
if(${MODULE_ADAPT_MLLR})
    target_sources(RasrMm PRIVATE MllrAdaptation.cc)
endif()
if(${MODULE_ADAPT_CMLLR})
    target_sources(RasrMm PRIVATE AffineFeatureTransformAccumulator.cc)
endif()

target_link_libraries(RasrMm PUBLIC RasrAm RasrCore RasrMath RasrMc RasrSpeech)

if(${MODULE_MM_BATCH})
    target_compile_options(RasrMm PRIVATE "-msse2")
endif()

add_executable(check-mm check.cc)

target_link_libraries(check-mm PRIVATE RasrMm)
