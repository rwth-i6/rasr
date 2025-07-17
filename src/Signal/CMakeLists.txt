set(RasrSignal_SOURCES
    CosineTransform.cc
    DcDetection.cc
    SilenceNormalization.cc
    Delay.cc
    Delimiter.cc
    EigenTransform.cc
    FramePrediction.cc
    FastFourierTransform.cc
    Filterbank.cc
    Module.cc
    Mrasta.cc
    Normalization.cc
    Preemphasis.cc
    RepeatingFramePrediction.cc
    Regression.cc
    ScatterEstimator.cc
    ScatterTransform.cc
    SegmentClustering.cc
    SegmentEstimator.cc
    TempoRAlPattern.cc
    VectorSequenceAggregation.cc
    VectorSequenceConcatenation.cc
    VectorResize.cc
    Window.cc
    WindowBuffer.cc
    WindowFunction.cc
)

if(${MODULE_SIGNAL_GAMMATONE})
    list(APPEND RasrSignal_SOURCES GammaTone.cc SpectralIntegration.cc TemporalIntegration.cc TimeWindowBuffer.cc)
endif()

if(${MODULE_SIGNAL_VTLN})
    list(APPEND RasrSignal_SOURCES LikelihoodFunction.cc BayesClassification.cc AprioriProbability.cc)
endif()

if(${MODULE_SIGNAL_VOICEDNESS})
    list(APPEND RasrSignal_SOURCES CrossCorrelation.cc PeakDetection.cc)
endif()

if(${MODULE_SIGNAL_PLP})
    list(APPEND RasrSignal_SOURCES ArEstimator.cc VectorTransform.cc AutoregressionToCepstrum.cc
         AutoregressionToSpectrum.cc
    )
endif()

if(${MODULE_SIGNAL_ADVANCED})
    list(
        APPEND
        RasrSignal_SOURCES
        ArxEstimator.cc
        Convolution.cc
        FastHartleyTransform.cc
        Formant.cc
        FrameInterpolation.cc
        GenericWarping.cc
        HarmonicSum.cc
        Histogram.cc
        HistogramNormalization.cc
        LinearFilter.cc
        LinearWarping.cc
        Lpc.cc
        MeanEstimator.cc
        NthOrderFeatures.cc
        PolinomialVectorInterpolation.cc
        QuantileEqualization.cc
        SampleNormalization.cc
        SegmentwiseFormantExtraction.cc
        SilenceDetection.cc
        Warping.cc
    )
endif()

if(${MODULE_SIGNAL_ADVANCED_NR})
    list(APPEND RasrSignal_SOURCES AllPolesPowerSpectrum.cc KaiserWindowFunction.cc RandomVector.cc
         WindowingFirFilter.cc
    )
endif()

add_library(RasrSignal STATIC ${RasrSignal_SOURCES})

target_link_libraries(RasrSignal PUBLIC RasrCore RasrFlow RasrMath RasrMm)

if(${MODULE_SIGNAL_ADVANCED_NR})
    target_link_libraries(RasrSignal PUBLIC RasrMathNr)
endif()

add_executable(check-signal check.cc)
target_link_libraries(check-signal PRIVATE RasrSignal)
