add_library(
    RasrSignal STATIC
    CosineTransform.cc
    DcDetection.cc
    Delay.cc
    Delimiter.cc
    EigenTransform.cc
    FastFourierTransform.cc
    Filterbank.cc
    FramePrediction.cc
    Module.cc
    Mrasta.cc
    Normalization.cc
    Preemphasis.cc
    Regression.cc
    RepeatingFramePrediction.cc
    ScatterEstimator.cc
    ScatterTransform.cc
    SegmentClustering.cc
    SegmentEstimator.cc
    SilenceNormalization.cc
    TempoRAlPattern.cc
    VectorResize.cc
    VectorSequenceAggregation.cc
    VectorSequenceConcatenation.cc
    Window.cc
    WindowBuffer.cc
    WindowFunction.cc
)

if(${MODULE_SIGNAL_GAMMATONE})
    target_sources(
        RasrSignal PRIVATE GammaTone.cc SpectralIntegration.cc
                           TemporalIntegration.cc TimeWindowBuffer.cc
    )
endif()

if(${MODULE_SIGNAL_VTLN})
    target_sources(
        RasrSignal PRIVATE LikelihoodFunction.cc BayesClassification.cc
                           AprioriProbability.cc
    )
endif()

if(${MODULE_SIGNAL_VOICEDNESS})
    target_sources(RasrSignal PRIVATE CrossCorrelation.cc PeakDetection.cc)
endif()

if(${MODULE_SIGNAL_PLP})
    target_sources(
        RasrSignal
        PRIVATE ArEstimator.cc VectorTransform.cc AutoregressionToCepstrum.cc
                AutoregressionToSpectrum.cc
    )
endif()

if(${MODULE_SIGNAL_ADVANCED})
    target_sources(
        RasrSignal
        PRIVATE ArxEstimator.cc
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
    target_sources(
        RasrSignal PRIVATE AllPolesPowerSpectrum.cc KaiserWindowFunction.cc
                           RandomVector.cc WindowingFirFilter.cc
    )
endif()

target_link_libraries(
    RasrSignal
    PUBLIC RasrFlow
           RasrBliss
           RasrMm
           RasrMath
           RasrMathLapack
           RasrMc
           RasrCore
)

if(${MODULE_SIGNAL_ADVANCED_NR})
    target_link_libraries(RasrSignal PUBLIC RasrCore RasrFlow RasrMath RasrMm)
endif()

add_rasr_check_executable(check-signal check.cc)

target_link_rasr_check_libraries(check-signal PRIVATE RasrSignal)
