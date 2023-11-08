set(FlfExt_SOURCES
        AcousticAlignment.cc
        MapDecoder.cc
        MtConfusionNetwork.cc
        WindowedLevenshteinDistanceDecoder.cc
)

add_library(RasrFlfExt STATIC ${FlfExt_SOURCES})