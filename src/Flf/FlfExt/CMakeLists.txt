add_library(
    RasrFlfExt STATIC AcousticAlignment.cc MapDecoder.cc MtConfusionNetwork.cc
                      WindowedLevenshteinDistanceDecoder.cc
)

target_link_libraries(
    RasrFlfExt
    PUBLIC RasrBliss
           RasrCore
           RasrFlf
           RasrFlfCore
           RasrFlow
           RasrFsa
           RasrMm
           RasrSpeech
)
