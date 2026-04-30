add_library(RasrAudio STATIC Module.cc Node.cc)

if(${MODULE_AUDIO_FFMPEG})
    target_sources(RasrAudio PRIVATE Ffmpeg.cc)
endif()

if(${MODULE_AUDIO_FLAC})
    target_sources(RasrAudio PRIVATE Flac.cc flac/FlacDecoder.cc)
endif()

if(${SYSTEM_NAME} STREQUAL "linux" AND ${MODULE_AUDIO_OSS})
    target_sources(RasrAudio PRIVATE Oss.cc)
endif()

if(${MODULE_AUDIO_RAW})
    target_sources(RasrAudio PRIVATE Raw.cc)
endif()

if(${MODULE_AUDIO_WAV_SYSTEM})
    target_sources(RasrAudio PRIVATE Wav.cc)
endif()

target_link_libraries(RasrAudio PUBLIC RasrCore RasrFlow)

add_executable(check-audio check.cc)

target_link_libraries(check-audio PRIVATE RasrAudio)
