set(Audio_SOURCES Module.cc Node.cc)

if (${MODULE_AUDIO_FFMPEG})
    list(APPEND Audio_SOURCES Ffmpeg.cc)
endif ()

if (${MODULE_AUDIO_FLAC})
    list(APPEND Audio_SOURCES Flac.cc flac/FlacDecoder.cc)
endif ()

if (${SYSTEM_NAME} STREQUAL "linux" AND ${MODULE_AUDIO_OSS})
    list(APPEND Audio_SOURCES Oss.cc)
endif ()

if (${MODULE_AUDIO_RAW})
    list(APPEND Audio_SOURCES Raw.cc)
endif ()

if (${MODULE_AUDIO_WAV_SYSTEM})
    list(APPEND Audio_SOURCES Wav.cc)
endif ()

add_library(RasrAudio STATIC ${Audio_SOURCES})

target_link_libraries(RasrAudio INTERFACE RasrFlow RasrCore)

add_executable(check-audio check.cc)
target_link_libraries(check-audio PRIVATE RasrAudio)
