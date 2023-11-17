/** Copyright 2020 RWTH Aachen University. All rights reserved.
 *
 *  Licensed under the RWTH ASR License (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.hltpr.rwth-aachen.de/rwth-asr/rwth-asr-license.html
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
#include "Module.hh"
#include <Flow/Registry.hh>
#include <Modules.hh>

// predefined filter
#ifdef MODULE_AUDIO_RAW
#include "Raw.hh"
#endif
#ifdef MODULE_AUDIO_WAV_SYSTEM
#include "Wav.hh"
#endif
#ifdef MODULE_AUDIO_FLAC
#include "Flac.hh"
#endif
#ifdef MODULE_AUDIO_NIST
#include "Nist.hh"
#endif
#if defined(MODULE_AUDIO_OSS) && defined(OS_linux)
#include "Oss.hh"
#endif
#ifdef MODULE_AUDIO_FFMPEG
#include "Ffmpeg.hh"
#endif

Audio::Module_::Module_() {
    Flow::Registry::Instance& registry = Flow::Registry::instance();
    // file input filters
#ifdef MODULE_AUDIO_RAW
    registry.registerFilter<RawFileInputNode>();
#endif

#ifdef MODULE_AUDIO_WAV_SYSTEM
    registry.registerFilter<WavInputNode>();
    registry.registerFilter<WavOutputNode>();
#endif

#ifdef MODULE_AUDIO_FLAC
    registry.registerFilter<FlacInputNode>();
#endif

#ifdef MODULE_AUDIO_NIST
    registry.registerFilter<NistInputNode>();
#endif

#if defined(MODULE_AUDIO_OSS) && defined(OS_linux)
    registry.registerFilter<OpenSoundSystemInputNode>();
    registry.registerFilter<OpenSoundSystemOutputNode>();
#endif

#ifdef MODULE_AUDIO_FFMPEG
    registry.registerFilter<FfmpegInputNode>();
#endif
}
