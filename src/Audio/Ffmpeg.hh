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
#ifndef _AUDIO_FFMPEG_HH
#define _AUDIO_FFMPEG_HH

#include <mutex>

#include <Flow/Vector.hh>

#include "Node.hh"

namespace Audio {

/** Flow node for reading Audio files with the ffmpeg library */
class FfmpegInputNode : public SourceNode {
public:
    typedef SourceNode Precursor;

    static Core::ParameterInt paramResampleRate;

    static std::string filterName() {
        return "audio-input-file-ffmpeg";
    }

    FfmpegInputNode(const Core::Configuration& c);
    virtual ~FfmpegInputNode() = default;

    virtual bool setParameter(const std::string& name, const std::string& value);

private:
    struct Internal;
    static std::once_flag ffmpeg_initialized;

    std::unique_ptr<Internal> internal_;
    Flow::Timestamp*          buffer_;
    u32                       resampleRate_;
    s64                       lastSeekTime_;

    virtual bool openFile_();
    virtual void closeFile_();
    virtual bool isFileOpen() const;
    virtual bool seek(SampleCount newSamplePos);
    virtual u32  read(u32 nSamples, Flow::Timestamp*& d);
};

}  // namespace Audio

#endif  // _AUDIO_FFMPEG_HH
