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
#ifndef _AUDIO_FLAC_HH
#define _AUDIO_FLAC_HH

#include "Node.hh"
#include "flac/FlacDecoder.hh"

namespace Audio {

/** Flow node for reading FLAC audio files */
class FlacInputNode : public SourceNode {
private:
    FlacDecoder* fd_;

    virtual bool isFileOpen() const {
        return fd_ != 0;
    }
    virtual bool openFile_();
    virtual void closeFile_();
    virtual bool seek(SampleCount newSamplePos);
    template<typename T>
    u32         readTyped(u32 nSamples, Flow::Timestamp*& d);
    virtual u32 read(u32 nSamples, Flow::Timestamp*& d);

public:
    static std::string filterName() {
        return "audio-input-file-flac";
    }
    FlacInputNode(const Core::Configuration& c);
};

}  // namespace Audio

#endif  // _AUDIO_FLAC_HH
