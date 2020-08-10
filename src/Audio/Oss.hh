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
#ifndef _AUDIO_OSS_HH
#define _AUDIO_OSS_HH

#include <Flow/Vector.hh>

#include "Node.hh"
#include "Raw.hh"

namespace Audio {

/**
     * Access to Open Sound System audio device.
     * OSS is the traditional Linux sound driver, and is also
     * available for on many other Unices,
     */
class OpenSoundSystemDevice : public virtual Node {
protected:
    int  fd_;
    bool isDeviceOpen() const {
        return fd_ >= 0;
    }
    bool openDevice();
    bool setDeviceTracks(u8& nTracks);
    bool setDeviceFormat(u8& sampleSize);
    bool setDeviceSampleRate(Flow::Time& sampleRate);
    void closeDevice();

    virtual void closeFile_() {
        closeDevice();
    }

public:
    static const Core::ParameterString paramDevice;
    static std::string                 filterName() {
        return "audio-input-device-oss";
    }
    OpenSoundSystemDevice(const Core::Configuration&);
    virtual bool isFileOpen() const {
        return isDeviceOpen();
    }
    virtual void setSampleRate(Flow::Time _sampleRate);
    virtual void setSampleSize(u8 _sampleSize);
    virtual void setTrackCount(u8 _trackCount);
};

/** Flow node for recording from OSS audio device */
class OpenSoundSystemInputNode : public RawSourceNode,
                                 public OpenSoundSystemDevice {
protected:
    virtual bool openFile_();
    template<typename T>
    u32         readTyped(u32 nSamples, Flow::Timestamp*& d);
    virtual u32 read(u32 nSamples, Flow::Timestamp*& d);

public:
    OpenSoundSystemInputNode(const Core::Configuration& c);
};

/** Flow node for playback on OSS audio device */
class OpenSoundSystemOutputNode : public SinkNode,
                                  public OpenSoundSystemDevice {
protected:
    virtual bool openFile_();
    template<typename T>
    bool         writeTyped(const Flow::Data*);
    virtual bool write(const Flow::Data*);

public:
    static std::string filterName() {
        return "audio-output-device-oss";
    }
    OpenSoundSystemOutputNode(const Core::Configuration& c);
};

}  // namespace Audio

#endif  // _AUDIO_OSS_HH
