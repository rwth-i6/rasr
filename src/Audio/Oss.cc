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
#include "Oss.hh"

#include <fcntl.h>
#include <math.h>
#include <sys/ioctl.h>
#include <sys/soundcard.h>
#include <unistd.h>

// #include <cc++/file.h>

using namespace Audio;

// ===========================================================================
const Core::ParameterString Audio::OpenSoundSystemDevice::paramDevice(
        "device", "name of audio device", "/dev/dsp");

OpenSoundSystemDevice::OpenSoundSystemDevice(const Core::Configuration& c)
        : Core::Component(c),
          Node(c),
          fd_(-1) {
    filename_ = paramDevice(config);
}

bool OpenSoundSystemDevice::openDevice() {
    require(!isDeviceOpen());

    fd_ = open(filename_.c_str(), O_RDWR);
    if (fd_ == -1) {
        error("cannot open DSP sound device \"%s\"", filename_.c_str());
        return false;
    }

    if (!(setDeviceFormat(sampleSize_) && setDeviceTracks(trackCount_) && setDeviceSampleRate(sampleRate_))) {
        closeFile();
        return false;
    }

    return true;
}

bool OpenSoundSystemDevice::setDeviceTracks(u8& _trackCount) {
    require(isDeviceOpen());

    int actualTrackCount = _trackCount;
    if (ioctl(fd_, SNDCTL_DSP_CHANNELS, &actualTrackCount) == -1) {
        error("during ioctl(SNDCTL_DSP_CHANNELS)");
        return false;
    }
    if (actualTrackCount != _trackCount) {
        warning("sound device does not support %d tracks", _trackCount);
        warning("sound device uses %d tracks instead.", actualTrackCount);
    }
    _trackCount = actualTrackCount;
    return true;
}

bool OpenSoundSystemDevice::setDeviceFormat(u8& _sampleSize) {
    require(isDeviceOpen());

    int format;
    switch (_sampleSize) {
        case 8:
            format = AFMT_U8;  // Shouldn't we use AFMT_S8 ?
            break;
        case 16:
#if __BYTE_ORDER == __LITTLE_ENDIAN
            format = AFMT_S16_LE;
#elif __BYTE_ORDER == __BIG_ENDIAN
            format = AFMT_S16_BE;
#else
#error "Machine has unsupported byte order!"
#endif
            break;
        default:
            error("unsupported sample size: %d bit", _sampleSize);
            return false;
    }

    int actualFormat = format;
    if (ioctl(fd_, SNDCTL_DSP_SETFMT, &actualFormat) == -1) {
        error("during ioctl(SNDCTL_DSP_SETFMT)");
        closeDevice();
        return false;
    }
    if (actualFormat != format) {
        warning("sound device does not support sample format %d", format);
        warning("sound device uses format %d instead.", actualFormat);
    }
    return true;
}

bool OpenSoundSystemDevice::setDeviceSampleRate(Flow::Time& _sampleRate) {
    require(isDeviceOpen());

    int actualSampleRate = int(rint(_sampleRate));
    if (ioctl(fd_, SNDCTL_DSP_SPEED, &actualSampleRate) == -1) {
        error("during ioctl(SNDCTL_DSP_SPEED)");
        return false;
    }
    if (actualSampleRate != int(_sampleRate)) {
        warning("sample rate of %fHz not supported", sampleRate_);
        warning("sound device uses %dHz instead", actualSampleRate);
    }
    _sampleRate = actualSampleRate;
    return true;
}

void OpenSoundSystemDevice::closeDevice() {
    require(fd_ >= 0);
    ::close(fd_);
    fd_ = -1;
}

void OpenSoundSystemDevice::setSampleRate(Flow::Time _sampleRate) {
    if (isDeviceOpen())
        setDeviceSampleRate(_sampleRate);
    Node::setSampleRate(_sampleRate);
}

void OpenSoundSystemDevice::setSampleSize(u8 _sampleSize) {
    if (isDeviceOpen())
        setDeviceFormat(_sampleSize);
    Node::setSampleSize(_sampleSize);
}

void OpenSoundSystemDevice::setTrackCount(u8 _trackCount) {
    if (isDeviceOpen())
        setDeviceTracks(_trackCount);
    Node::setTrackCount(_trackCount);
}

// ===========================================================================
OpenSoundSystemInputNode::OpenSoundSystemInputNode(const Core::Configuration& c)
        : Core::Component(c),
          Node(c),
          RawSourceNode(c),
          OpenSoundSystemDevice(c) {}

bool OpenSoundSystemInputNode::openFile_() {
    return openDevice();
}

template<typename T>
u32 OpenSoundSystemInputNode::readTyped(u32 nSamples, Flow::Timestamp*& d) {
    require(fd_ >= 0);
    Flow::Vector<T>* v           = new Flow::Vector<T>(nSamples * trackCount_);
    size_t           bytesToRead = nSamples * trackCount_ * sizeof(T);
    ssize_t          bytesRead   = ::read(fd_, (void*)(&*(v->begin())), bytesToRead);
    if (bytesRead <= 0) {
        if (bytesRead < 0)
            error("read failed");
        delete v;
        d = 0;
        return 0;
    }
    v->resize(bytesRead / sizeof(T));
    d = v;
    return v->size() / trackCount_;
}

u32 OpenSoundSystemInputNode::read(u32 nSamples, Flow::Timestamp*& d) {
    require(isFileOpen());
    require(d == 0);
    switch (sampleSize_) {
        case 8:
            return readTyped<s8>(nSamples, d);
        case 16:
            return readTyped<s16>(nSamples, d);
        default:
            error("unsupported sample size: %d bit", sampleSize_);
            return false;
    }
}

// ===========================================================================
OpenSoundSystemOutputNode::OpenSoundSystemOutputNode(const Core::Configuration& c)
        : Core::Component(c),
          Node(c),
          SinkNode(c),
          OpenSoundSystemDevice(c) {}

bool OpenSoundSystemOutputNode::openFile_() {
    return openDevice();
}

template<typename T>
bool OpenSoundSystemOutputNode::writeTyped(const Flow::Data* _in) {
    const Flow::Vector<T>* in(dynamic_cast<const Flow::Vector<T>*>(_in));
    ssize_t                bytesToWrite = sizeof(T) * in->size();
    ssize_t                bytesWritten = ::write(fd_, &*(in->begin()), bytesToWrite);
    if (bytesWritten < 0)
        error("write failed");
    return (bytesWritten == bytesToWrite);
}

bool OpenSoundSystemOutputNode::write(const Flow::Data* in) {
    /*! \todo Should look at datatype not at sample size. */
    switch (sampleSize_) {
        case 8:
            return writeTyped<s8>(in);
        case 16:
            return writeTyped<s16>(in);
        default:
            error("unsupported sample size: %d bit", sampleSize_);
            return false;
    }
}
