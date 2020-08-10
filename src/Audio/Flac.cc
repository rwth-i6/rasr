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
#include "Flac.hh"
#include <Flow/Vector.hh>
#include "flac/FlacDecoder.hh"

using namespace Audio;

// ===========================================================================

FlacInputNode::FlacInputNode(const Core::Configuration& c)
        : Core::Component(c), Node(c), SourceNode(c), fd_(0) {}

bool FlacInputNode::openFile_() {
    fd_ = new FlacDecoder();
    if (!fd_->open(filename_.c_str())) {
        error("could not open FLAC file '%s' for reading", filename_.c_str());
        delete fd_;
        return false;
    }

    // init sample format
    setSampleRate(fd_->getSampleRate());
    setSampleSize(fd_->getBitsPerSample());
    setTrackCount(fd_->getChannels());
    setTotalSampleCount(fd_->getTotalSamples());

    return true;
}

void FlacInputNode::closeFile_() {
    delete fd_;
    fd_ = nullptr;
}

template<typename T>
u32 FlacInputNode::readTyped(u32 nSamples, Flow::Timestamp*& d) {
    require(isFileOpen());
    require(d == 0);
    Flow::Vector<T>* v           = new Flow::Vector<T>(trackCount_ * nSamples);
    int              samplesRead = fd_->read(nSamples, &*(v->begin()));
    int              bytesRead   = samplesRead * trackCount_ * sizeof(T);
    if (bytesRead <= 0) {
        if (bytesRead < 0)
            error("FlacDecode::read failed");
        delete v;
        d = 0;
        return 0;
    }
    v->resize(bytesRead / sizeof(T));
    d = v;
    return v->size() / trackCount_;
}

u32 FlacInputNode::read(u32 nSamples, Flow::Timestamp*& d) {
    require(isFileOpen());
    require(d == 0);
    switch (sampleSize_) {
        case 16:
            return readTyped<s16>(nSamples, d);
        default:
            error("unsupported sample size: %d bit", sampleSize_);
            return false;
    }
}

bool FlacInputNode::seek(SampleCount newSamplePos) {
    require(isFileOpen());
    if (!fd_->seek(newSamplePos)) {
        error("FileDecoder seek failed");
        return false;
    }
    sampleCount_ = newSamplePos;
    return true;
}
