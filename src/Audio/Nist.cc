#include "Nist.hh"
#include <Flow/Vector.hh>

namespace Audio {
namespace NistPrivate {
extern "C" {
#include "nist/sp/sphere.h"
}
}  // namespace NistPrivate
}  // namespace Audio

using namespace Audio;
using namespace NistPrivate;

// ===========================================================================
struct NistInputNode::Handle : public NistPrivate::SP_FILE {};

NistInputNode::NistInputNode(const Core::Configuration& c)
        : Core::Component(c), Node(c), SourceNode(c), spf_(0) {}

bool NistInputNode::openFile_() {
    spf_ = (Handle*)sp_open((char*)filename_.c_str(), (char*)"r");
    if (spf_ == (SP_FILE*)0) {
        error("could not open nist file '%s' for reading", filename_.c_str());
        return false;
    }

    long v;

    sp_h_get_field(spf_, (char*)"sample_rate", T_INTEGER, (void**)&v);
    setSampleRate(v);

    sp_h_get_field(spf_, (char*)"sample_n_bytes", T_INTEGER, (void**)&v);
    setSampleSize(v * 8);

    sp_h_get_field(spf_, (char*)"channel_count", T_INTEGER, (void**)&v);
    setTrackCount(v);

    sp_h_get_field(spf_, (char*)"sample_count", T_INTEGER, (void**)&v);
    setTotalSampleCount(v);

    return true;
}

void NistInputNode::closeFile_() {
    sp_close(spf_);
    spf_ = 0;
}

bool NistInputNode::seek(SampleCount newSamplePos) {
    require(isFileOpen());
    if (sp_seek(spf_, newSamplePos, 0) != 0) {
        error("sp_seek failed");
        return false;
    }
    sampleCount_ = newSamplePos;
    return true;
}

template<typename T>
u32 NistInputNode::readTyped(u32 nSamples, Flow::Timestamp*& d) {
    require(spf_);
    Flow::Vector<T>* v                = new Flow::Vector<T>(nSamples * trackCount_);
    u32              samplesRead      = 0;
    int              sphereReturnCode = sp_read_data(&*(v->begin()), nSamples, spf_);
    if (sphereReturnCode <= 0) {
        if (!sp_eof(spf_))
            error("sp_read_data failed on file '%s' (code %d)", filename_.c_str(), sp_error(spf_));
        delete v;
        d = 0;
    }
    else {
        samplesRead = sphereReturnCode;
        v->resize(samplesRead * trackCount_);
        d = v;
    }
    ensure(samplesRead <= nSamples);
    return samplesRead;
}

u32 NistInputNode::read(u32 nSamples, Flow::Timestamp*& d) {
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
