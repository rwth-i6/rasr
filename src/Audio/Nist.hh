#ifndef _AUDIO_NIST_HH
#define _AUDIO_NIST_HH

#include "Node.hh"

namespace Audio {

/** Flow node for reading NIST Sphere audio files */
class NistInputNode : public SourceNode {
private:
    struct Handle;
    Handle* spf_;

    virtual bool isFileOpen() const {
        return spf_ != 0;
    }
    virtual bool openFile_();
    virtual void closeFile_();
    virtual bool seek(SampleCount newSamplePos);
    template<typename T>
    u32         readTyped(u32 nSamples, Flow::Timestamp*& d);
    virtual u32 read(u32 nSamples, Flow::Timestamp*& d);

public:
    static std::string filterName() {
        return "audio-input-file-nist";
    }
    NistInputNode(const Core::Configuration& c);
};

}  // namespace Audio

#endif  // _AUDIO_NIST_HH
