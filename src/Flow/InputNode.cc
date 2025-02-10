#include "InputNode.hh"
#include "Timestamp.hh"
#include "Vector.hh"

using namespace Flow;

namespace {
template<typename T>
Flow::Timestamp* createTimestamp(T const* data, unsigned num_samples) {
    Flow::Vector<T>* out = new Flow::Vector<T>(num_samples);
    std::copy(data, data + num_samples, out->data());
    return out;
}
}  // namespace

const Core::ParameterInt    InputNode::paramSampleRate("sample-rate", "sample rate of input data", 1, 1);
const Core::Choice          InputNode::choiceSampleType("s8", static_cast<unsigned>(Flow::SampleType::SampleTypeS8),
                                                        "u8", static_cast<unsigned>(Flow::SampleType::SampleTypeU8),
                                                        "s16", static_cast<unsigned>(Flow::SampleType::SampleTypeS16),
                                                        "u16", static_cast<unsigned>(Flow::SampleType::SampleTypeU16),
                                                        "f32", static_cast<unsigned>(Flow::SampleType::SampleTypeF32),
                                                        Core::Choice::endMark());
const Core::ParameterChoice InputNode::paramSampleType("sample-type", &choiceSampleType, "data type of the samples", static_cast<unsigned>(Flow::SampleType::SampleTypeU16));
const Core::ParameterInt    InputNode::paramTrackCount("track-count", "number of tracks in the stream", 1, 1);
const Core::ParameterInt    InputNode::paramBlockSize("block-size", "number of samples per flow vector", 4096, 1);

InputNode::InputNode(const Core::Configuration& c)
        : Core::Component(c),
          Precursor(c),
          sampleRate_(paramSampleRate(c)),
          sampleType_(static_cast<Flow::SampleType>(paramSampleType(c))),
          trackCount_(paramTrackCount(c)),
          blockSize_(paramBlockSize(c)),
          byteStreamAppender_(),
          queue_(),
          sampleCount_(0u),
          eos_(true),
          eosReceived_(false) {
}

bool InputNode::setParameter(const std::string& name, const std::string& value) {
    if (paramSampleRate.match(name)) {
        sampleRate_ = paramSampleRate(value);
    }
    else if (paramSampleType.match(name)) {
        sampleType_ = static_cast<Flow::SampleType>(paramSampleType(value));
    }
    else if (paramTrackCount.match(name)) {
        trackCount_ = paramTrackCount(value);
    }
    else if (paramBlockSize.match(name)) {
        blockSize_ = paramBlockSize(value);
    }
    else {
        return Precursor::setParameter(name, value);
    }
    return true;
}

bool InputNode::configure() {
    Core::Ref<Flow::Attributes> a(new Flow::Attributes());
    a->set("sample-rate", sampleRate_);
    a->set("track-count", trackCount_);
    switch (sampleType_) {
        case Flow::SampleType::SampleTypeS8:
            a->set("datatype", Flow::Vector<s8>::type()->name());
            break;
        case Flow::SampleType::SampleTypeU8:
            a->set("datatype", Flow::Vector<u8>::type()->name());
            break;
        case Flow::SampleType::SampleTypeS16:
            a->set("datatype", Flow::Vector<s16>::type()->name());
            break;
        case Flow::SampleType::SampleTypeU16:
            a->set("datatype", Flow::Vector<u16>::type()->name());
            break;
        case Flow::SampleType::SampleTypeF32:
            a->set("datatype", Flow::Vector<f32>::type()->name());
            break;
        default:
            error("unsupported sample type: %d", static_cast<unsigned>(sampleType_));
            return false;
    }
    unsigned sample_size = static_cast<unsigned>(sampleType_) & 0xFF;
    a->set("sample-size", sample_size);
    return putOutputAttributes(0, a);
}

bool InputNode::work(Flow::PortId out) {
    unsigned sample_size = static_cast<unsigned>(sampleType_) & 0xFF;
    if ((not(eos_ and not eosReceived_)) and (queue_.size() < blockSize_ * sample_size)) {
        do {  // at least once call byteStreamAppender because it might remove the eos status
            byteStreamAppender_(queue_);
        } while (queue_.size() < blockSize_ * sample_size and not eos_);
    }
    if (queue_.empty()) {
        if (resetSampleCount_) {
            sampleCount_ = 0ul;
        }
        return putEos(out);
    }
    // remove possible partial samples at EOS
    unsigned full_samples = queue_.size() / sample_size;
    if (eos_ and queue_.size() % sample_size != 0ul) {
        queue_.resize(full_samples * sample_size);
    }
    // remove possible partial samples in case of multi-channel audio
    if (full_samples % trackCount_ != 0ul) {
        full_samples -= full_samples % trackCount_;
        if (eos_) {
            queue_.resize(full_samples * sample_size);
        }
    }
    unsigned num_samples = std::min<unsigned>(blockSize_, full_samples);
    std::vector<char> buffer(num_samples * sample_size);
    std::copy(queue_.begin(), queue_.begin() + num_samples * sample_size, buffer.begin());
    for (size_t i = 0ul; i < num_samples * sample_size; i++) {
        queue_.front() = 0;  // erase data
        queue_.pop_front();
    }
    Flow::Timestamp* v = nullptr;
    switch (sampleType_) {
        case Flow::SampleType::SampleTypeS8:
            v = createTimestamp<s8>(reinterpret_cast<s8*>(buffer.data()), num_samples);
            break;
        case Flow::SampleType::SampleTypeU8:
            v = createTimestamp<u8>(reinterpret_cast<u8*>(buffer.data()), num_samples);
            break;
        case Flow::SampleType::SampleTypeS16:
            v = createTimestamp<s16>(reinterpret_cast<s16*>(buffer.data()), num_samples);
            break;
        case Flow::SampleType::SampleTypeU16:
            v = createTimestamp<u16>(reinterpret_cast<u16*>(buffer.data()), num_samples);
            break;
        case Flow::SampleType::SampleTypeF32:
            v = createTimestamp<f32>(reinterpret_cast<f32*>(buffer.data()), num_samples);
            break;
        default:
            error("unsupported sample type: %d", static_cast<unsigned>(sampleType_));
            return false;
    }
    for (unsigned i = 0ul; i < num_samples; i++) {
        std::fill(buffer.begin(), buffer.end(), 0);  // erase data
    }
    v->setStartTime(Flow::Time(sampleCount_) / Flow::Time(sampleRate_) / Flow::Time(trackCount_));
    sampleCount_ += num_samples;
    v->setEndTime(Flow::Time(sampleCount_) / Flow::Time(sampleRate_) / Flow::Time(trackCount_));
    return putData(out, v);
}
