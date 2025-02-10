#ifndef STREAMING_INPUT_NODE_HH
#define STREAMING_INPUT_NODE_HH
#include <functional>
#include <queue>
#include "Node.hh"

using ByteStreamAppender = std::function<void(std::deque<char>&)>;

namespace Flow {

class InputNode : public Flow::SourceNode {
public:
	using Precursor = Flow::SourceNode;
	static const Core::ParameterInt    paramSampleRate;
	static const Core::Choice          choiceSampleType;
	static const Core::ParameterChoice paramSampleType;
	static const Core::ParameterInt    paramTrackCount;
	static const Core::ParameterInt    paramBlockSize;
	static std::string filterName();
	InputNode(const Core::Configuration& c);
	virtual ~InputNode() = default;
	virtual bool setParameter(const std::string& name, const std::string& value);
	virtual bool configure();
	virtual bool work(Flow::PortId out);
	void setByteStreamAppender(ByteStreamAppender const& bsa);
	bool getEOS() const;
	void setEOS(bool eos);
	bool getEOSReceived() const;
	void setEOSReceived(bool eosReceived);
	bool getResetSampleCount() const;
	void setResetSampleCount(bool resetSampleCount);

private:
	unsigned         sampleRate_;
	Flow::SampleType sampleType_;
	unsigned         trackCount_;
	unsigned         blockSize_;
	ByteStreamAppender byteStreamAppender_;
	std::deque<char>   queue_;
	unsigned           sampleCount_;
	bool               eos_;
	bool               eosReceived_;
	bool               resetSampleCount_;
};

// ---------- inline implementations ----------
inline std::string InputNode::filterName() {
        return "stream-input";
}
inline void InputNode::setByteStreamAppender(ByteStreamAppender const& bsa) {
	byteStreamAppender_ = bsa;
}
inline bool InputNode::getEOS() const {
	return eos_;
}
inline void InputNode::setEOS(bool eos) {
	eos_ = eos;
}
inline bool InputNode::getEOSReceived() const {
	return eosReceived_;
}
inline void InputNode::setEOSReceived(bool eosReceived) {
	eosReceived_ = eosReceived;
}
inline bool InputNode::getResetSampleCount() const {
	return resetSampleCount_;
}
inline void InputNode::setResetSampleCount(bool resetSampleCount) { 
	resetSampleCount_ = resetSampleCount;
}

} // namespace Flow

#endif  // INPUT_NODE_HH
