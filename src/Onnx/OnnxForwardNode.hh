/*
 * Copyright 2022 AppTek LLC. All rights reserved.
 */
#ifndef _ONNX_FORWARD_NODE_HH
#define _ONNX_FORWARD_NODE_HH

#include <deque>

#include <Flow/Attributes.hh>
#include <Flow/Datatype.hh>
#include <Flow/Node.hh>
#include <Flow/Timestamp.hh>
#include <Mm/Types.hh>

#include "IOSpecification.hh"
#include "Session.hh"
#include "Value.hh"

namespace Onnx {

class OnnxForwardNode : public Flow::SleeveNode {
public:
    typedef Flow::SleeveNode Precursor;

    static Core::ParameterString paramId;

    static std::string filterName();

    explicit OnnxForwardNode(Core::Configuration const& c);
    virtual ~OnnxForwardNode() = default;

    bool setParameter(const std::string& name, const std::string& value) override;
    bool work(Flow::PortId p) override;

protected:
    bool computation_done_;  // current segment finished yes/no

    // onnx related members
    Session                                         session_;
    static const std::vector<Onnx::IOSpecification> ioSpec_;  // currently fixed to "features", "feature-size" and "output"
    const IOMapping                                 mapping_;
    IOValidator                                     validator_;

    // session-run related members
    const std::string              features_onnx_name_;
    const std::string              features_size_onnx_name_;
    const std::vector<std::string> output_onnx_names_;

    std::deque<Flow::Timestamp> timestamps_;
    std::deque<Flow::Data*>     output_cache_;
    size_t                      current_output_frame_;

    // Helpers to convert from flow input data to an ONNX value
    Onnx::Value toValue(std::deque<Flow::DataPtr<Flow::Timestamp>> const& data) const;
    template<typename T>
    Onnx::Value vectorToValue(std::deque<Flow::DataPtr<Flow::Timestamp>> const& data) const;

    // Helpers to convert ONNX values back to Flow data and add them to the output cache
    void appendToOutput(Onnx::Value const& value);
    template<typename T>
    void appendVectorsToOutput(Onnx::Value const& value);
};

// inline implementations

inline std::string OnnxForwardNode::filterName() {
    return {"onnx-forward"};
}

}  // namespace Onnx

#endif  // _ONNX_FORWARD_NODE_HH
