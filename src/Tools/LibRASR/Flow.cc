/* Copyright 2025 RWTH Aachen University. All rights reserved.
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

#include "Flow.hh"

#include <Flow/Data.hh>
#include <Flow/InputNode.hh>
#include <Flow/Module.hh>
#include <Speech/Module.hh>

PYBIND11_DECLARE_HOLDER_TYPE(T, Core::Ref<T>, true);
PYBIND11_DECLARE_HOLDER_TYPE(T, Flow::DataPtr<T>, true);  // confirm

void bindFlow(py::module_& m) {
    py::class_<Flow::Data, Flow::DataPtr<Flow::Data>>(m, "Data")
            .def(py::init<>())
            .def(py::init<const Flow::Data&>())
            .def_static("eos", &Flow::Data::eos, py::return_value_policy::reference)
            .def("compare_address", [](Flow::Data& self, const Flow::Data* data) { return &self != data; })
            .def("__eq__", &Flow::Data::operator==, py::is_operator())
            .def("datatype", &Flow::Data::datatype, py::return_value_policy::reference_internal);

    py::class_<Flow::Datatype>(m, "Datatype")
            .def("name", &Flow::Datatype::name, py::return_value_policy::reference_internal);

    /*
     * virtual Data* clone() const
     * const Datatype* datatype() const
     * virtual Core::XmlWriter& dump(Core::XmlWriter&) const
     * virtual bool             read(Core::BinaryInputStream&)
     * virtual bool write(Core::BinaryOutputStream&) const
     * static inline Data* ood()
     * static bool isSentinel(const ThreadSafeReferenceCounted* object)
     * static bool isNotSentinel(const ThreadSafeReferenceCounted* object)
     */

    py::class_<Flow::Timestamp, Flow::Data, Flow::DataPtr<Flow::Timestamp>>(m, "Timestamp")
            .def(py::init<Flow::Time, Flow::Time>())
            .def("set_start_time", &Flow::Timestamp::setStartTime)
            .def("get_start_time", &Flow::Timestamp::getStartTime)
            .def("start_time", &Flow::Timestamp::startTime)
            .def("set_end_time", &Flow::Timestamp::setEndTime)
            .def("get_end_time", &Flow::Timestamp::getEndTime)
            .def("end_time", &Flow::Timestamp::endTime)
            .def("set_timestamp", &Flow::Timestamp::setTimestamp)
            .def("expand_timestamp", &Flow::Timestamp::expandTimestamp)
            .def("invalidate_timestamp", &Flow::Timestamp::invalidateTimestamp)
            .def("is_valid_timestamp", &Flow::Timestamp::isValidTimestamp)
            .def("equals_to_start_time", &Flow::Timestamp::equalsToStartTime)
            .def("equal_start_time", &Flow::Timestamp::equalStartTime)
            .def("equals_to_end_time", &Flow::Timestamp::equalsToEndTime)
            .def("equal_end_time", &Flow::Timestamp::equalEndTime)
            .def("contains", (bool(Flow::Timestamp::*)(Flow::Time) const) & Flow::Timestamp::contains)
            .def("contains", (bool(Flow::Timestamp::*)(const Flow::Timestamp&) const) & Flow::Timestamp::contains)
            .def("overlap", &Flow::Timestamp::overlap);

    //    static const Datatype* type()
    //    virtual Core::XmlWriter& dump(Core::XmlWriter& o) const
    //    bool read(Core::BinaryInputStream& i)
    //    bool write(Core::BinaryOutputStream& o) const

    py::class_<Mm::Feature, Core::Ref<Mm::Feature>>(m, "MmFeature")
            .def(py::init<size_t>())
            .def("add", (size_t(Mm::Feature::*)(const Core::TsRef<const Mm::Feature::Vector>&)) & Mm::Feature::add)
            .def("add", (void(Mm::Feature::*)(size_t, const Core::TsRef<const Mm::Feature::Vector>&)) & Mm::Feature::add)
            .def("set", (void(Mm::Feature::*)(size_t, const Core::TsRef<const Mm::Feature::Vector>&)) & Mm::Feature::set)
            .def("set", (void(Mm::Feature::*)(const std::vector<size_t>&, const Core::TsRef<const Mm::Feature::Vector>&)) & Mm::Feature::set)
            .def("clear", &Mm::Feature::clear)
            .def("main_stream", &Mm::Feature::mainStream, py::return_value_policy::take_ownership)
            .def("set_number_of_streams", &Mm::Feature::setNumberOfStreams)
            .def("num_streams", &Mm::Feature::nStreams);

    py::class_<Speech::Feature, Mm::Feature, Core::Ref<Speech::Feature>>(m, "Feature")
            .def(py::init<>())
            .def(py::init<Flow::DataPtr<Speech::Feature::FlowVector>&>())  // must be tested
            .def(py::init<Flow::DataPtr<Speech::Feature::FlowFeature>&>())
            .def("set_timestamp", &Speech::Feature::setTimestamp)
            .def("timestamp", &Speech::Feature::timestamp, py::return_value_policy::reference_internal)
            .def("take", (void(Speech::Feature::*)(Flow::DataPtr<Speech::Feature::FlowVector>&)) & Speech::Feature::take)
            .def("take", (void(Speech::Feature::*)(Flow::DataPtr<Speech::Feature::FlowFeature>&)) & Speech::Feature::take)
            .def("take", (bool(Speech::Feature::*)(Flow::DataPtr<Flow::Timestamp>&)) & Speech::Feature::take);

    // virtual Mm::FeatureDescription* getDescription(const Core::Configurable& parent) const

    py::class_<Flow::AbstractNode>(m, "AbstractNode")  // Core::Component
            .def_readonly_static("param_threaded", &Flow::AbstractNode::paramThreaded)
            .def_readonly_static("param_ignore_unknown_parameters", &Flow::AbstractNode::paramIgnoreUnknownParameters)  // Core::ParameterBool
            .def("run", &Flow::AbstractNode::Run)
            .def("set_threaded", (void(Flow::AbstractNode::*)(bool)) & Flow::AbstractNode::setThreaded)
            .def("set_threaded", (void(Flow::AbstractNode::*)(const std::string&)) & Flow::AbstractNode::setThreaded)
            .def("is_threaded", &Flow::AbstractNode::isThreaded)
            .def("add_parameter", &Flow::AbstractNode::addParameter)  //Core::StringExpression
            .def("set_parameter", &Flow::AbstractNode::setParameter)
            .def("erase_output_attributes", &Flow::AbstractNode::eraseOutputAttributes)
            .def("configure", &Flow::AbstractNode::configure)
            .def("work", &Flow::AbstractNode::work)
            .def("getRemaining_dataLen", &Flow::AbstractNode::getRemainingDataLen)  // o ssize_t, PortId
            .def("addUnresolved_parameter", &Flow::AbstractNode::addUnresolvedParameter)
            .def("unresolved_attributes", &Flow::AbstractNode::unresolvedAttributes, py::return_value_policy::reference_internal)
            .def("__lt__", &Flow::AbstractNode::operator<);

    // friend std::ostream& operator<<(std::ostream& o, const AbstractNode& n);
    // friend std::ostream& operator<<(std::ostream& o, const AbstractNode* n) {

    py::class_<Flow::Node, Flow::AbstractNode>(m, "Node")
            .def("get_input", &Flow::Node::getInput)
            .def("get_output", &Flow::Node::getOutput)
            .def("configure", &Flow::Node::configure)
            .def("work", &Flow::Node::work);

    py::class_<Flow::SourceNode, Flow::Node>(m, "SourceNode")
            .def("get_output", &Flow::SourceNode::getOutput);

    py::class_<Flow::InputNode, Flow::SourceNode>(m, "InputNode")
            .def(py::init<const Core::Configuration&>())
            .def_readonly_static("param_sample_rate", &Flow::InputNode::paramSampleRate)  // Core::Parameters
            .def_readonly_static("choice_sample_type", &Flow::InputNode::choiceSampleType)
            .def_readonly_static("param_sample_type", &Flow::InputNode::paramSampleType)
            .def_readonly_static("param_track_count", &Flow::InputNode::paramTrackCount)
            .def_readonly_static("param_block_size", &Flow::InputNode::paramBlockSize)
            .def_static("filter_name", &Flow::InputNode::filterName)
            .def("set_parameter", &Flow::InputNode::setParameter)
            .def("configure", &Flow::InputNode::configure)
            .def("work", &Flow::InputNode::work)
            .def("set_byte_stream_appender", &Flow::InputNode::setByteStreamAppender)
            .def("get_eos", &Flow::InputNode::getEOS)
            .def("set_eos", &Flow::InputNode::setEOS)
            .def("get_eos_received", &Flow::InputNode::getEOSReceived)
            .def("set_eos_received", &Flow::InputNode::setEOSReceived)
            .def("get_reset_sample_count", &Flow::InputNode::getResetSampleCount)
            .def("set_reset_sample_count", &Flow::InputNode::setResetSampleCount);

    // void setByteStreamAppender(ByteStreamAppender const& bsa);

    py::class_<Flow::Network, Flow::AbstractNode>(m, "Network")
            .def(py::init<const Core::Configuration&, bool>())
            .def("build_from_string", &Flow::Network::buildFromString)
            .def("build_from_file", &Flow::Network::buildFromFile)
            .def("set_type_name", &Flow::Network::setTypeName)
            .def("get_type_name", &Flow::Network::getTypeName, py::return_value_policy::reference_internal)
            .def("add_node", &Flow::Network::addNode)
            .def("get_node", &Flow::Network::getNode, py::return_value_policy::reference_internal)
            .def("add_link", &Flow::Network::addLink)
            .def("declare_parameter", &Flow::Network::declareParameter)
            .def("add_parameter_use", &Flow::Network::addParameterUse)  // Core::StringExpression
            .def("add_input", &Flow::Network::addInput)
            .def("get_input", &Flow::Network::getInput)
            .def("add_output", &Flow::Network::addOutput)
            .def("get_output", &Flow::Network::getOutput)
            .def("output_name", &Flow::Network::outputName, py::return_value_policy::reference_internal)
            .def("outputs", &Flow::Network::outputs)
            .def("activate_output", &Flow::Network::activateOutput)
            .def("put_data", &Flow::Network::putData)  // Data
            .def("put_eos", &Flow::Network::putEos)
            .def("put_ood", &Flow::Network::putOod)
            .def("get_port_link", &Flow::Network::getPortLink, py::return_value_policy::reference_internal)  // Link
            .def("get_data", (bool(Flow::Network::*)(Flow::PortId, Flow::DataPtr<Flow::Data>&)) & Flow::Network::getData)
            .def("put_attributes", &Flow::Network::putAttributes)  // Attributes
            .def("get_attribute", &Flow::Network::getAttribute)
            .def("set_parameter", &Flow::Network::setParameter)
            .def("configure", &Flow::Network::configure)
            .def("work", &Flow::Network::work)
            .def("get_remaining_data_len", &Flow::Network::getRemainingDataLen)
            .def("reset", &Flow::Network::reset)
            .def("go", &Flow::Network::go)
            .def("set_filename", &Flow::Network::setFilename)
            .def("filename", &Flow::Network::filename, py::return_value_policy::reference_internal)
            .def("configure_all", &Flow::Network::configureAll);

    // friend std::ostream& operator<<(std::ostream& o, const Network& n);

    py::class_<Flow::DataSource, Flow::Network>(m, "FlowDataSource", py::multiple_inheritance())
            .def(py::init<const Core::Configuration&, bool>())
            .def("get_data", (bool(Flow::DataSource::*)(Flow::PortId, Flow::DataPtr<Flow::Data>&)) & Flow::DataSource::getData);  // vector

    py::class_<Speech::DataSource, Flow::DataSource>(m, "DataSource")
            .def(py::init<const Core::Configuration&, bool>())
            .def("initialize", &Speech::DataSource::initialize)
            .def("finalize", &Speech::DataSource::finalize)
            .def("get_data", (bool(Speech::DataSource::*)(Flow::PortId, Core::Ref<Speech::Feature>&)) & Speech::DataSource::getData)
            .def("get_data", (bool(Speech::DataSource::*)(Core::Ref<Speech::Feature>&)) & Speech::DataSource::getData)
            .def("get_data", (bool(Speech::DataSource::*)()) & Speech::DataSource::getData)
            .def("convert", &Speech::DataSource::convert)
            .def("main_port_id", &Speech::DataSource::mainPortId)
            .def("nun_frames", &Speech::DataSource::nFrames, py::return_value_policy::reference_internal)
            .def("real_time", &Speech::DataSource::realTime)
            .def("set_progress_indication", &Speech::DataSource::setProgressIndication)
            .def("get_data", (bool(Speech::DataSource::*)(Flow::PortId, Flow::DataPtr<Flow::Data>&)) & Speech::DataSource::getData)
            .def("get_data", (bool(Speech::DataSource::*)(Flow::DataPtr<Flow::Data>&)) & Speech::DataSource::getData);  // be sure of template issue
}
