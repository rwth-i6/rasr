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

#include "Corpus.hh"

#include <Bliss/CorpusDescription.hh>
#include <Bliss/CorpusParser.hh>
#include <Speech/Module.hh>

PYBIND11_DECLARE_HOLDER_TYPE(T, Core::Ref<T>, true);

class PublicCorpusVisitor : public Speech::CorpusVisitor {
public:
    using Speech::CorpusVisitor::CorpusVisitor;
    using Speech::CorpusVisitor::enterCorpus;
    using Speech::CorpusVisitor::enterRecording;
    using Speech::CorpusVisitor::leaveCorpus;
    using Speech::CorpusVisitor::leaveRecording;
    using Speech::CorpusVisitor::visitSegment;
    using Speech::CorpusVisitor::visitSpeechSegment;
};

class PyCorpusVisitor : public Speech::CorpusVisitor {
public:
    using Speech::CorpusVisitor::CorpusVisitor;
    void enterCorpus(Bliss::Corpus* corpus) override {
        PYBIND11_OVERRIDE(void, Speech::CorpusVisitor, enterCorpus, corpus);
    }
    void leaveCorpus(Bliss::Corpus* corpus) override {
        PYBIND11_OVERRIDE(void, Speech::CorpusVisitor, leaveCorpus, corpus);
    }
    void enterRecording(Bliss::Recording* recording) override {
        PYBIND11_OVERRIDE(void, Speech::CorpusVisitor, enterRecording, recording);
    }
    void leaveRecording(Bliss::Recording* recording) override {
        PYBIND11_OVERRIDE(void, Speech::CorpusVisitor, leaveRecording, recording);
    }
    void visitSegment(Bliss::Segment* segment) override {
        PYBIND11_OVERRIDE(void, Speech::CorpusVisitor, visitSegment, segment);
    }
    void visitSpeechSegment(Bliss::SpeechSegment* segment) override {
        PYBIND11_OVERRIDE(void, Speech::CorpusVisitor, visitSpeechSegment, segment);
    }
};

void bindCorpus(py::module_& m) {
    py::class_<Bliss::NamedCorpusEntity>(m, "NamedCorpusEntity")
            .def("parent", &Bliss::NamedCorpusEntity::parent, py::return_value_policy::reference_internal)
            .def("set_parent", &Bliss::NamedCorpusEntity::setParent)
            .def("name", &Bliss::NamedCorpusEntity::name, py::return_value_policy::reference_internal)
            .def("full_name", &Bliss::NamedCorpusEntity::fullName)
            .def("set_name", &Bliss::NamedCorpusEntity::setName)
            .def("set_remove_prefix", &Bliss::NamedCorpusEntity::setRemovePrefix)
            .def("is_anonymous", &Bliss::NamedCorpusEntity::isAnonymous);

    py::class_<Bliss::Speaker, Bliss::NamedCorpusEntity> speaker(m, "Speaker");
    speaker
            .def(py::init<Bliss::ParentEntity*>())
            .def("gender", &Bliss::Speaker::gender)
            .def("set_parent", &Bliss::Speaker::setParent);

    py::enum_<Bliss::Speaker::Gender>(speaker, "Gender")
            .value("unknown", Bliss::Speaker::Gender::unknown)
            .value("male", Bliss::Speaker::Gender::male)
            .value("female", Bliss::Speaker::Gender::female)
            .value("num_genders", Bliss::Speaker::Gender::nGenders)
            .export_values();

    py::class_<Bliss::AcousticCondition, Bliss::NamedCorpusEntity>(m, "AcousticCondition")
            .def(py::init<Bliss::ParentEntity*>())
            .def("set_parent", &Bliss::AcousticCondition::setParent);

    py::class_<Bliss::ParentEntity, Bliss::NamedCorpusEntity>(m, "ParentEntity")
            .def("is_name_reserved", &Bliss::ParentEntity::isNameReserved)
            .def("reserve_name", &Bliss::ParentEntity::reserveName);

    py::class_<Bliss::Directory<Bliss::Speaker>>(m, "DirectorySpeaker")
            .def(py::init<>())
            .def(py::init<const Bliss::Directory<Bliss::Speaker>&>())
            .def("has_key", &Bliss::Directory<Bliss::Speaker>::hasKey)
            .def("add", &Bliss::Directory<Bliss::Speaker>::add)
            .def("lookup", &Bliss::Directory<Bliss::Speaker>::lookup, py::return_value_policy::reference_internal);

    py::class_<Bliss::Directory<Bliss::AcousticCondition>>(m, "DirectoryAcousticCondition")
            .def(py::init<>())
            .def(py::init<const Bliss::Directory<Bliss::AcousticCondition>&>())
            .def("has_key", &Bliss::Directory<Bliss::AcousticCondition>::hasKey)
            .def("add", &Bliss::Directory<Bliss::AcousticCondition>::add)
            .def("lookup", &Bliss::Directory<Bliss::AcousticCondition>::lookup, py::return_value_policy::reference_internal);

    py::class_<Bliss::CorpusSection, Bliss::ParentEntity>(m, "CorpusSection")
            .def(py::init<Bliss::CorpusSection*>())
            .def("parent", &Bliss::CorpusSection::parent, py::return_value_policy::reference_internal)
            .def("level", &Bliss::CorpusSection::level)
            .def("speaker", &Bliss::CorpusSection::speaker)
            .def("default_speaker", &Bliss::CorpusSection::defaultSpeaker)
            .def("condition", &Bliss::CorpusSection::condition)
            .def("default_condition", &Bliss::CorpusSection::defaultCondition);

    py::class_<Bliss::Corpus, Bliss::CorpusSection>(m, "Corpus")
            .def(py::init<Bliss::Corpus*>());

    py::class_<Bliss::Recording, Bliss::CorpusSection>(m, "Recording")
            .def(py::init<Bliss::Corpus*>())
            .def("audio", &Bliss::Recording::audio, py::return_value_policy::reference_internal)
            .def("set_audio", &Bliss::Recording::setAudio)
            .def("video", &Bliss::Recording::video, py::return_value_policy::reference_internal)
            .def("set_video", &Bliss::Recording::setVideo)
            .def("duration", &Bliss::Recording::duration)
            .def("set_duration", &Bliss::Recording::setDuration);

    py::class_<Bliss::Segment, Bliss::ParentEntity> segment(m, "Segment");
    segment
            .def(py::init<Bliss::Segment::Type, Bliss::Recording*>())
            .def("recording", &Bliss::Segment::recording, py::return_value_policy::reference_internal)
            .def("set_recording", &Bliss::Segment::setRecording)
            .def("parent", &Bliss::Segment::parent, py::return_value_policy::reference_internal)
            .def("type", &Bliss::Segment::type)
            .def("set_type", &Bliss::Segment::setType)
            .def("start", &Bliss::Segment::start)
            .def("set_start", &Bliss::Segment::setStart)
            .def("end", &Bliss::Segment::end)
            .def("set_end", &Bliss::Segment::setEnd)
            .def("track", &Bliss::Segment::track)
            .def("set_track", &Bliss::Segment::setTrack)
            .def("condition", &Bliss::Segment::condition, py::return_value_policy::reference_internal)
            .def("set_condition", &Bliss::Segment::setCondition)
            .def("accept", &Bliss::Segment::accept);

    py::enum_<Bliss::Segment::Type>(segment, "Type")
            .value("type_speech", Bliss::Segment::Type::typeSpeech)
            .value("type_other", Bliss::Segment::Type::typeOther)
            .value("num_types", Bliss::Segment::Type::nTypes)
            .export_values();

    py::class_<Bliss::SpeechSegment, Bliss::Segment>(m, "SpeechSegment")
            .def(py::init<Bliss::Recording*>())
            .def("orth", &Bliss::SpeechSegment::orth, py::return_value_policy::reference_internal)
            .def("set_orth", &Bliss::SpeechSegment::setOrth)
            .def("left_context_orth", &Bliss::SpeechSegment::leftContextOrth, py::return_value_policy::reference_internal)
            .def("set_left_context_orth", &Bliss::SpeechSegment::setLeftContextOrth)
            .def("right_context_orth", &Bliss::SpeechSegment::rightContextOrth, py::return_value_policy::reference_internal)
            .def("set_right_context_orth", &Bliss::SpeechSegment::setRightContextOrth)
            .def("speaker", &Bliss::SpeechSegment::speaker, py::return_value_policy::reference_internal)
            .def("set_speaker", &Bliss::SpeechSegment::setSpeaker)
            .def("accept", &Bliss::SpeechSegment::accept);

    py::class_<Bliss::SegmentVisitor>(m, "SegmentVisitor")
            .def("visit_segment", &Bliss::SegmentVisitor::visitSegment)
            .def("visit_speech_segment", &Bliss::SegmentVisitor::visitSpeechSegment);

    py::class_<Bliss::CorpusVisitor, Bliss::SegmentVisitor>(m, "BlissCorpusVisitor")
            .def("enter_recording", &Bliss::CorpusVisitor::enterRecording)
            .def("leave_recording", &Bliss::CorpusVisitor::leaveRecording)
            .def("enter_corpus", &Bliss::CorpusVisitor::enterCorpus)
            .def("leave_corpus", &Bliss::CorpusVisitor::leaveCorpus);

    py::class_<Core::ParameterString>(m, "ParameterString");

    py::class_<Core::ParameterBool>(m, "ParameterBool");

    py::class_<Core::ParameterInt>(m, "ParameterInt");

    py::class_<Core::ParameterStringVector>(m, "ParameterStringVector");

    py::class_<Bliss::CorpusDescription>(m, "CorpusDescription")
            .def(py::init<const Core::Configuration&>())
            .def("file", &Bliss::CorpusDescription::file, py::return_value_policy::reference_internal)
            .def("accept", &Bliss::CorpusDescription::accept)
            .def("total_segment_count", &Bliss::CorpusDescription::totalSegmentCount)
            .def_readonly_static("param_filename", &Bliss::CorpusDescription::paramFilename)
            .def_readonly_static("param_allow_empty_whitelist", &Bliss::CorpusDescription::paramAllowEmptyWhitelist)
            .def_readonly_static("param_encoding", &Bliss::CorpusDescription::paramEncoding)
            .def_readonly_static("param_partition", &Bliss::CorpusDescription::paramPartition)
            .def_readonly_static("param_partition_selection", &Bliss::CorpusDescription::paramPartitionSelection)
            .def_readonly_static("param_skip_first_segments", &Bliss::CorpusDescription::paramSkipFirstSegments)
            .def_readonly_static("param_segments_to_skip", &Bliss::CorpusDescription::paramSegmentsToSkip)
            .def_readonly_static("param_recording_based_partition", &Bliss::CorpusDescription::paramRecordingBasedPartition)
            .def_readonly_static("param_segment_order", &Bliss::CorpusDescription::paramSegmentOrder)
            .def_readonly_static("param_segment_order_lookup_name", &Bliss::CorpusDescription::paramSegmentOrderLookupName)
            .def_readonly_static("param_segment_order_shuffle", &Bliss::CorpusDescription::paramSegmentOrderShuffle)
            .def_readonly_static("param_segment_order_shuffle_seed", &Bliss::CorpusDescription::paramSegmentOrderShuffleSeed)
            .def_readonly_static("param_segment_order_sort_by_time_length", &Bliss::CorpusDescription::paramSegmentOrderSortByTimeLength)
            .def_readonly_static("param_segment_order_sort_by_time_length_chunk_size", &Bliss::CorpusDescription::paramSegmentOrderSortByTimeLengthChunkSize)
            .def_readonly_static("param_python_segment_order", &Bliss::CorpusDescription::paramPythonSegmentOrder)
            .def_readonly_static("param_python_segment_order_mod_path", &Bliss::CorpusDescription::paramPythonSegmentOrderModPath)
            .def_readonly_static("param_python_segment_order_mod_name", &Bliss::CorpusDescription::paramPythonSegmentOrderModName)
            .def_readonly_static("param_python_segment_order_config", &Bliss::CorpusDescription::paramPythonSegmentOrderConfig);

    py::class_<Bliss::ProgressReportingVisitorAdaptor, Bliss::CorpusVisitor>(m, "ProgressReportingVisitorAdaptor")
            .def("set_visitor", &Bliss::ProgressReportingVisitorAdaptor::setVisitor)
            .def("enter_corpus", &Bliss::ProgressReportingVisitorAdaptor::enterCorpus)
            .def("leave_corpus", &Bliss::ProgressReportingVisitorAdaptor::leaveCorpus)
            .def("enter_recording", &Bliss::ProgressReportingVisitorAdaptor::enterRecording)
            .def("leave_recording", &Bliss::ProgressReportingVisitorAdaptor::leaveRecording)
            .def("visit_segment", &Bliss::ProgressReportingVisitorAdaptor::visitSegment)
            .def("visit_speech_segment", &Bliss::ProgressReportingVisitorAdaptor::visitSpeechSegment);

    py::class_<Core::StringExpression>(m, "StringExpression")
            .def(py::init<>())
            .def("is_constant", &Core::StringExpression::isConstant)
            .def("has_variable", &Core::StringExpression::hasVariable)
            .def("value", &Core::StringExpression::value)
            .def("is_constant", &Core::StringExpression::isConstant)
            .def("set_variable", &Core::StringExpression::setVariable)
            .def("set_variables", &Core::StringExpression::setVariables)
            .def("clear", (bool(Core::StringExpression::*)(const std::string&)) & Core::StringExpression::clear)
            .def("clear", (void(Core::StringExpression::*)()) & Core::StringExpression::clear);

    py::class_<Bliss::CorpusDescriptionParser>(m, "CorpusDescriptionParser")
            .def(py::init<const Core::Configuration&>())
            .def("accept", &Bliss::CorpusDescriptionParser::accept)
            .def_readonly_static("param_audio_dir", &Bliss::CorpusDescriptionParser::paramAudioDir)
            .def_readonly_static("param_video_dir", &Bliss::CorpusDescriptionParser::paramVideoDir)
            .def_readonly_static("param_remove_corpus_name_prefix", &Bliss::CorpusDescriptionParser::paramRemoveCorpusNamePrefix)
            .def_readonly_static("param_captialize_transcriptions", &Bliss::CorpusDescriptionParser::paramCaptializeTranscriptions)
            .def_readonly_static("param_gemenize_transcriptions", &Bliss::CorpusDescriptionParser::paramGemenizeTranscriptions)
            .def_readonly_static("param_progress", &Bliss::CorpusDescriptionParser::paramProgress);

    py::class_<Speech::CorpusVisitor, Bliss::CorpusVisitor, PyCorpusVisitor>(m, "SpeechCorpusVisitor", py::multiple_inheritance())
            .def(py::init<const Core::Configuration&>())
            .def("enter_corpus", &PublicCorpusVisitor::enterCorpus)
            .def("leave_corpus", &PublicCorpusVisitor::leaveCorpus)
            .def("enter_recording", &PublicCorpusVisitor::enterRecording)
            .def("leave_recording", &PublicCorpusVisitor::leaveRecording)
            .def("visitSegment", &PublicCorpusVisitor::visitSegment)
            .def("visitSpeechSegment", &PublicCorpusVisitor::visitSpeechSegment)
            .def(
                    "sign_on",
                    py::overload_cast<Speech::CorpusProcessor*>(&Speech::CorpusVisitor::signOn),
                    py::keep_alive<1, 2>());

    py::class_<Speech::CorpusProcessor>(m, "CorpusProcessor")
            .def(py::init<const Core::Configuration&>())
            .def("sign_on", &Speech::CorpusProcessor::signOn)
            .def("enter_corpus", &Speech::CorpusProcessor::enterCorpus)
            .def("leave_corpus", &Speech::CorpusProcessor::leaveCorpus)
            .def("enter_recording", &Speech::CorpusProcessor::enterRecording)
            .def("leave_recording", &Speech::CorpusProcessor::leaveRecording)
            .def("enter_segment", &Speech::CorpusProcessor::enterSegment)
            .def("process_segment", &Speech::CorpusProcessor::processSegment)
            .def("leave_segment", &Speech::CorpusProcessor::leaveSegment)
            .def("enter_speech_segment", &Speech::CorpusProcessor::enterSpeechSegment)
            .def("process_speech_segment", &Speech::CorpusProcessor::processSpeechSegment)
            .def("leave_speech_segment", &Speech::CorpusProcessor::leaveSpeechSegment);
}
