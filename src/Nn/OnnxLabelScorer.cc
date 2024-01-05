/* Copyright 2020 RWTH Aachen University. All rights reserved.
 *
 * Licensed under the RWTH ASR License (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 * 	http://www.hltpr.rwth-aachen.de/rwth-asr/rwth-asr-license.html
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "OnnxLabelScorer.hh"
// #include "Prior.hh"

using namespace Nn;

const Core::ParameterBool OnnxModelBase::paramTransformOuputLog(
        "transform-output-log",
        "apply log to tensorflow output",
        false);

const Core::ParameterBool OnnxModelBase::paramTransformOuputNegate(
        "transform-output-negate",
        "negate tensorflow output (after log)",
        false);

const Core::ParameterInt OnnxModelBase::paramMaxBatchSize(
        "max-batch-size",
        "maximum number of histories forwarded in one go",
        64, 1);

OnnxModelBase::OnnxModelBase(const Core::Configuration& config) :
        Core::Component(config),
        Precursor(config),
        session_(select("session")),
        mapping_(select("io-map"), ioSpec_),
        validator_(select("validator")),
        features_onnx_name_(mapping_.getOnnxName("features")),
        features_size_onnx_name_(mapping_.getOnnxName("features-size")),
        output_onnx_names_({mapping_.getOnnxName("output")}),
        maxBatchSize_(paramMaxBatchSize(config)) {

  bool valid = validator_.validate(ioSpec_, mapping_, session_);
  if (not valid) {
    warning("Failed to validate input model.");
  }

  bool transform_output_log = paramTransformOuputLog(config);
  bool transform_output_negate = paramTransformOuputNegate(config);
  if (transform_output_log && transform_output_negate) {
    decoding_output_transform_function_ = [](Score v, Score scale){ return -scale * std::log(v); };
    log() << "apply -log(.) to model output";
  } else if (transform_output_log) {
    decoding_output_transform_function_ = [](Score v, Score scale){ return scale * std::log(v); };
    log() << "apply log(.) to model output";
  } else if (transform_output_negate) {
    decoding_output_transform_function_ = [](Score v, Score scale){ return -scale * v; };
    log() << "apply -(.) to model output";
  } else if (scale_ != 1.0) {
    decoding_output_transform_function_ = [](Score v, Score scale){ return scale * v; };
  }

  init();
  reset();

  // debug
  Core::ParameterBool paramDebug("debug", "", false);
  debug_ = paramDebug(config);
}


OnnxModelBase::~OnnxModelBase() {
    reset();
    //delete startHistoryDescriptor_;
}

const std::vector<Onnx::IOSpecification> OnnxModelBase::ioSpec_ = {
        Onnx::IOSpecification{
                "features",
                Onnx::IODirection::INPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -1, -2}, {1, -1, -2}}},
        Onnx::IOSpecification{
                "features-size",
                Onnx::IODirection::INPUT,
                true,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::INT32},
                {{-1}}},
        Onnx::IOSpecification{
                "output",
                Onnx::IODirection::OUTPUT,
                false,
                {Onnx::ValueType::TENSOR},
                {Onnx::ValueDataType::FLOAT},
                {{-1, -1, -2}, {1, -1, -2}}}};

void OnnxModelBase::reset() {
    Precursor::reset();
    //batch_.clear();
    cacheHashQueue_.clear();
}

void OnnxModelBase::init() {}

OnnxFfnnTransducer::OnnxFfnnTransducer(const Core::Configuration& config) :
	Core::Component(config),
	Precursor(config) {}

LabelHistory OnnxFfnnTransducer::startHistory() {

	return labelHistoryManager_->history(new LabelHistoryDescriptor());
}

const std::vector<Score>& OnnxFfnnTransducer::getScores(const LabelHistory& h, bool isLoop) {

	return inputBuffer_.at(decodeStep_);

}

/*void OnnxFfnnTransducer::cleanUpBeforeExtension(u32 minPos) {
	scoreCache_.clear();
	batchHashQueue_.clear();
	scoreTransitionCache_.clear();

	if (isPositionDependent_) {
		// cache clean up w.r.t min position among all hypotheses (otherwise memory expensive ?)
		for (std::pair<const u32, ScoreCache>& kv : positionScoreCache_)
			if (kv.first < minPos)
				kv.second.clear();
        }
}*/
