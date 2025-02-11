/** Copyright 2025 RWTH Aachen University. All rights reserved.
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

#ifndef ONNX_ENCODER_HH
#define ONNX_ENCODER_HH

#include <Nn/LabelScorer/Encoder.hh>

#include <Onnx/Model.hh>

namespace Onnx {

// Encoder that runs the input features through an ONNX model
class OnnxEncoder : public virtual Nn::Encoder {
    typedef Nn::Encoder Precursor;

public:
    OnnxEncoder(Core::Configuration const& config);

protected:
    virtual void encode() override;

private:
    Model onnxModel_;

    std::string featuresName_;
    std::string featuresSizeName_;
    std::string outputName_;
};

}  // namespace Onnx

#endif  // ONNX_ENCODER_HH
