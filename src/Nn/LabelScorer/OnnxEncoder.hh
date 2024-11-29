
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

#ifndef ONNX_ENCODER_HH
#define ONNX_ENCODER_HH

#include "Encoder.hh"

#include <Onnx/Model.hh>

namespace Nn {

// Encoder that runs the input features through an ONNX model
class OnnxEncoder : public Encoder {
    typedef Encoder Precursor;

public:
    OnnxEncoder(const Core::Configuration config);

protected:
    void encode() override;

private:
    Onnx::Model onnxModel_;

    std::string featuresName_;
    std::string featuresSizeName_;
    std::string outputName_;
};

}  // namespace Nn

#endif  // ONNX_ENCODER_HH
