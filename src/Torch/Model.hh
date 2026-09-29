/** Copyright 2026 RWTH Aachen University. All rights reserved.
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

#ifndef TORCH_MODEL_HH
#define TORCH_MODEL_HH

#include <Core/Component.hh>
#include "IOSpecParser.hh"
#include "IOSpecification.hh"
#include "Session.hh"

#include <string>
#include <vector>

#pragma push_macro("ensure")
#undef ensure
#include <torch/torch.h>
#pragma pop_macro("ensure")

namespace Torch {

/*
 * Wraps a Torch model session together with its expected runtime I/O specification.
 * The model owns the low-level Session and optionally validates the exported model once at
 * construction time using either the configured positional I/O specification or a JSON-based I/O specification.
 */
class Model : public Core::Component {
public:
    enum class IoSpecMode {
        ConfigSpec,
        JsonSpec,
    };

public:
    Model(Core::Configuration const& config);

    // Assemble runtime input vector according to configured positional mapping
    std::vector<torch::Tensor> makeInputs(torch::Tensor const& features, torch::Tensor const& lengths) const;

    // Access known outputs according to configured positional mapping
    torch::Tensor const& outputsFrom(std::vector<torch::Tensor> const& outputs) const;

    Session& session() {
        return session_;
    }
    Session const& session() const {
        return session_;
    }

    bool hasJsonIoSpec() const {
        return ioSpecMode_ == IoSpecMode::JsonSpec;
    }
    IoSpecMode ioSpecMode() const {
        return ioSpecMode_;
    }
    std::string const& ioSpecJsonPath() const {
        return ioSpecJsonPath_;
    }
    ParsedIoSpec const& parsedIoSpec() const {
        return ioSpec_;
    }
    bool hasParsedIoSpec() const {
        return ioSpec_.valid;
    }

    size_t featuresPosition() const {
        return featuresPosition_;
    }
    size_t featuresLengthsPosition() const {
        return featuresLengthsPosition_;
    }
    size_t outputsPosition() const {
        return outputsPosition_;
    }

private:
    void initializeIoSpecMode(Core::Configuration const& config);
    void parseIoSpecJson();

    Session      session_;
    ParsedIoSpec ioSpec_;
    IoSpecMode   ioSpecMode_ = IoSpecMode::ConfigSpec;
    std::string  ioSpecJsonPath_;

    size_t featuresPosition_;
    size_t featuresLengthsPosition_;
    size_t outputsPosition_;
};

}  // namespace Torch

#endif  // TORCH_MODEL_HH
