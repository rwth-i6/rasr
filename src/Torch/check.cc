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

#include <Core/Application.hh>
#include <Core/Configuration.hh>

#include <Nn/LabelScorer/DataView.hh>

#include "TorchEncoder.hh"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

const Core::ParameterInt paramNumFrames(
        "num-frames",
        "number of dummy input frames",
        42);

const Core::ParameterInt paramFeatureDimension(
        "feature-dimension",
        "feature dimension of the dummy input",
        80);

const Core::ParameterInt paramMaxValuesToPrint(
        "max-values-to-print",
        "maximum number of output values to print",
        50);

class TestApplication : public Core::Application {
public:
    std::string getUsage() const override {
        return "test Torch encoder via RASR interface using dummy features\n";
    }

    int main(const std::vector<std::string>& arguments) {
        try {
            std::cout << "Load encoder..." << std::endl;
            Nn::ModelCache      modelCache;
            Torch::TorchEncoder encoder(config, modelCache);
            std::cout << "Encoder loaded." << std::endl;

            size_t num_frames       = static_cast<size_t>(paramNumFrames(config));
            size_t num_feature      = static_cast<size_t>(paramFeatureDimension(config));
            size_t maxValuesToPrint = static_cast<size_t>(paramMaxValuesToPrint(config));

            if (num_frames <= 0) {
                std::cerr << "num-frames must be > 0" << std::endl;
            }
            if (num_feature <= 0) {
                std::cerr << "feature-dimension must be > 0" << std::endl;
            }

            std::cout << "Create dummy input:\n"
                      << "  time-frames    = " << num_frames << "\n"
                      << "  feature-dim    = " << num_feature << std::endl;

            std::vector<float> features(num_frames * num_feature);
            for (size_t i = 0; i < features.size(); ++i) {
                features[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
            }
            auto featureStorage = std::shared_ptr<float[]>(new float[num_frames * num_feature]);
            std::copy(features.begin(), features.end(), featureStorage.get());

            std::cout << "Feed encoder..." << std::endl;
            for (size_t t = 0; t < num_frames; ++t) {
                Nn::DataView frame(featureStorage, num_feature, t * num_feature);
                encoder.addInput(frame);
            }

            encoder.signalNoMoreFeatures();

            std::cout << "Fetch outputs..." << std::endl;

            size_t numSpans         = 0;
            size_t numValuesPrinted = 0;

            while (true) {
                auto out = encoder.getNextOutput();
                if (!out) {
                    break;
                }
                ++numSpans;
                std::cout << "Output span " << numSpans
                          << ": [" << out->input_start << ", " << out->input_end << ")"
                          << " size=" << out->encoding.size() << std::endl;

                for (size_t i = 0; i < out->encoding.size() && numValuesPrinted < maxValuesToPrint; ++i) {
                    std::cout << "  outputs[" << numValuesPrinted << "] = " << out->encoding[i] << std::endl;
                    ++numValuesPrinted;
                }
            }
            std::cout << "Done.\n";
        }
        catch (std::exception const& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }

        return 0;
    }
};

APPLICATION(TestApplication)