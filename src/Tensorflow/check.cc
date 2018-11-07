/** Copyright 2018 RWTH Aachen University. All rights reserved.
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

#include "Session.hh"
#include "TensorMap.hh"
#include "VanillaGraphLoader.hh"

const Core::ParameterInt paramNumFrames("num-frames", "number of timeframes", 1);
const Core::ParameterInt paramNumFeatures("num-features", "number of features", 1);

class TestApplication : public Core::Application {
public:
    std::string getUsage() const {
        return "short program to test tensorflow integration\n";
    }

    int main(const std::vector<std::string>& arguments) {
        Tensorflow::Session            session(select("session"));
        Tensorflow::VanillaGraphLoader loader(select("loader"));

        std::unique_ptr<Tensorflow::Graph> graph = loader.load_graph();
        session.addGraph(*graph);
        loader.initialize(session);

        std::vector<Tensorflow::Tensor> output;

        std::vector<Math::FastMatrix<f32>> batch;
        size_t                             num_frames   = paramNumFrames(config);
        size_t                             num_features = paramNumFeatures(config);
        batch.emplace_back(num_frames, num_features);

        std::vector<std::pair<std::string, Tensorflow::Tensor>> inputs;
        Tensorflow::Tensor                                      data;
        data.set(batch);
        std::cerr << "data size: " << data.dimInfo() << std::endl;

        Tensorflow::TensorInputMap input_map(select("input-map"));
        require(input_map.has_info("features"));
        Tensorflow::TensorInputInfo const& feature_info = input_map.get_info("features");
        inputs.push_back(std::make_pair(feature_info.tensor_name(), data));

        Tensorflow::TensorOutputMap output_map(select("output-map"));
        require(output_map.has_info("classes"));
        Tensorflow::TensorOutputInfo const& classes_info = output_map.get_info("classes");

        session.run(inputs, {classes_info.tensor_name()}, {}, output);
        std::cerr << "output size: " << output[0].dimInfo() << " " << output[0].dataTypeName() << std::endl;

        std::vector<Math::FastMatrix<f32>> outputs;
        output[0].get(outputs, false);
        std::cerr << "mat size: " << outputs.size() << " " << outputs[0].nRows() << " " << outputs[0].nColumns() << std::endl;

        return 0;
    }
};

APPLICATION(TestApplication)
