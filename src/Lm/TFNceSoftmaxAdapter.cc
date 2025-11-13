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
#include "TFNceSoftmaxAdapter.hh"

namespace Lm {

void TFNceSoftmaxAdapter::init(Tensorflow::Session& session, Tensorflow::TensorInputMap const& input_map, Tensorflow::TensorOutputMap const& output_map) {
    session_    = &session;
    input_map_  = &input_map;
    output_map_ = &output_map;
}

Score TFNceSoftmaxAdapter::get_score(Lm::CompressedVectorPtr<float> const& nn_out, size_t output_idx) {
    std::vector<size_t> output_idxs(1, output_idx);
    return get_scores(nn_out, output_idxs)[0];
}

std::vector<Score> TFNceSoftmaxAdapter::get_scores(Lm::CompressedVectorPtr<float> const& nn_out, std::vector<size_t> const& output_idxs) {
    auto const& output_idx_tensor_info = input_map_->get_info("output_idxs");
    auto const& nn_output_tensor_info  = input_map_->get_info("nn_output");
    auto const& softmax_tensor_info    = output_map_->get_info("nce_softmax");

    // output indices
    std::vector<int> output_idxs_vec(output_idxs.size());
    std::copy(output_idxs.begin(), output_idxs.end(), output_idxs_vec.begin());
    Tensorflow::Tensor output_idxs_tensor = Tensorflow::Tensor::create(output_idxs_vec);

    // pre-softmax nn output
    std::vector<Math::FastMatrix<float>> nn_output;
    nn_output.emplace_back(nn_out->size(), 1);
    nn_out->uncompress(&nn_output[0](0, 0), nn_output[0].nRows());
    Tensorflow::Tensor nn_output_tensor = Tensorflow::Tensor::create(nn_output, true);

    std::vector<std::pair<std::string, Tensorflow::Tensor>> inputs;
    inputs.emplace_back(output_idx_tensor_info.tensor_name(), output_idxs_tensor);
    inputs.emplace_back(nn_output_tensor_info.tensor_name(), nn_output_tensor);
    std::vector<Tensorflow::Tensor> outputs;
    session_->run(inputs, {softmax_tensor_info.tensor_name()}, {}, outputs);

    std::vector<Score> scores;
    outputs[0].get(0, 0, scores);
    return scores;
}

}  // namespace Lm
