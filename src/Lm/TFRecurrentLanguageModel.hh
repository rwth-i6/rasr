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
#ifndef _LM_TF_RECURRENT_LANGUAGE_MODEL_HH
#define _LM_TF_RECURRENT_LANGUAGE_MODEL_HH

#include <Tensorflow/GraphLoader.hh>
#include <Tensorflow/Module.hh>
#include <Tensorflow/Session.hh>
#include <Tensorflow/TensorMap.hh>

#include "AbstractNNLanguageModel.hh"

namespace Lm {

class TFRecurrentLanguageModel : public AbstractNNLanguageModel {
public:
    typedef AbstractNNLanguageModel Precursor;
    typedef f32                     FeatureType;

    static Core::ParameterBool   paramTransformOuputLog;
    static Core::ParameterBool   paramTransformOuputNegate;
    static Core::ParameterInt    paramMaxBatchSize;
    static Core::ParameterBool   paramDumpScores;
    static Core::ParameterString paramDumpScoresPrefix;

    TFRecurrentLanguageModel(Core::Configuration const& c, Bliss::LexiconRef l);
    virtual ~TFRecurrentLanguageModel() = default;

    virtual History startHistory() const;
    virtual History extendedHistory(History const& hist, Token w) const;
    virtual Score score(History const& hist, Token w) const;

protected:
    virtual void load();

private:
    bool                        transform_output_log_;
    bool                        transform_output_negate_;
    std::function<Score(Score)> output_transform_function_;
    size_t                      max_batch_size_;
    bool                        dump_scores_;
    std::string                 dump_scores_prefix_;

    mutable Tensorflow::Session              session_;
    std::unique_ptr<Tensorflow::GraphLoader> loader_;
    std::unique_ptr<Tensorflow::Graph>       graph_;
    Tensorflow::TensorInputMap               tensor_input_map_;
    Tensorflow::TensorOutputMap              tensor_output_map_;

    std::vector<std::string> initializer_tensor_names_;
    std::vector<std::string> output_tensor_names_;
    std::vector<std::string> read_vars_tensor_names_;

    History empty_history_; // a history used to provide the previous (all zero) state to the first real history (1 sentence-begin token)
};

} // namespace Lm

#endif // _LM_TF_RECURRENT_LANGUAGE_MODEL_HH

