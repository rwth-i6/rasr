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

#include <deque>
#include <future>
#include <thread>

#include <Tensorflow/GraphLoader.hh>
#include <Tensorflow/Module.hh>
#include <Tensorflow/Session.hh>
#include <Tensorflow/TensorMap.hh>
#include <Core/readerwriterqueue.h>

#include "AbstractNNLanguageModel.hh"
#include "CompressedVector.hh"
#include "SearchSpaceAwareLanguageModel.hh"
#include "SoftmaxAdapter.hh"
#include "StateManager.hh"

namespace Lm {

class TFRecurrentLanguageModel : public AbstractNNLanguageModel, public SearchSpaceAwareLanguageModel {
public:
    struct TimeStatistics {
        std::chrono::duration<double, std::milli> total_duration;
        std::chrono::duration<double, std::milli> early_request_duration;
        std::chrono::duration<double, std::milli> request_duration;
        std::chrono::duration<double, std::milli> prepare_duration;
        std::chrono::duration<double, std::milli> merge_state_duration;
        std::chrono::duration<double, std::milli> set_state_duration;
        std::chrono::duration<double, std::milli> run_nn_output_duration;
        std::chrono::duration<double, std::milli> set_nn_output_duration;
        std::chrono::duration<double, std::milli> get_new_state_duration;
        std::chrono::duration<double, std::milli> split_state_duration;
        std::chrono::duration<double, std::milli> softmax_output_duration;

        TimeStatistics  operator+(TimeStatistics const& other) const;
        TimeStatistics& operator+=(TimeStatistics const& other);

        void write(Core::XmlChannel& channel) const;
        void write(std::ostream& out) const;
    };

    typedef AbstractNNLanguageModel                               Precursor;
    typedef moodycamel::BlockingReaderWriterQueue<History const*> HistoryQueue;

    static const Core::ParameterBool   paramTransformOuputLog;
    static const Core::ParameterBool   paramTransformOuputNegate;
    static const Core::ParameterInt    paramMinBatchSize;
    static const Core::ParameterInt    paramOptBatchSize;
    static const Core::ParameterInt    paramMaxBatchSize;
    static const Core::ParameterInt    paramHistoryPruningThreshold;
    static const Core::ParameterInt    paramPrunedHistoryLength;
    static const Core::ParameterFloat  paramBatchPruningThreshold;
    static const Core::ParameterBool   paramAllowReducedHistory;
    static const Core::ParameterBool   paramDumpScores;
    static const Core::ParameterString paramDumpScoresPrefix;
    static const Core::ParameterBool   paramLogMemory;
    static const Core::ParameterBool   paramFreeMemory;
    static const Core::ParameterInt    paramFreeMemoryDelay;
    static const Core::ParameterBool   paramAsync;
    static const Core::ParameterBool   paramSingleStepOnly;
    static const Core::ParameterBool   paramVerbose;

    TFRecurrentLanguageModel(Core::Configuration const& c, Bliss::LexiconRef l);
    virtual ~TFRecurrentLanguageModel();

    virtual History startHistory() const;
    virtual History extendedHistory(History const& hist, Token w) const;
    virtual History extendedHistory(History const& hist, Bliss::Token::Id w) const;
    virtual History reducedHistory(History const& hist, u32 limit) const;
    virtual Score   score(History const& hist, Token w) const;
    virtual bool    scoreCached(History const& hist, Token w) const;

    virtual void startFrame(Search::TimeframeIndex time) const;
    virtual void setInfo(History const& hist, SearchSpaceInformation const& info) const;

protected:
    virtual void load();

private:
    bool                        transform_output_log_;
    bool                        transform_output_negate_;
    std::function<Score(Score)> output_transform_function_;
    size_t                      min_batch_size_;
    size_t                      opt_batch_size_;
    size_t                      max_batch_size_;
    size_t                      history_pruning_threshold_;
    size_t                      pruned_history_length_;
    Score                       batch_pruning_threshold_;
    bool                        allow_reduced_history_;
    bool                        dump_scores_;
    std::string                 dump_scores_prefix_;
    bool                        log_memory_;
    bool                        free_memory_;
    Search::TimeframeIndex      free_memory_delay_;
    bool                        async_;
    bool                        single_step_only_;
    bool                        verbose_;

    mutable Tensorflow::Session              session_;
    std::unique_ptr<Tensorflow::GraphLoader> loader_;
    std::unique_ptr<Tensorflow::Graph>       graph_;
    Tensorflow::TensorInputMap               tensor_input_map_;
    Tensorflow::TensorOutputMap              tensor_output_map_;

    CompressedVectorFactoryPtr<float> state_comp_vec_factory_;
    CompressedVectorFactoryPtr<float> nn_output_comp_vec_factory_;

    std::vector<Tensorflow::Variable> state_variables_;
    std::unique_ptr<StateManager>     state_manager_;
    std::unique_ptr<SoftmaxAdapter>   softmax_adapter_;

    std::vector<std::string> initializer_tensor_names_;
    std::vector<std::string> output_tensor_names_;
    std::vector<std::string> read_vars_tensor_names_;

    History empty_history_;  // a history used to provide the previous (all zero) state to the first real history (1 sentence-begin token)

    mutable Core::XmlChannel       statistics_;
    mutable Search::TimeframeIndex current_time_;
    mutable std::vector<double>    run_time_;
    mutable std::vector<size_t>    run_count_;
    mutable double                 total_wait_time_;
    mutable double                 total_start_frame_time_;
    mutable double                 total_expand_hist_time_;
    mutable TimeStatistics         fwd_statistics_;

    // members for async forwarding
    std::thread background_forwarder_thread_;
    bool        should_stop_;

    mutable std::atomic<History const*>  to_fwd_;
    mutable std::promise<History const*> to_fwd_finished_;

    mutable std::vector<History const*> pending_;
    mutable HistoryQueue                fwd_queue_;
    mutable HistoryQueue                finished_queue_;

    History extendHistoryWithOutputIdx(History const& hist, size_t w) const;

    void background_forward() const;
    template<bool async>
    void forward(Lm::History const* hist) const;
};

}  // namespace Lm

#endif  // _LM_TF_RECURRENT_LANGUAGE_MODEL_HH
