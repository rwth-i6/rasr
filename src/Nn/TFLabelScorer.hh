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
 *
 *  author: Wei Zhou
 */

#ifndef TF_LABEL_SCORER_HH
#define TF_LABEL_SCORER_HH

#include <Tensorflow/GraphLoader.hh>
#include <Tensorflow/Module.hh>
#include <Tensorflow/Session.hh>
#include <Tensorflow/TensorMap.hh>
#include <Tensorflow/Tensor.hh>
#include "LabelScorer.hh"

namespace Nn {
 
typedef std::vector<Tensorflow::Tensor> TensorList;
typedef std::vector<std::pair<std::string, Tensorflow::Tensor>> MappedTensorList;

struct TFLabelHistory : public LabelHistoryBase {
  std::vector<Score> scores;
  TensorList variables;
  u32 position;
  bool isBlank; // for next feedback

  typedef LabelHistoryBase Precursor;

  TFLabelHistory() : Precursor(), position(0), isBlank(false) {}
  TFLabelHistory(const TFLabelHistory& ref) : 
      Precursor(ref), scores(ref.scores), variables(ref.variables), 
      position(ref.position), isBlank(ref.isBlank) {}
};


// Encoder-Decoder Label Scorer based on Tensorflow back-end
// computation logics based on a predefined order of I/O and op collections in graph
// prerequisite: model graph compilation that parse the model into these collections
class TFModelBase : public LabelScorer {
    typedef LabelScorer Precursor;

  public:
    // config params for graph computation
    static const Core::ParameterBool paramTransformOuputLog;
    static const Core::ParameterBool paramTransformOuputNegate;
    static const Core::ParameterInt paramMaxBatchSize;

    // overwrite descriptor in derived class for specific history
    typedef TFLabelHistory LabelHistoryDescriptor;

  public:
    TFModelBase(const Core::Configuration& config);
    virtual ~TFModelBase();

    virtual void reset();
    virtual void cleanUpBeforeExtension(u32 minPos) { cacheHashQueue_.clear(); }

    // history handling
    virtual LabelHistory startHistory();
    virtual void extendLabelHistory(LabelHistory& h, LabelIndex idx, u32 position, bool isLoop);

    // encoding
    virtual void encode();

    // get scores for the next output position
    virtual const std::vector<Score>& getScores(const LabelHistory& h, bool isLoop);

  protected:
    void init();
    void initDecoder();
    void initStartHistory();
    void loadPrior();

    // ---- batch-wise graph computation ----
    virtual void initComputation();

    virtual void makeBatch(LabelHistoryDescriptor* targetLhd);
    virtual void decodeBatch();

    virtual void feedBatchVariables();
    virtual void feedDecodeInput(MappedTensorList& inputs);
    virtual void updateBatchVariables(bool post=false);
    virtual void fetchBatchVariables();

    virtual void addPriorToBatch();
    virtual void computeBatchScores();
    virtual void processBatchOutput(const TensorList& outputs);
    // --------------------------------------

    bool debug_;
    void debugFetch(const std::vector<std::string>& fetchNames, std::string msg="");

  protected:
    // Note: graph related params follow snake_case naming style
    mutable Tensorflow::Session              session_;
    std::unique_ptr<Tensorflow::GraphLoader> loader_;
    std::unique_ptr<Tensorflow::Graph>       graph_;

    // --- encoder ---
    std::string encoding_input_tensor_name_;
    std::string encoding_input_seq_length_tensor_name_;

    // --- decoder ---
    std::vector<std::string> decoding_input_tensor_names_;
    std::vector<std::string> decoding_output_tensor_names_;
    std::vector<u32> decoding_input_ndims_;
    std::vector<u32> decoding_output_ndims_;
    // binary function including scaling 
    std::function<Score(Score, Score)> decoding_output_transform_function_;

    std::vector<std::string> var_feed_names_;
    std::vector<std::string> var_feed_ops_;
    std::vector<std::string> var_fetch_names_;

    // --- step ops ---
    std::vector<std::string> encoding_ops_;
    std::vector<std::string> decoding_ops_;
    std::vector<std::string> var_update_ops_;
    std::vector<std::string> var_post_update_ops_;

    // --- global ---
    std::vector<std::string> global_var_feed_names_;
    std::vector<std::string> global_var_feed_ops_;
    
  protected:
    LabelHistoryDescriptor* startHistoryDescriptor_; // only common stuff, no states or scores

    typedef std::vector<LabelHistoryDescriptor*> Batch;
    Batch batch_;
    std::deque<size_t> cacheHashQueue_;
    u32 maxBatchSize_;

    typedef std::unordered_map<size_t, std::vector<Score>> ScoreCache;
    ScoreCache contextLogPriors_;
};


// Attention-based Encoder-Decoder Model
// attention mechanism only in model graph (soft/hard): no additional latent variable here
class TFAttentionModel : public TFModelBase {
    typedef TFModelBase Precursor;

  public:
    TFAttentionModel(const Core::Configuration& config) :
        Core::Component(config), 
        Precursor(config) { needEndProcessing_ = true; }
};


// RNN-Transducer|Aligner
// - blank-based topology
//   - strictly monotonic (time|alignment-sync search w.r.t. decodeStep_)
//     - either global_var simplification for enc_position
//       or empty global_var: each hyp has its own position state_var (always +1)
//     - optional label loop: different score and history handling
//   - vertical transition (alignment-sync search)
//     - empty global_var: each hyp has its own position state_var (+1 for blank)
//     - additional ending detection/processing based on position
// - non-blank based topology: HMM-like with label loop
// - feedback: always the last alignment label (masking done in the graph)
// - dependency (recombination)
//   - default: output label sequence
//   - optional include blanks (e.g. towards full alignment sequence)
//   - optional include loops
class TFRnnTransducer : public TFModelBase {
    typedef TFModelBase Precursor;

  public:
    static const Core::ParameterBool paramLoopFeedbackAsBlank;
    static const Core::ParameterBool paramVerticalTransition;

  public:
    TFRnnTransducer(const Core::Configuration& config);

    bool useVerticalTransition() const { return verticalTransition_; }

    void increaseDecodeStep();
    void extendLabelHistory(LabelHistory& h, LabelIndex idx, u32 position, bool isLoop);

  protected:
    void feedDecodeInput(MappedTensorList& inputs);
    void setDecodePosition(u32 pos);

  private:
    LabelIndex blankLabelIndex_;
    bool loopFeedbackAsBlank_;
    bool verticalTransition_;
};


// no state vars or scores: just label sequence and context hash
struct NgramLabelHistory : public LabelHistoryBase {
  size_t forwardHash, loopHash;
  u32 position; // only for position-aware ffnn-transducer

  typedef LabelHistoryBase Precursor;

  NgramLabelHistory() : Precursor(), forwardHash(0), loopHash(0), position(0) {}
  NgramLabelHistory(const NgramLabelHistory& ref) : 
      Precursor(ref), forwardHash(ref.forwardHash), loopHash(ref.loopHash), position(ref.position) {}
  NgramLabelHistory(const LabelSequence& labSeq, LabelIndex nextIdx) :
      Precursor(), forwardHash(0), loopHash(0), position(0) {
    // always fixed context size (+1) and right-most latest 
    LabelSequence newSeq(labSeq.begin()+1, labSeq.end());
    newSeq.push_back(nextIdx);
    labelSeq.swap(newSeq);
  }
};

// FFNN transducer with ngram context (no recurrency in decoder)
// - strictly monotonic topology only + global_var simplification for enc_position
// - both time-synchronous and label-synchronous search possible
//   - latter: re-interpreted segmental decoding based on frame-wise output
// - label topology
//    - either HMM-topology: loop without blank 
//    - or RNA-topology: blank without loop
// - dependency
//   - output/segment label sequence or alignment sequence
//   - additional first-order relative-position (so far only for RNA topology)
// Note: speed-up with context embedding lookup should be configured in the model graph
class TFFfnnTransducer : public TFModelBase {
    typedef TFModelBase Precursor;
    typedef NgramLabelHistory LabelHistoryDescriptor;

  public:
    static const Core::ParameterInt paramContextSize;
    static const Core::ParameterBool paramCacheHistory;
    static const Core::ParameterBool paramImplicitTransition;
    static const Core::ParameterBool paramExplicitTransition;
    static const Core::ParameterBool paramUseRelativePosition;
    static const Core::ParameterBool paramRenormTransition;

  public:
    TFFfnnTransducer(Core::Configuration const& config);
    ~TFFfnnTransducer();

    void reset();
    void cleanUpBeforeExtension(u32 minPos);

    bool useRelativePosition() const { return useRelativePosition_; }

    // history handling
    LabelHistory startHistory();
    void extendLabelHistory(LabelHistory& h, LabelIndex idx, u32 position, bool isLoop);

    // global position of encodings
    void increaseDecodeStep();

    // get label scores for the next output position
    const std::vector<Score>& getScores(const LabelHistory& h, bool isLoop);

    // get segment scores for the next label segment given start position
    const SegmentScore& getSegmentScores(const LabelHistory& h, LabelIndex segIdx, u32 startPos);

  protected:
    void initComputation() {}
    void makeBatch(LabelHistoryDescriptor* targetLhd);
    void decodeBatch(ScoreCache& scoreCache);
    void computeBatchScores(ScoreCache& scoreCache);
    void setDecodePosition(u32 pos);

    const std::vector<Score>& getScoresWithTransition(const LabelHistory& h, bool isLoop);
    Score getExclusiveScore(Score score);

    // for segmental decoding
    const std::vector<Score>& getPositionScores(size_t hash, u32 pos);
    void makePositionBatch(size_t hash, const ScoreCache& scoreCache);

  private:
    u32 contextSize_;
    bool cacheHistory_;

    // context (and position) dependent cache: central handling of scores instead of each history
    ScoreCache scoreCache_;
    std::unordered_set<size_t> batchHashQueue_;
    std::vector<size_t> batchHash_;

    // HMM topology differs w.r.t. loopUpdateHistory_, if true then
    // - alignment sequence dependency (otherwise output/segment label sequence)
    // - loop scoring based on previous frame labels (otherwise segment labels)
    bool hmmTopology_;
    typedef std::unordered_map<size_t, LabelSequence> LabelSeqCache;
    LabelSeqCache labelSeqCache_; // only for HMM topology: need clean up if not cacheHistory_ ?
    ScoreCache scoreTransitionCache_;
    bool implicitTransition_;
    bool explicitTransition_;
    bool renormTransition_;

    LabelIndex blankLabelIndex_;
    bool useRelativePosition_;

    // for segmental decoding {position: {context: scores}}
    std::unordered_map<u32, ScoreCache> positionScoreCache_;
};


// TODO segmental model with explicit duration model ?

} // namesapce

#endif

