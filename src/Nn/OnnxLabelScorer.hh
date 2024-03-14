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

#ifndef ONNX_LABEL_SCORER_HH
#define ONNX_LABEL_SCORER_HH

#include <Onnx/IOSpecification.hh>
#include <Onnx/Session.hh>
#include <Onnx/Value.hh>
#include "LabelScorer.hh"

#include "TFLabelScorer.hh"  // this is for the NgramLabelHistory pls adjust later ??????

namespace Nn {

typedef std::vector<Onnx::Value> ValueList;
typedef std::vector<std::pair<std::string, Onnx::Value>> MappedValueList;

struct OnnxLabelHistory : public LabelHistoryBase {
    std::vector<Score> scores;
    std::shared_ptr<ValueList> variables;
    u32  position;
    bool isBlank;  // for next feedback

    typedef LabelHistoryBase Precursor;

    OnnxLabelHistory()
            : Precursor(), variables(nullptr), position(0), isBlank(false) {}
    OnnxLabelHistory(const OnnxLabelHistory& ref) :
            Precursor(ref), scores(ref.scores), variables(ref.variables),
            position(ref.position), isBlank(ref.isBlank) {}
};

// Encoder-Decoder Label Scorer based on Onnx back-end
class OnnxModelBase : public LabelScorer {
    typedef LabelScorer Precursor;

public:
    // config params for graph computation
    static const Core::ParameterBool paramTransformOuputLog;
    static const Core::ParameterBool paramTransformOuputNegate;
    static const Core::ParameterInt  paramMaxBatchSize;

    // overwrite descriptor in derived class for specific history
    typedef OnnxLabelHistory LabelHistoryDescriptor;

public:
    std::vector<f32> scores;
    OnnxModelBase(const Core::Configuration& config);
    virtual ~OnnxModelBase();

    virtual void reset();
    virtual void cleanUpBeforeExtension(u32 minPos) { cacheHashQueue_.clear(); }

    // history handling
    virtual LabelHistory startHistory();
    virtual void         extendLabelHistory(LabelHistory& h, LabelIndex idx, u32 position, bool isLoop);

    // encoding

    ValueList encoder_outputs; //????
    virtual void encode();

    // get scores for the next output position
    virtual const std::vector<Score>& getScores(const LabelHistory& h, bool isLoop);

protected:
    void initStartHistory();
    void loadPrior();

    // ---- batch-wise graph computation ----
    virtual void initComputation();

    virtual void makeBatch(LabelHistoryDescriptor* targetLhd);
    virtual void decodeBatch();

    //virtual void feedBatchVariables();
    virtual void feedDecodeInput(MappedValueList& inputs);
    // virtual void fetchBatchVariables();

    virtual void addPriorToBatch();
    //virtual void computeBatchScores();
    virtual void processBatchOutput(const ValueList& decoder_outputs);
    // --------------------------------------

protected:
    // onnx related members
    Onnx::Session                                   encoderSession_;
    static const std::vector<Onnx::IOSpecification> encoderIoSpec_;  // currently fixed to "features", "feature-size" and "output"
    const Onnx::IOMapping                           encoderMapping_;

    Onnx::Session                                   decoderSession_;
    static const std::vector<Onnx::IOSpecification> decoderIoSpec_;  // currently fixed to "features", "feature-size" and "output"
    const Onnx::IOMapping                           decoderMapping_;

    Onnx::IOValidator                               validator_;

    // session-run related members
    const std::string              encoder_features_name_;
    const std::string              encoder_features_size_name_;
    const std::string              encoder_output_name_;
    const std::string              encoder_output_size_name_;

    const std::string              decoder_input_name_;
    const std::string              decoder_input_size_name_;
    const std::string              decoder_hidden_input_name_;
    const std::string              decoder_feedback_name_;
    const std::string              decoder_hidden_output_name_;
    const std::string              decoder_output_name_;

    // binary function including scaling
    std::function<Score(Score, Score)> decoding_output_transform_function_;

protected:
    LabelHistoryDescriptor* startHistoryDescriptor_; // only common stuff, no states or scores

    typedef std::vector<LabelHistoryDescriptor*> Batch;
    Batch batch_;
    std::deque<size_t> cacheHashQueue_;
    u32                maxBatchSize_;

    typedef std::unordered_map<size_t, std::vector<Score>> ScoreCache;
    ScoreCache                                             contextLogPriors_;
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
    class OnnxFfnnTransducer : public OnnxModelBase {
        typedef OnnxModelBase Precursor;
        typedef NgramLabelHistory LabelHistoryDescriptor;

    public:
        static const Core::ParameterInt paramContextSize;
        static const Core::ParameterBool paramCacheHistory;
        static const Core::ParameterBool paramImplicitTransition;
        static const Core::ParameterBool paramExplicitTransition;
        static const Core::ParameterBool paramUseRelativePosition;
        static const Core::ParameterBool paramRenormTransition;

    public:
        OnnxFfnnTransducer(Core::Configuration const& config);
        ~OnnxFfnnTransducer();

        void reset();
        void cleanUpBeforeExtension(u32 minPos);

        bool useRelativePosition() const { return useRelativePosition_; }

        // history handling
        LabelHistory startHistory();
        void extendLabelHistory(LabelHistory& h, LabelIndex idx, u32 position, bool isLoop);

//        // global position of encodings
//        void increaseDecodeStep();

        // get label scores for the next output position
        const std::vector<Score>& getScores(const LabelHistory& h, bool isLoop);

        // get segment scores for the next label segment given start position
        const SegmentScore& getSegmentScores(const LabelHistory& h, LabelIndex segIdx, u32 startPos);

    protected:
        void initComputation() {}
        void makeBatch(LabelHistoryDescriptor* targetLhd);
        void decodeBatch(ScoreCache& scoreCache);
        void computeBatchScores(ScoreCache& scoreCache, MappedValueList& inputs);
//        void setDecodePosition(u32 pos);

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

}  // namespace Nn

#endif
