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

#ifndef LABEL_SCORER_HH
#define LABEL_SCORER_HH

#include <Core/ReferenceCounting.hh>
#include <Math/FastMatrix.hh>
#include <Speech/Feature.hh>
#include "LabelHistoryManager.hh"

namespace Nn {

typedef Search::Score Score;
typedef std::vector<std::pair<u32,Score>> SegmentScore;
typedef std::unordered_map<std::string, LabelIndex> LabelIndexMap;

// base class of models for label scoring (basic supports except scoring)
class LabelScorer : public virtual Core::Component, 
                    public Core::ReferenceCounted {
  public:
    // config params
    static const Core::ParameterString paramLabelFile;
    static const Core::ParameterInt paramNumOfClasses;
    static const Core::ParameterBool paramPreComputeEncoding;
    static const Core::ParameterInt paramBufferSize;
    static const Core::ParameterFloat paramScale;
    static const Core::ParameterBool paramUsePrior;
    static const Core::ParameterInt paramPriorContextSize;
    static const Core::ParameterBool paramLoopUpdateHistory;
    static const Core::ParameterBool paramBlankUpdateHistory;
    static const Core::ParameterBool paramPositionDependent;
    static const Core::ParameterIntVector paramReductionFactors;
    static const Core::ParameterBool paramUseStartLabel;
    static const Core::ParameterFloat paramSegmentLengthScale;
    static const Core::ParameterInt paramMinSegmentLength;
    static const Core::ParameterInt paramMaxSegmentLength;

  public:
    LabelScorer(const Core::Configuration&);
    virtual ~LabelScorer() { delete labelHistoryManager_; }

    const Core::Dependency& getDependency() const { return dependency_; }

    virtual void reset();
    virtual void cleanUpBeforeExtension(u32 minPos) {} // each search step

    // labels
    LabelIndex numClasses() const { return numClasses_; }
    const LabelIndexMap& getLabelIndexMap();
    // special labels: either in the vocab file or configurable (hard-coded naming)
    LabelIndex getBlankLabelIndex() const { return getSpecialLabelIndex("<blank>", "blank-label-index"); }
    LabelIndex getStartLabelIndex() const { return getSpecialLabelIndex("<s>", "start-label-index"); }
    LabelIndex getEndLabelIndex() const { return getSpecialLabelIndex("</s>", "end-label-index"); }
    LabelIndex getUnknownLabelIndex() const { return getSpecialLabelIndex("<unk>", "unknown-label-index"); }
    LabelIndex getNoContextLabelIndex() const;

    // special flags for various models, e.g. attention, segmental, RNN-T
    bool needEndProcess() const { return needEndProcessing_ || isPositionDependent_; }
    bool isPositionDependent() const { return isPositionDependent_; }
    virtual bool useRelativePosition() const { return false; }
    virtual bool useVerticalTransition() const { return false; }

    // inputs
    virtual void addInput(Core::Ref<const Speech::Feature> f) {
      inputBuffer_.emplace_back( *(f->mainStream().get()) );
      ++nInput_;
    }
    virtual void clearBuffer() {
        inputBuffer_.clear();
        decodeStep_ = 0;
    }
    u32 bufferSize() const { return inputBuffer_.size(); }
    virtual bool bufferFilled() const { return eos_ || inputBuffer_.size() >= bufferSize_; }
    void setEOS() { eos_ = true; }
    bool reachEOS() const { return eos_; }

    virtual void increaseDecodeStep() { ++decodeStep_; }
    // stopping criteria
    // - needEndProcessing_: stop by search (additional max input length stop)
    // - time synchronous: stop by decodeStep reach end
    virtual bool reachEnd() const;
    virtual bool maybeFinalSegment(u32 startPos) const;

    // naming after encoder-decoder framework, but can also be beyond
    virtual void encode() {}
    virtual u32 getEncoderLength() const;

    // ---- label history and scores ----
    virtual bool isHistoryDependent() const { return true; }
    virtual bool loopUpdateHistory() const { return loopUpdateHistory_; }
    virtual bool blankUpdateHistory() const { return blankUpdateHistory_; }

    // start up label history (to be overwritten)
    virtual LabelHistory startHistory() = 0;

    // extend history and possibly update caching (to be overwritten)
    virtual void extendLabelHistory(LabelHistory& h, LabelIndex idx, u32 position, bool isLoop) = 0;

    // get label scores for the next output position (to be overwritten)
    virtual const std::vector<Score>& getScores(const LabelHistory& h, bool isLoop) = 0;

    // get segment scores for the next label segment given start position
    virtual const SegmentScore& getSegmentScores(const LabelHistory& h, LabelIndex segId,
                                                 u32 startPos) { return segmentScore_; }
    // ---------------------------

    static Score logSumExp(const std::vector<Score>& scores);
    static Score computeScoreSum(const std::vector<Score>& scores);

  protected:
    void init(); // Note: not virtual
    LabelIndex getSpecialLabelIndex(const std::string&, const std::string&) const;
    u32 getReducedLength(u32 len) const; // input length after possible downsampling

  protected:
    LabelHistoryManager* labelHistoryManager_;
    Core::Dependency dependency_;

    std::vector<std::vector<f32> > inputBuffer_; // hard coded Mm::FeatureType = f32
    u32 nInput_; // total number of inputs
    std::vector<int> redFactors_; // input (time) reduction factors
    bool eos_; // end of input stream

    f32 scale_;
    LabelIndex numClasses_;

    // prior for model bias correction
    bool usePrior_;
    u32 priorContextSize_;
    std::vector<f32> logPriors_; // context-independent prior

    bool loopUpdateHistory_;
    bool blankUpdateHistory_;
    bool needEndProcessing_;
    bool isPositionDependent_;

    bool useStartLabel_;
    LabelIndex startLabelIndex_;
    s32 startPosition_;
    u32 decodeStep_; // global decoding step

    // for segmental decoding
    SegmentScore segmentScore_;
    f32 segLenScale_;
    u32 minSegLen_;
    u32 maxSegLen_; // speech only

  private:
    LabelIndexMap labelIndexMap_;
    u32 bufferSize_; // maximum number for input frames
};

// posteriors computed beforehand, e.g. front-end forwarding
// - compatible with any 0-order (or + simple TDP) time-synchronized model (hybrid, ctc, etc.)
// - also support 1st-order model as cached scores for all context (vocab^2)
class PrecomputedScorer : public LabelScorer {
    typedef LabelScorer Precursor;
    typedef LabelHistoryBase LabelHistoryDescriptor;

  public:
    static const Core::ParameterBool paramFirstOrder;

  public:
    PrecomputedScorer(const Core::Configuration&);

    // input log posterior scores
    void addInput(Core::Ref<const Speech::Feature> f);

    // no or 1st-order history
    bool isHistoryDependent() const { return firstOrder_; }
    LabelHistory startHistory();
    void extendLabelHistory(LabelHistory& h, LabelIndex idx, u32 position, bool isLoop);

    // get label scores for the next output position
    const std::vector<Score>& getScores(const LabelHistory& h, bool isLoop);

    void cleanUpBeforeExtension(u32 minPos);

  private:
    LabelHistoryDescriptor* getHistory(LabelIndex idx);

  private:
    bool firstOrder_;
    std::vector<std::vector<Score> >     cachedScore_; // avoid redundant copy
    std::vector<LabelHistoryDescriptor*> cachedHistory_; // quick access

    LabelIndex blankLabelIndex_;
};

} // namespace

#endif
