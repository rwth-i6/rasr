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
#ifndef _FLF_PUSH_FORWARD_RESCORING_HH
#define _FLF_PUSH_FORWARD_RESCORING_HH

#include <Lm/LanguageModel.hh>

#include "FlfCore/Lattice.hh"
#include "RescoreInternal.hh"

namespace Flf {
class PushForwardRescorer : public Core::Component {
public:
    typedef Core::Component Precursor;

    enum RescorerType {
        singleBestRescoringType,
        replacementApproximationRescoringType,
        tracebackApproximationRescoringType
    };

    static const Core::Choice          rescorerTypeChoice;
    static const Core::ParameterChoice paramRescorerType;
    static const Core::ParameterInt    paramMaxHypothesis;
    static const Core::ParameterFloat  paramPruningThreshold;
    static const Core::ParameterInt    paramHistoryLimit;
    static const Core::ParameterFloat  paramLookaheadScale;
    static const Core::ParameterBool   paramDelayedRescoring;
    static const Core::ParameterInt    paramDelayedRescoringMaxHyps;

    PushForwardRescorer(Core::Configuration const& config, Core::Ref<Lm::LanguageModel> lm);
    ~PushForwardRescorer() = default;

    virtual ConstLatticeRef rescore(ConstLatticeRef l, ScoreId id);

private:
    Core::Ref<Lm::LanguageModel> lm_;
    RescorerType                 rescoring_type_;
    unsigned                     max_hyps_;
    Flf::Score                   pruning_threshold_;
    unsigned                     history_limit_;
    Flf::Score                   lookahead_scale_;
    bool                         delayed_rescoring_;
    unsigned                     delayed_rescoring_max_hyps_;
};

class PushForwardRescoringNode : public RescoreSingleDimensionNode {
public:
    typedef RescoreSingleDimensionNode Precursor;

    PushForwardRescoringNode(std::string const& name, Core::Configuration const& config);
    virtual ~PushForwardRescoringNode() = default;

    virtual void init(std::vector<std::string> const& arguments);

    virtual void sync() {
        rescored_lattice_.reset();
    }

protected:
    virtual ConstLatticeRef rescore(ConstLatticeRef l, ScoreId id);

private:
    std::unique_ptr<PushForwardRescorer> rescorer_;

    ConstLatticeRef rescored_lattice_;
};

NodeRef createPushForwardRescoringNode(std::string const& name, Core::Configuration const& config);
}  // namespace Flf

#endif  // _FLF_PUSH_FORWARD_RESCORING_HH
