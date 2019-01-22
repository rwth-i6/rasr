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
} // namespace Flf

#endif // _FLF_PUSH_FORWARD_RESCORING_HH
