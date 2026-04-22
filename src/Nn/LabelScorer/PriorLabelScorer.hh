#ifndef PRIOR_LABEL_SCORER_HH
#define PRIOR_LABEL_SCORER_HH

#include <Nn/Prior.hh>

#include "NoOpLabelScorer.hh"

namespace Nn {

/*
 * Label Scorer that assumes the input features are the output of a (negative) log_softmax layer. It optionally
 * negates the output then subtracts a prior.
 *
 * This is useful for example when the scores are computed externally.
 */
class PriorLabelScorer : public StepwiseNoOpLabelScorer {
public:
    using Precursor = StepwiseNoOpLabelScorer;

    static const Core::ParameterBool paramNegateOutput;

    PriorLabelScorer(const Core::Configuration& config);

    // Gets a prior-corrected accessor for the buffered scores at the requested step
    std::optional<ScoreAccessorRef> getScoreAccessor(ScoringContextRef scoringContext) override;

private:
    const bool negateOutput_;

    std::shared_ptr<Nn::Prior<Score>> prior_;
};

}  // namespace Nn

#endif  // PRIOR_LABEL_SCORER_HH
