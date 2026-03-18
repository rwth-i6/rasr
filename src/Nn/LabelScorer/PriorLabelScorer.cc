#include "PriorLabelScorer.hh"

#include "ScoreAccessor.hh"
#include "ScoringContext.hh"
#include "TransitionTypes.hh"

namespace {

class PriorScoreAccessor : public Nn::ScoreAccessor {
public:
    using Score = Nn::Score;

    PriorScoreAccessor(Core::Ref<ScoreAccessor> scoreAccessor, bool negateOutput, std::shared_ptr<Nn::Prior<Score>> prior)
        : scoreAccessor_(scoreAccessor), negateOutput_(negateOutput), prior_(prior) {}

    virtual Score getScore(Nn::TransitionType transitionType, Nn::LabelIndex labelIndex = Nn::invalidLabelIndex) const {
        Score score = scoreAccessor_->getScore(transitionType, labelIndex); 
        if (negateOutput_) {
            score = -score;
        }
        if (prior_->scale() != 0.0) {
            score += prior_->at(labelIndex) * prior_->scale();
        }
        return score;
    }

    virtual Nn::TimeframeIndex getTime() const {
        return scoreAccessor_->getTime();
    }

private:
    Core::Ref<ScoreAccessor>          scoreAccessor_;
    const bool                        negateOutput_;
    std::shared_ptr<Nn::Prior<Score>> prior_;
};

using PriorScoreAccessorRef = Core::Ref<PriorScoreAccessor> ;

}  // namespace

namespace Nn {

const Core::ParameterBool PriorLabelScorer::paramNegateOutput("negate-output", "wether to negate the scores obtained from Score/DataViewMessages", false);

PriorLabelScorer::PriorLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config),
          negateOutput_(paramNegateOutput(config)),
          prior_(new Nn::Prior<Score>(config)) {
    if (prior_->scale() != 0.0) {
        prior_->read();
    }
}

std::optional<ScoreAccessorRef> PriorLabelScorer::getScoreAccessor(ScoringContextRef scoringContext) {
    auto res = StepwiseNoOpLabelScorer::getScoreAccessor(scoringContext);
    if (res) {
        return Core::ref(new PriorScoreAccessor(*res, negateOutput_, prior_));
    }
    return res;
}

}  // namespace Nn
