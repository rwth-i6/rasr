#include "PriorLabelScorer.hh"

#include "ScoreAccessor.hh"
#include "ScoringContext.hh"
#include "TransitionTypes.hh"

namespace {

class PriorScoreAccessor : public Nn::ScoreAccessor {
public:
    using Score = Nn::Score;

    PriorScoreAccessor(Core::Ref<ScoreAccessor> scoreAccessor, bool negateInput, std::shared_ptr<Nn::Prior<Score>> prior)
            : scoreAccessor_(scoreAccessor), negateInput_(negateInput), prior_(prior) {}

    virtual Score getScore(Nn::TransitionType transitionType, Nn::LabelIndex labelIndex = Nn::invalidLabelIndex) const override {
        Score score = scoreAccessor_->getScore(transitionType, labelIndex);
        if (negateInput_) {
            score = -score;
        }
        if (prior_->scale() != 0.0) {
            // The Prior class returns scores in +log prob space
            // Thus add it here to the -log prob space score
            score += prior_->at(labelIndex) * prior_->scale();
        }
        return score;
    }

    std::optional<Nn::DenseScoreSpan> getDenseScores() const override {
        auto denseScores = scoreAccessor_->getDenseScores();
        if (not denseScores) {
            return std::nullopt;
        }

        if (negateInput_) {
            for (auto& term : denseScores->terms) {
                term.scale *= -1.0;
            }
        }

        if (prior_->scale() != 0.0) {
            require(denseScores->size() == prior_->size());  // Prior size must match base scorer vocab size
            std::span<Score const> priorScores(&prior_->at(0), prior_->size());
            denseScores->terms.push_back(Nn::DenseScoreTerm{.scores = priorScores, .scale = prior_->scale()});
        }

        return denseScores;
    }

    virtual Nn::TimeframeIndex getTime() const override {
        return scoreAccessor_->getTime();
    }

private:
    Core::Ref<ScoreAccessor>          scoreAccessor_;
    const bool                        negateInput_;
    std::shared_ptr<Nn::Prior<Score>> prior_;
};

using PriorScoreAccessorRef = Core::Ref<PriorScoreAccessor>;

}  // namespace

namespace Nn {

const Core::ParameterBool PriorLabelScorer::paramNegateInput("negate-input", "whether to negate the scores obtained from Score/DataViewMessages", false);

PriorLabelScorer::PriorLabelScorer(Core::Configuration const& config)
        : Core::Component(config),
          Precursor(config),
          negateInput_(paramNegateInput(config)),
          prior_(new Nn::Prior<Score>(config)) {
    if (prior_->scale() != 0.0) {
        prior_->read();
    }
}

std::optional<ScoreAccessorRef> PriorLabelScorer::getScoreAccessor(ScoringContextRef scoringContext) {
    auto res = StepwiseNoOpLabelScorer::getScoreAccessor(scoringContext);
    if (res) {
        return Core::ref(new PriorScoreAccessor(*res, negateInput_, prior_));
    }
    return res;
}

}  // namespace Nn
