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
#include "StatePosteriorFeatureScorer.hh"
#include "Module.hh"

using namespace Mm;

StatePosteriorFeatureScorer::CachedStatePosteriorContextScorer::CachedStatePosteriorContextScorer(
        Core::Ref<const Feature> feature,
        const StatePosteriorFeatureScorer *featureScorer,
        size_t cacheSize)
:
        CachedAssigningFeatureScorer::CachedAssigningContextScorer(featureScorer, cacheSize),
        feature_(feature),
    minimumIndex_(Core::Type<DensityIndex>::max),
    minimumScore_(Core::Type<Weight>::max),
        scale_(1),
        initialize_(none)
{
    //    require(featureVector.size() == featureScorer->dimension());
}

void StatePosteriorFeatureScorer::CachedStatePosteriorContextScorer::_workDensityScores(
    MixtureIndex marginMixture) const
{
    if (!filter_) {
        return;
    }
    AssigningFeatureScorer::AssigningScorer scorer = featureScorer()->fs_->getAssigningScorer(feature_);
    const bool useViterbi = featureScorer()->useViterbi();
    Weight minScore = Core::Type<Weight>::max;
    for (Filter::const_iterator sIt = filter_->begin(); sIt != filter_->end(); ++ sIt) {
        const MixtureIndex mix = sIt->first;
        const Topology &topo = topology(mix);
        const Weight prior = sIt->second;
        if (useViterbi) {
            const Weight score = prior + scale_ * scorer->score(mix);
            const DensityIndex dns = topo[scorer->bestDensity(mix)];
            scores_[dns] = score;
            if (mix == marginMixture) {
                scores_[dns] += featureScorer()->margin_;
            }
            if (score < minScore) {
                minScore = score;
                minimumIndex_ = dns;
            }
        } else {
            for (DensityIndex dnsInMix = 0; dnsInMix < topo.size(); ++ dnsInMix) {
                const Weight score = scale_ * scorer->score(mix, dnsInMix) + prior;
                const DensityIndex dns = topo[dnsInMix];
                scores_[dns] = score;
                if (mix == marginMixture) {
                    scores_[dns] += featureScorer()->margin_;
                }
                if (score < minScore) {
                    minScore = score;
                    minimumIndex_ = dns;
                }
            }
        }
    }
    minimumScore_ = minScore;

    Core::XmlChannel &statistics = featureScorer()->statisticsChannel_;
    if (statistics.isOpen()) {
        statistics << Core::XmlOpen("prior")
        + Core::XmlAttribute("n-mixtures", featureScorer()->nMixtures());
        for (Filter::const_iterator sIt = filter_->begin(); sIt != filter_->end(); ++ sIt) {
            statistics << "(" << sIt->first << "," << sIt->second << ") ";
        }
        statistics << Core::XmlClose("prior");
    }
}

void StatePosteriorFeatureScorer::CachedStatePosteriorContextScorer::workMixtureScores() const
{
    if (filter_) {
        AssigningFeatureScorer::AssigningScorer scorer = featureScorer()->fs_->getAssigningScorer(feature_);
        const bool useViterbi = featureScorer()->useViterbi();
        Weight minScore = Core::Type<Weight>::max;
        for (Filter::const_iterator sIt = filter_->begin(); sIt != filter_->end(); ++ sIt) {
            const MixtureIndex mix = sIt->first;
            const Weight prior = sIt->second;
            if (useViterbi) {
                scores_[mix] = prior + scale_ * scorer->score(mix);
                if (scores_[mix] < minScore) {
                    minScore = scores_[mix];
                    minimumIndex_ = mix;
                }
            } else {
                require(useViterbi);
            }
        }
    }
    initialize_ |= mixtureScore;
}

void StatePosteriorFeatureScorer::CachedStatePosteriorContextScorer::pruneScores() const
{
    Weight threshold = featureScorer()->pruningThreshold_;
    if (threshold < Core::Type<Weight>::max) {
        threshold += minimumScore();
        ScoresAndDensities scores;
        PosteriorsAndDensities::iterator it = scores_.begin();
        for (; it != scores_.end(); ++ it) {
            if (it->second < threshold) {
                scores[it->first] = it->second;
            }
        }
        scores_ = scores;
    }

    Core::XmlChannel &statistics = featureScorer()->statisticsChannel_;
    if (statistics.isOpen()) {
        statistics << Core::XmlOpen("s")
        + Core::XmlAttribute("n-active-densities", scores_.size())
        + Core::XmlAttribute("absolute-pruning-threshold", threshold);
        PosteriorsAndDensities::const_iterator it = scores_.begin();
        for (; it != scores_.end(); ++ it) {
            statistics << "(" << it->first << "," << it->second << ") ";
        }
        statistics << Core::XmlClose("s");
    }
}

void StatePosteriorFeatureScorer::CachedStatePosteriorContextScorer::workPosteriors() const
{
    const Weight minScore = minimumScore();
    f64 sum = 0;
    PosteriorsAndDensities::const_iterator minIt = scores_.find(minimumIndex_);
    for (PosteriorsAndDensities::const_iterator it = scores_.begin(); it != scores_.end(); ++ it) {
        p_[it->first] = minScore - it->second;
        if (it != minIt) {
            sum += exp(p_[it->first]);
        }
    }
    Weight scaledLogZ = log1p(sum);
    for (PosteriorsAndDensities::iterator it = p_.begin(); it != p_.end(); ++ it) {
        it->second = exp(it->second - scaledLogZ);
    }
    logZ_ = scaledLogZ - minScore;

    Core::XmlChannel &statistics = featureScorer()->statisticsChannel_;
    if (statistics.isOpen()) {
        Weight sum = 0;
        PosteriorsAndDensities::const_iterator it = p_.begin();
        for (; it != p_.end(); ++ it) {
            sum += it->second;
        }
        statistics << Core::XmlOpen("p")
        + Core::XmlAttribute("n-active-densities", p_.size())
        + Core::XmlAttribute("log-partition-function", logZ_)
        + Core::XmlAttribute("total-probability", sum);

        for (it = p_.begin(); it != p_.end(); ++ it) {
            statistics << "(" << it->first << "," << it->second << ") ";
        }
        statistics << Core::XmlClose("p");
    }
}

void StatePosteriorFeatureScorer::CachedStatePosteriorContextScorer::workLikelihoods() const
{
    for (PosteriorsAndDensities::const_iterator it = scores_.begin(); it != scores_.end(); ++ it) {
        p_[it->first] = exp(-it->second);
    }
    Core::XmlChannel &statistics = featureScorer()->statisticsChannel_;
    if (statistics.isOpen()) {
        statistics << Core::XmlOpen("p")
        + Core::XmlAttribute("n-active-densities", p_.size());
        for (PosteriorsAndDensities::const_iterator it = p_.begin(); it != p_.end(); ++ it) {
            statistics << "(" << it->first << "," << it->second << ") ";
        }
        statistics << Core::XmlClose("p");
    }
}


void StatePosteriorFeatureScorer::CachedStatePosteriorContextScorer::workDensityScores(
        MixtureIndex marginMixture) const
{
    reset();
    _workDensityScores(marginMixture);
    pruneScores();
    initialize_ |= densityScore;
}

void StatePosteriorFeatureScorer::CachedStatePosteriorContextScorer::workDensityPosteriors(
        MixtureIndex marginMixture) const
{
    if (!(initialize_ & densityScore)) {
        workDensityScores(marginMixture);
    }
    workPosteriors();
    initialize_ |= densityPosterior;
}

void StatePosteriorFeatureScorer::CachedStatePosteriorContextScorer::workMixturePosteriors() const
{
    reset();
    workMixtureScores();
    pruneScores();
    workPosteriors();
    initialize_ |= mixture;
}

void StatePosteriorFeatureScorer::CachedStatePosteriorContextScorer::workMixtureLikelihoods() const
{
    reset();
    workMixtureScores();
    pruneScores();
    workLikelihoods();
    initialize_ |= mixture;
}


void StatePosteriorFeatureScorer::CachedStatePosteriorContextScorer::reset() const
{
    scores_.clear();
    p_.clear();
    minimumIndex_ = Core::Type<DensityIndex>::max;
    minimumScore_ = Core::Type<Weight>::max;
    initialize_ = none;
}

const StatePosteriorFeatureScorer::CachedStatePosteriorContextScorer::ScoresAndDensities&
StatePosteriorFeatureScorer::CachedStatePosteriorContextScorer::scoresAndDensities() const
{
    if (!(initialize_ & densityScore)) {
        workDensityScores();
    }
    return scores_;
}

const StatePosteriorFeatureScorer::CachedStatePosteriorContextScorer::ScoresAndMixtures&
StatePosteriorFeatureScorer::CachedStatePosteriorContextScorer::scoresAndMixtures() const
{
    if (!(initialize_ & mixtureScore)) {
        workMixtureScores();
        pruneScores();
    }
    return scores_;
}


const StatePosteriorFeatureScorer::PosteriorsAndDensities&
StatePosteriorFeatureScorer::CachedStatePosteriorContextScorer::posteriorsAndDensities(MixtureIndex marginMixture) const
{
    if (!(initialize_ & densityPosterior)) {
        workDensityPosteriors(marginMixture);
    }
    return p_;
}

const StatePosteriorFeatureScorer::PosteriorsAndMixtures&
StatePosteriorFeatureScorer::CachedStatePosteriorContextScorer::posteriorsAndMixtures() const
{
    if (!(initialize_ & mixture)) {
        workMixturePosteriors();
    }
    return p_;
}

const StatePosteriorFeatureScorer::PosteriorsAndMixtures&
StatePosteriorFeatureScorer::CachedStatePosteriorContextScorer::likelihoodAndMixtures() const
{
    if (!(initialize_ & mixture)) {
        workMixtureLikelihoods();
    }
    return p_;
}

void StatePosteriorFeatureScorer::CachedStatePosteriorContextScorer::posteriorsAndMixtures(
        IndicesAndWeights &priors) const
{
    AssigningFeatureScorer::AssigningScorer scorer = featureScorer()->fs_->getAssigningScorer(feature_);
    require(featureScorer()->useViterbi());
    Mm::MixtureIndex minIndex = invalidMixture;
    Weight minScore = Core::Type<Weight>::max;
    std::vector<IndexAndWeight> &scores = priors;
    for (u32 i = 0; i < priors.size(); ++ i) {
        scores[i].w = priors[i].w + scale_ * scorer->score(priors[i].e);
        if (scores[i].w < minScore) {
            minScore = scores[i].w;
            minIndex = i;
        }
    }

    f64 sum = 0;
    for (u32 i = 0; i < scores.size(); ++ i) {
        scores[i].w = minScore - scores[i].w;
        if (i != minIndex) {
            sum += exp(scores[i].w);
        }
    }
    const Weight logZ = log1p(sum);
    std::vector<IndexAndWeight> &p = scores;
    for (u32 i = 0; i < p.size(); ++ i) {
        p[i].w = exp(scores[i].w - logZ);
    }
}

/**
 * StatePosteriorFeatureScorer
 */
const Core::ParameterFloat StatePosteriorFeatureScorer::paramPruningThreshold(
        "pruning-threshold",
    "densities with scores higher than the minimum score plus this threshold are pruned",
        Core::Type<Weight>::max, 0);

const Core::ParameterBool StatePosteriorFeatureScorer::paramViterbi(
        "viterbi",
        "only the best density of a mixture is considered",
        true);

const Core::ParameterFloat StatePosteriorFeatureScorer::paramScale(
        "scale",
        "scaling factor for scores from the feature scorer",
        1,
        0);

const Core::ParameterFloat StatePosteriorFeatureScorer::paramMargin(
        "margin",
        "margin, i.e., score of correct mixture index is reduced by the margin",
        0,
        0);

const Core::ParameterIntVector StatePosteriorFeatureScorer::paramDisregardDensities(
        "disregard-densities", "list of densitiy indices that are disregarded in posterior calculation, e.g. states of mul-phoneme",
        ",", 0);

StatePosteriorFeatureScorer::StatePosteriorFeatureScorer(
        const Core::Configuration &c)
:
        Core::Component(c),
        Precursor(c),
        scale_(paramScale(config)),
        pruningThreshold_(paramPruningThreshold(config)),
        viterbi_(paramViterbi(config)),
        statisticsChannel_(config, "statistics"),
        margin_(paramMargin(config)),
        disregardDensities_(paramDisregardDensities(config))
{}

StatePosteriorFeatureScorer::StatePosteriorFeatureScorer(
        const Core::Configuration &c,
        Core::Ref<const AbstractMixtureSet> mixtureSet)
:
        Core::Component(c),
        Precursor(c),
        scale_(paramScale(config)),
        pruningThreshold_(paramPruningThreshold(config)),
        viterbi_(paramViterbi(config)),
        statisticsChannel_(config, "statistics"),
        margin_(paramMargin(config)),
        disregardDensities_(paramDisregardDensities(config))
{
    setFeatureScorer(
            Mm::Module::instance().createAssigningFeatureScorer(
                    select("feature-scorer"), mixtureSet));
}

AssigningFeatureScorer::ScoreAndBestDensity
StatePosteriorFeatureScorer::calculateScoreAndDensity(
        const CachedAssigningFeatureScorer::CachedAssigningContextScorer* cs, MixtureIndex mixtureIndex) const
{
    AssigningFeatureScorer::ScoreAndBestDensity result;
    result.score = 0;
    result.bestDensity = Core::Type<size_t>::max;
    const CachedStatePosteriorContextScorer *c = required_cast(const CachedStatePosteriorContextScorer*, cs);
    require(c->featureScorer()->useViterbi());
    const PosteriorsAndDensities &posteriors = c->posteriorsAndDensities();
    const Topology &topo = topology(mixtureIndex);
    for (DensityIndex dnsInMix = 0; dnsInMix < topo.size(); ++ dnsInMix) {
        if (posteriors.find(topo[dnsInMix]) != posteriors.end()) {
            result.score = posteriors.find(topo[dnsInMix])->second;
            result.bestDensity = dnsInMix;
            break;
        }
    }
    return result;
}

/*
 * DefaultFilter
 */
class DefaultFilter : public StatePosteriorFeatureScorer::Filter
{
public:
    DefaultFilter(MixtureIndex nMixtures) {
        for (MixtureIndex mix = 0; mix < nMixtures; ++ mix) {
            map_[mix] = 0;
        }
    }
};

StatePosteriorFeatureScorer::FilterRef StatePosteriorFeatureScorer::defaultFilter() const
{
    if (!defaultFilter_) {
        defaultFilter_ = Core::ref(new DefaultFilter(nMixtures()));
    }
    return defaultFilter_;
}

/*
 * SingleMixtureFilter
 */
class SingleMixtureFilter : public StatePosteriorFeatureScorer::Filter
{
public:
    SingleMixtureFilter(MixtureIndex mixtureIndex) {
        map_[mixtureIndex] = 0;
    }
};

AssigningFeatureScorer::AssigningScorer StatePosteriorFeatureScorer::getAssigningScorer(
        Core::Ref<const Feature> feature) const
{
    Core::Ref<CachedStatePosteriorContextScorer> scorer(
            new CachedStatePosteriorContextScorer(
                    feature, this, nMixtures()));
    scorer->setFilter(filter_,disregardDensities_);
    scorer->setScale(scale_);
    return AssigningFeatureScorer::AssigningScorer(scorer);
}

AssigningFeatureScorer::AssigningScorer StatePosteriorFeatureScorer::getAssigningScorer(
        const FeatureVector &featureVector) const
{
    Core::Ref<const Feature> feature(new Feature(featureVector));
    return getAssigningScorer(feature);
}

void StatePosteriorFeatureScorer::setFilter(const MixtureIndex &mixtureIndex)
{
    this->setFilter(FilterRef(new SingleMixtureFilter(mixtureIndex)));
}
